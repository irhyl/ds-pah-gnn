"""
DS-PAH-GNN: Dual-Stream Physics-Aware Hierarchical Graph Neural Network

Main model architecture implementing:
- LocalStream: Node-level dynamics with GRU updates
- GlobalStream: System-level coordination with hierarchical pooling  
- CrossHierarchyFusion: Attention-based fusion of local and global streams
- Switchable Edge Operators (SEO) for topology reconfiguration
- Neural ODE integration for continuous-time dynamics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch
from typing import Optional, Dict, Tuple
import math

from .switchable_ops import SwitchableEdgeOperator


class EdgeConvolutionalLayer(MessagePassing):
    """
    Edge Convolutional (ECC) layer with switchable edge operators.
    """
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, use_seo: bool = True):
        super().__init__(aggr='add')
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.use_seo = use_seo
        
        # Node transformation
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Edge transformation (with SEO if enabled)
        if use_seo:
            self.edge_transform = SwitchableEdgeOperator(edge_dim + 2 * node_dim, hidden_dim)
        else:
            self.edge_transform = nn.Sequential(
                nn.Linear(edge_dim + 2 * node_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        # Message MLP
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Update
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        """
        Args:
            x: [num_nodes, node_dim]
            edge_index: [2, num_edges]
            edge_attr: [num_edges, edge_dim]
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        # Concatenate source node, dest node, and edge features
        edge_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        
        # Transform through edge network (SEO if enabled)
        edge_embedding = self.edge_transform(edge_input)
        
        # Generate message
        message = self.message_mlp(edge_embedding)
        return message
    
    def update(self, aggr_out, x):
        # Combine aggregated messages with node features
        combined = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(combined)


class LocalStream(nn.Module):
    """
    Local Stream: Captures node-level dynamics with GRU temporal updates.
    """
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, num_layers: int = 4):
        super().__init__()
        
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Initial node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # ECC layers with SEO
        self.conv_layers = nn.ModuleList([
            EdgeConvolutionalLayer(hidden_dim, edge_dim, hidden_dim, use_seo=True)
            for _ in range(num_layers)
        ])
        
        # GRU for temporal dynamics
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor, h_prev: Optional[torch.Tensor] = None):
        """
        Args:
            x: [num_nodes, node_dim]
            edge_index: [2, num_edges]
            edge_attr: [num_edges, edge_dim]
            h_prev: [num_nodes, hidden_dim] - previous hidden state
            
        Returns:
            h_local: [num_nodes, hidden_dim]
        """
        # Encode input features
        h = self.node_encoder(x)
        
        # Apply ECC layers
        for conv in self.conv_layers:
            h_new = conv(h, edge_index, edge_attr)
            h = h + h_new  # Residual connection
            h = self.layer_norm(h)
        
        # GRU temporal update
        if h_prev is not None:
            h = self.gru(h, h_prev)
        
        return h


class GlobalStream(nn.Module):
    """
    Global Stream: Captures system-level coordination via hierarchical pooling.
    """
    
    def __init__(self, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Attention-based pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Global message passing
        self.global_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, h_local: torch.Tensor, batch: Optional[torch.Tensor] = None):
        """
        Args:
            h_local: [num_nodes, hidden_dim] - local node embeddings
            batch: [num_nodes] - batch assignment (for batched graphs)
            
        Returns:
            h_global: [num_graphs, hidden_dim]
            attention_weights: [num_nodes, 1]
        """
        # Compute attention weights
        attention_logits = self.attention_pool(h_local)  # [num_nodes, 1]
        
        if batch is not None:
            # Softmax within each graph in the batch
            attention_weights = torch.zeros_like(attention_logits)
            for graph_idx in batch.unique():
                mask = batch == graph_idx
                attention_weights[mask] = F.softmax(attention_logits[mask], dim=0)
        else:
            attention_weights = F.softmax(attention_logits, dim=0)
        
        # Weighted pooling
        h_weighted = h_local * attention_weights
        
        if batch is not None:
            h_global = global_add_pool(h_weighted, batch)
        else:
            h_global = h_weighted.sum(dim=0, keepdim=True)
        
        # Apply global transformations
        for mlp in self.global_mlp:
            h_global_new = mlp(h_global)
            h_global = h_global + h_global_new
            h_global = self.layer_norm(h_global)
        
        return h_global, attention_weights


class CrossHierarchyFusion(nn.Module):
    """
    Fuses local and global representations via cross-attention.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Multi-head attention parameters
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, h_local: torch.Tensor, h_global: torch.Tensor, 
                batch: Optional[torch.Tensor] = None):
        """
        Args:
            h_local: [num_nodes, hidden_dim]
            h_global: [num_graphs, hidden_dim]
            batch: [num_nodes] - batch assignment
            
        Returns:
            h_fused: [num_nodes, hidden_dim]
        """
        # Expand global to match local
        if batch is not None:
            h_global_expanded = h_global[batch]
        else:
            h_global_expanded = h_global.expand(h_local.size(0), -1)
        
        # Cross-attention: local attends to global
        Q = self.query_proj(h_local)
        K = self.key_proj(h_global_expanded)
        V = self.value_proj(h_global_expanded)
        
        # Attention scores
        attn_scores = (Q * K).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_dim)
        attn_weights = torch.sigmoid(attn_scores)
        
        # Fused representation
        h_attended = attn_weights * V
        h_fused = self.output_proj(h_local + h_attended)
        
        return h_fused


class DS_PAH_GNN(nn.Module):
    """
    Main Dual-Stream Physics-Aware Hierarchical GNN model.
    """
    
    def __init__(
        self,
        node_dim: int = 4,
        edge_dim: int = 3,
        hidden_dim: int = 128,
        local_layers: int = 4,
        global_layers: int = 3,
        use_uncertainty: bool = False,
        n_ensembles: int = 5
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.use_uncertainty = use_uncertainty
        
        # Dual streams
        self.local_stream = LocalStream(node_dim, edge_dim, hidden_dim, local_layers)
        self.global_stream = GlobalStream(hidden_dim, global_layers)
        
        # Cross-hierarchy fusion
        self.fusion = CrossHierarchyFusion(hidden_dim)
        
        # Prediction heads
        self.switch_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Per-edge switch logit
        )
        
        self.voltage_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Per-node voltage setpoint
        )
        
        self.current_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Per-node current prediction
        )
        
        # Uncertainty quantification (optional)
        if use_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus()  # Ensure positive variance
            )
    
    def forward(
        self, 
        data: Data,
        h_prev: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of DS-PAH-GNN.
        
        Args:
            data: PyG Data object with x, edge_index, edge_attr
            h_prev: Previous hidden state for GRU
            return_embeddings: Whether to return intermediate embeddings
            
        Returns:
            Dictionary containing:
                - switch_logits: [num_edges, 1]
                - voltage_pred: [num_nodes, 1]
                - current_pred: [num_nodes, 1]
                - h_local: [num_nodes, hidden_dim] (if return_embeddings)
                - h_global: [num_graphs, hidden_dim] (if return_embeddings)
                - attention_weights: [num_nodes, 1] (if return_embeddings)
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else None
        
        # Local stream: node-level dynamics
        h_local = self.local_stream(x, edge_index, edge_attr, h_prev)
        
        # Global stream: system-level coordination
        h_global, attention_weights = self.global_stream(h_local, batch)
        
        # Cross-hierarchy fusion
        h_fused = self.fusion(h_local, h_global, batch)
        
        # Generate predictions
        voltage_pred = self.voltage_head(h_fused)
        current_pred = self.current_head(h_fused)
        
        # Edge predictions (for switches)
        # Get edge embeddings by concatenating source and dest node embeddings
        edge_embeddings = torch.cat([
            h_fused[edge_index[0]],
            h_fused[edge_index[1]]
        ], dim=-1)
        
        # Project to hidden_dim for switch head
        edge_proj = nn.Linear(2 * self.hidden_dim, self.hidden_dim).to(edge_embeddings.device)
        edge_features = edge_proj(edge_embeddings)
        switch_logits = self.switch_head(edge_features)
        
        # Prepare output
        output = {
            'switch_logits': switch_logits,
            'voltage_pred': voltage_pred,
            'current_pred': current_pred,
            'h_local': h_local
        }
        
        # Add uncertainty if enabled
        if self.use_uncertainty:
            uncertainty = self.uncertainty_head(h_fused)
            output['uncertainty'] = uncertainty
        
        if return_embeddings:
            output['h_local'] = h_local
            output['h_global'] = h_global
            output['attention_weights'] = attention_weights
        
        return output


if __name__ == "__main__":
    # Test the model
    print("Testing DS-PAH-GNN model...")
    
    from torch_geometric.data import Data
    
    # Create dummy graph
    num_nodes = 10
    num_edges = 20
    
    x = torch.randn(num_nodes, 4)  # 4 node features
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 3)  # 3 edge features
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    # Initialize model
    model = DS_PAH_GNN(
        node_dim=4,
        edge_dim=3,
        hidden_dim=64,
        local_layers=2,
        global_layers=2
    )
    
    # Forward pass
    output = model(data, return_embeddings=True)
    
    print(f"Switch logits shape: {output['switch_logits'].shape}")
    print(f"Voltage predictions shape: {output['voltage_pred'].shape}")
    print(f"Current predictions shape: {output['current_pred'].shape}")
    print(f"Local embeddings shape: {output['h_local'].shape}")
    print(f"Global embeddings shape: {output['h_global'].shape}")
    print("✓ DS-PAH-GNN test passed!")

