"""
Temporal encoding for time-series power system data.

Supports:
- Positional encoding for time steps
- Sequence modeling of graph snapshots
- Temporal smoothness constraints
"""

import torch
import torch.nn as nn
import math
from typing import List, Optional
from torch_geometric.data import Data, Batch


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal sequences.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor, t: int = 0) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model] or [batch, d_model]
            t: Time step index
            
        Returns:
            x with positional encoding added
        """
        if x.dim() == 3:
            # Sequence input
            seq_len = x.size(1)
            x = x + self.pe[:seq_len, :].unsqueeze(0)
        else:
            # Single time step
            x = x + self.pe[t, :].unsqueeze(0)
        
        return x


class TemporalGraphEncoder(nn.Module):
    """
    Encodes sequences of graph snapshots with temporal awareness.
    """
    
    def __init__(self, node_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        
        # Temporal feature encoder
        self.temporal_mlp = nn.Sequential(
            nn.Linear(node_dim + 1, hidden_dim),  # +1 for time feature
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)
    
    def forward(
        self,
        graph_sequence: List[Data],
        return_sequence: bool = False
    ) -> torch.Tensor:
        """
        Process a sequence of graph snapshots.
        
        Args:
            graph_sequence: List of PyG Data objects (length T)
            return_sequence: If True, return all time steps; else return last
            
        Returns:
            Encoded temporal features
        """
        T = len(graph_sequence)
        
        # Process each time step
        temporal_features = []
        for t, graph in enumerate(graph_sequence):
            # Add time as feature
            time_feature = torch.ones(graph.x.size(0), 1) * (t / T)
            x_with_time = torch.cat([graph.x, time_feature], dim=-1)
            
            # Encode
            h_t = self.temporal_mlp(x_with_time)
            temporal_features.append(h_t)
        
        # Stack into sequence: [num_nodes, T, hidden_dim]
        temporal_seq = torch.stack(temporal_features, dim=1)
        
        # Apply positional encoding
        temporal_seq = self.pos_encoding(temporal_seq)
        
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(temporal_seq)
        
        if return_sequence:
            return lstm_out  # [num_nodes, T, hidden_dim]
        else:
            return lstm_out[:, -1, :]  # [num_nodes, hidden_dim]


class TemporalSmoothnessloss(nn.Module):
    """
    Encourages smooth transitions in predictions over time.
    """
    
    def __init__(self, order: int = 1):
        """
        Args:
            order: Derivative order (1 = first derivative, 2 = second derivative)
        """
        super().__init__()
        self.order = order
    
    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [batch, seq_len, feature_dim] - predictions over time
            
        Returns:
            smoothness_loss: Scalar loss encouraging smooth transitions
        """
        if predictions.size(1) < 2:
            return torch.tensor(0.0, device=predictions.device)
        
        if self.order == 1:
            # First-order: penalize differences between consecutive steps
            diff = predictions[:, 1:, :] - predictions[:, :-1, :]
            loss = (diff ** 2).mean()
        elif self.order == 2:
            # Second-order: penalize acceleration
            if predictions.size(1) < 3:
                return torch.tensor(0.0, device=predictions.device)
            first_diff = predictions[:, 1:, :] - predictions[:, :-1, :]
            second_diff = first_diff[:, 1:, :] - first_diff[:, :-1, :]
            loss = (second_diff ** 2).mean()
        else:
            raise ValueError(f"Order {self.order} not supported")
        
        return loss


class TimeContextAttention(nn.Module):
    """
    Attention mechanism over temporal context.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.query_net = nn.Linear(hidden_dim, hidden_dim)
        self.key_net = nn.Linear(hidden_dim, hidden_dim)
        self.value_net = nn.Linear(hidden_dim, hidden_dim)
        self.scale = math.sqrt(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_dim]
            
        Returns:
            attended: [batch, seq_len, hidden_dim]
        """
        Q = self.query_net(x)
        K = self.key_net(x)
        V = self.value_net(x)
        
        # Attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Attended output
        attended = torch.matmul(attn_weights, V)
        
        return attended


if __name__ == "__main__":
    # Test temporal encoding
    print("Testing temporal encoder...")
    
    # Create dummy sequence of graphs
    from torch_geometric.data import Data
    
    num_nodes = 10
    node_dim = 4
    T = 5  # sequence length
    
    graph_sequence = []
    for t in range(T):
        x = torch.randn(num_nodes, node_dim)
        edge_index = torch.randint(0, num_nodes, (2, 20))
        edge_attr = torch.randn(20, 3)
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        graph_sequence.append(graph)
    
    # Test temporal encoder
    encoder = TemporalGraphEncoder(node_dim=node_dim, hidden_dim=64)
    
    # Encode last time step
    h_last = encoder(graph_sequence, return_sequence=False)
    print(f"Last time step encoding shape: {h_last.shape}")
    
    # Encode full sequence
    h_seq = encoder(graph_sequence, return_sequence=True)
    print(f"Full sequence encoding shape: {h_seq.shape}")
    
    # Test smoothness loss
    predictions = torch.randn(2, T, 16)  # [batch=2, seq=5, features=16]
    smooth_loss = TemporalSmoothnessloss(order=1)
    loss = smooth_loss(predictions)
    print(f"Temporal smoothness loss: {loss.item():.4f}")
    
    # Test time attention
    attn = TimeContextAttention(hidden_dim=64)
    x_seq = torch.randn(2, T, 64)
    attended = attn(x_seq)
    print(f"Time attention output shape: {attended.shape}")
    
    print("✓ Temporal encoder tests passed!")

