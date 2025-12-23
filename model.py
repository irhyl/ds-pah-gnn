import torch
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, HeteroConv, Linear, global_mean_pool, SAGPooling

class MicrogridGNN(torch.nn.Module):
    def __init__(self, hidden_channels=64, out_channels=1):
        super().__init__()
        
        # 1. Encoders
        # Bus: [V, P] -> Hidden
        self.bus_lin = Linear(2, hidden_channels)
        # Device: [V, P, SOC, Type] -> Hidden
        self.device_lin = Linear(4, hidden_channels)
        
        # Edge Encoder: [Conductance, Status] -> Hidden
        self.edge_lin = Linear(2, hidden_channels)
        
        # 2. Message Passing Layers
        # We use GINEConv which supports edge attributes
        self.conv1 = HeteroConv({
            ('bus', 'connects', 'bus'): GINEConv(Linear(hidden_channels, hidden_channels)),
            ('bus', 'feeds', 'device'): GINEConv(Linear(hidden_channels, hidden_channels)),
            ('device', 'rev_feeds', 'bus'): GINEConv(Linear(hidden_channels, hidden_channels)),
        }, aggr='sum') # 'sum' mimics Kirchhoff's Current Law
        
        self.conv2 = HeteroConv({
            ('bus', 'connects', 'bus'): GINEConv(Linear(hidden_channels, hidden_channels)),
            ('bus', 'feeds', 'device'): GINEConv(Linear(hidden_channels, hidden_channels)),
            ('device', 'rev_feeds', 'bus'): GINEConv(Linear(hidden_channels, hidden_channels)),
        }, aggr='sum')
        
        # 3. Hierarchical Pooling (SAGPool)
        # "Multi-Resolution": We keep top 50% of critical nodes (ratio=0.5)
        self.pool = SAGPooling(hidden_channels, ratio=0.5)
        
        # 4. Decoder: Predict Global Loss
        # Input dim is doubled because we concat [Global_Context, Critical_Context]
        self.head = Linear(hidden_channels * 2, out_channels)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, batch=None):
        # --- Step 1: Encode Features ---
        x_dict = {
            'bus': F.leaky_relu(self.bus_lin(x_dict['bus'])),
            'device': F.leaky_relu(self.device_lin(x_dict['device']))
        }
        
        # Encode Edge Attributes for every edge type
        encoded_edge_attr = {}
        for key, attr in edge_attr_dict.items():
            encoded_edge_attr[key] = F.leaky_relu(self.edge_lin(attr))

        # --- Step 2: Message Passing ---
        x_dict = self.conv1(x_dict, edge_index_dict, edge_attr_dict=encoded_edge_attr)
        x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        
        # --- Step 3: Message Passing (Layer 2) ---
        x_dict = self.conv2(x_dict, edge_index_dict, edge_attr_dict=encoded_edge_attr)
        x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        
        # --- Step 4: Multi-Resolution Readout ---
        bus_embedding = x_dict['bus']
        
        # A. Global Resolution (Coarse View)
        # Standard mean pooling over the entire grid
        if batch is not None:
            global_feat = global_mean_pool(bus_embedding, batch)
        else:
            global_feat = torch.mean(bus_embedding, dim=0, keepdim=True)
        
        # B. Local/Critical Resolution (Fine View)
        # Use SAGPool to select and aggregate only the most critical buses
        # We use the bus-bus connectivity and physics-aware edge attributes for attention
        edge_index_bus = edge_index_dict[('bus', 'connects', 'bus')]
        edge_attr_bus = encoded_edge_attr[('bus', 'connects', 'bus')]
        
        x_pooled, _, _, batch_pooled, _, _ = self.pool(
            bus_embedding, edge_index_bus, edge_attr=edge_attr_bus, batch=batch
        )
        
        if batch_pooled is not None and batch_pooled.numel() > 0:
            local_feat = global_mean_pool(x_pooled, batch_pooled)
        else:
            local_feat = torch.mean(x_pooled, dim=0, keepdim=True)
            
        # --- Step 5: Prediction ---
        # Concatenate Global + Local features
        combined_feat = torch.cat([global_feat, local_feat], dim=1)
        out = self.head(combined_feat)
        return out

    def get_node_embeddings(self, x_dict, edge_index_dict, edge_attr_dict):
        """
        Extracts the learned node embeddings after message passing.
        Useful for t-SNE visualization to show what the GNN has learned.
        """
        # --- Step 1: Encode ---
        x_dict = {
            'bus': F.leaky_relu(self.bus_lin(x_dict['bus'])),
            'device': F.leaky_relu(self.device_lin(x_dict['device']))
        }
        encoded_edge_attr = {}
        for key, attr in edge_attr_dict.items():
            encoded_edge_attr[key] = F.leaky_relu(self.edge_lin(attr))

        # --- Step 2 & 3: Message Passing ---
        x_dict = self.conv1(x_dict, edge_index_dict, edge_attr_dict=encoded_edge_attr)
        x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        
        x_dict = self.conv2(x_dict, edge_index_dict, edge_attr_dict=encoded_edge_attr)
        x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        
        return x_dict