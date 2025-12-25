import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, Linear, global_mean_pool, SAGEConv


class VanillaGNN(torch.nn.Module):
    """A vanilla heterogeneous GNN baseline that ignores physics edge attributes.

    The forward signature matches the main model for drop-in evaluation. Encoders
    for node features are created lazily to accommodate possible variations in
    dataset feature dimensions.
    """
    def __init__(self, hidden_channels=64, out_channels=1):
        super().__init__()
        # Placeholder encoders; will initialize lazily on first forward
        self.bus_lin = None
        self.device_lin = None

        self.hidden_channels = hidden_channels

        # Create per-relation SAGEConv modules and manage aggregation manually
        from torch import nn
        self.rel_keys = [
            ('bus', 'connects', 'bus'),
            ('bus', 'feeds', 'device'),
            ('device', 'rev_feeds', 'bus')
        ]

        self.conv1_rel = nn.ModuleDict()
        self.conv2_rel = nn.ModuleDict()
        for src, rel, dst in self.rel_keys:
            key = f"{src}__{rel}__{dst}"
            self.conv1_rel[key] = SAGEConv(hidden_channels, hidden_channels)
            self.conv2_rel[key] = SAGEConv(hidden_channels, hidden_channels)

        self.head = Linear(hidden_channels, out_channels)

    def _init_encoders(self, bus_feat_dim, device_feat_dim):
        # Initialize encoders if they are not yet created
        if self.bus_lin is None:
            self.bus_lin = Linear(bus_feat_dim, self.hidden_channels)
        if self.device_lin is None:
            self.device_lin = Linear(device_feat_dim, self.hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None, batch=None):
        # Lazy initialization of encoders based on input feature sizes
        bus_feat_dim = x_dict['bus'].shape[1]
        device_feat_dim = x_dict['device'].shape[1]
        self._init_encoders(bus_feat_dim, device_feat_dim)

        # Encode nodes
        x = {
            'bus': F.relu(self.bus_lin(x_dict['bus'])),
            'device': F.relu(self.device_lin(x_dict['device']))
        }

        # --- Manual heterogeneous message passing (layer 1) ---
        agg = {k: [] for k in x.keys()}  # collect messages for each dst node type
        for src, rel, dst in self.rel_keys:
            key = f"{src}__{rel}__{dst}"
            if (src, rel, dst) not in edge_index_dict:
                continue
            edge_index = edge_index_dict[(src, rel, dst)]
            # If source and dest are same type, pass single tensor; else pass tuple
            if src == dst:
                msg = self.conv1_rel[key](x[dst], edge_index)
            else:
                msg = self.conv1_rel[key]((x[src], x[dst]), edge_index)
            agg[dst].append(msg)

        # Sum aggregated messages per node type
        x_next = {}
        for ntype, msgs in agg.items():
            if msgs:
                x_next[ntype] = F.relu(sum(msgs))
            else:
                x_next[ntype] = F.relu(x[ntype])

        # --- Layer 2 ---
        agg2 = {k: [] for k in x.keys()}
        for src, rel, dst in self.rel_keys:
            key = f"{src}__{rel}__{dst}"
            if (src, rel, dst) not in edge_index_dict:
                continue
            edge_index = edge_index_dict[(src, rel, dst)]
            if src == dst:
                msg = self.conv2_rel[key](x_next[dst], edge_index)
            else:
                msg = self.conv2_rel[key]((x_next[src], x_next[dst]), edge_index)
            agg2[dst].append(msg)

        x = {}
        for ntype, msgs in agg2.items():
            if msgs:
                x[ntype] = F.relu(sum(msgs))
            else:
                x[ntype] = F.relu(x_next[ntype])

        bus_emb = x['bus']
        if batch is not None:
            graph_emb = global_mean_pool(bus_emb, batch)
        else:
            graph_emb = torch.mean(bus_emb, dim=0, keepdim=True)

        out = self.head(graph_emb)
        return out


class SimpleMLP(torch.nn.Module):
    """A small MLP baseline trained on aggregated features."""
    def __init__(self, in_dim=3, hidden=64, out_dim=1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden//2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden//2, out_dim)
        )

    def forward(self, x):
        return self.net(x)
