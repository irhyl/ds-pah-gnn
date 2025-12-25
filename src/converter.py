import torch
from torch_geometric.data import HeteroData

def raw_to_pyg(snapshot):
    """
    Converts a raw simulation snapshot (dict) into a PyG HeteroData object.
    """
    data = HeteroData()
    
    # 1. Separate Nodes by Type
    # We map global IDs to local indices: global_id -> (type, local_idx)
    id_map = {}
    
    bus_feats = []      # [Voltage, Power]
    device_feats = []   # [Voltage, Power, SOC, Type_OneHot]
    
    for node in snapshot['nodes']:
        # Feature: [Voltage (p.u.), Power (p.u.)]
        base_feat = [node['v'], node['p']]
        
        if node['type'] in ['root', 'station']:
            id_map[node['id']] = ('bus', len(bus_feats))
            bus_feats.append(base_feat)
        else:
            # Map all leaf nodes (charger, pv, storage) to 'device'
            type_encoding = 0.0
            if node['type'] == 'pv': type_encoding = 1.0
            if node['type'] == 'storage': type_encoding = 2.0
            
            # SOC is relevant for storage, 0.0 for others
            soc = node.get('soc', 0.0)
            
            id_map[node['id']] = ('device', len(device_feats))
            device_feats.append(base_feat + [soc, type_encoding])
            
    # Create Node Tensors
    data['bus'].x = torch.tensor(bus_feats, dtype=torch.float)
    data['device'].x = torch.tensor(device_feats, dtype=torch.float)
    
    # 2. Build Edges
    bus_bus_src, bus_bus_dst = [], []
    bus_bus_attr = [] # [Conductance, Status]
    
    bus_dev_src, bus_dev_dst = [], []
    bus_dev_attr = []
    
    for edge in snapshot['edges']:
        # CRITICAL: Skip open switches so the Graph Topology changes
        if edge['status'] == 0: continue

        u_type, u_idx = id_map[edge['u']]
        v_type, v_idx = id_map[edge['v']]
        
        # Physics-Aware Feature: Conductance (1/R) is more useful for GNN than R
        # Add epsilon to avoid division by zero
        attr = [1.0 / (edge['r'] + 1e-6), float(edge['status'])]
        
        # Logic: Bus <-> Bus
        if u_type == 'bus' and v_type == 'bus':
            bus_bus_src.extend([u_idx, v_idx])
            bus_bus_dst.extend([v_idx, u_idx])
            bus_bus_attr.extend([attr, attr]) # Undirected
            
        # Logic: Bus -> Device
        elif u_type == 'bus' and v_type == 'device':
            bus_dev_src.append(u_idx)
            bus_dev_dst.append(v_idx)
            bus_dev_attr.append(attr)

    # Create Edge Tensors
    def to_tensor(src, dst, attr):
        if not src:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 2), dtype=torch.float)
        return torch.tensor([src, dst], dtype=torch.long), torch.tensor(attr, dtype=torch.float)

    data['bus', 'connects', 'bus'].edge_index, data['bus', 'connects', 'bus'].edge_attr = \
        to_tensor(bus_bus_src, bus_bus_dst, bus_bus_attr)
    data['bus', 'feeds', 'device'].edge_index, data['bus', 'feeds', 'device'].edge_attr = \
        to_tensor(bus_dev_src, bus_dev_dst, bus_dev_attr)
    data['device', 'rev_feeds', 'bus'].edge_index, data['device', 'rev_feeds', 'bus'].edge_attr = \
        to_tensor(bus_dev_dst, bus_dev_src, bus_dev_attr)

    # 3. Add Labels
    # Handle case where loss might be missing (though unlikely in this pipeline)
    loss = snapshot.get('loss', 0.0)
    data.y_loss = torch.tensor([loss], dtype=torch.float)
    
    return data