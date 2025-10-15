"""
Topology generator for dual-bus power distribution systems.

This module generates synthetic power distribution network topologies
with dual-bus architecture, radial feeders, and switchable tie-lines.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, Tuple, Optional
import random


def sample_dual_bus_topology(
    n_buses: int = 2,
    n_feeders_per_bus: int = 2,
    nodes_per_feeder: int = 5,
    tie_switches: int = 1,
    seed: Optional[int] = None,
    params: Optional[Dict] = None
) -> Data:
    """
    Sample a dual-bus topology with radial feeders and optional tie-switches.
    
    Args:
        n_buses: Number of buses (typically 2 for dual-bus)
        n_feeders_per_bus: Number of feeders per bus
        nodes_per_feeder: Number of nodes in each feeder
        tie_switches: Number of tie-switches between buses
        seed: Random seed for reproducibility
        params: Additional parameters dictionary
        
    Returns:
        PyG Data object with node features, edge indices, edge features, and metadata
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Override with params if provided
    if params is not None:
        n_buses = params.get('n_buses', n_buses)
        n_feeders_per_bus = params.get('n_feeders_per_bus', n_feeders_per_bus)
        nodes_per_feeder = params.get('nodes_per_feeder', nodes_per_feeder)
        tie_switches = params.get('tie_switches', tie_switches)
    
    # Calculate total nodes
    n_station_nodes = n_buses  # One substation per bus
    n_feeder_nodes = n_buses * n_feeders_per_bus * nodes_per_feeder
    n_total = n_station_nodes + n_feeder_nodes
    
    # Initialize edge list
    edge_src = []
    edge_dst = []
    edge_types = []  # 0: feeder line, 1: tie-switch
    edge_resistance = []
    edge_switch_state = []  # 0: off, 1: on
    
    # Metadata mapping
    node_to_bus = {}
    node_to_feeder = {}
    node_types = []  # 0: substation, 1: feeder node
    
    current_node_idx = 0
    
    # Create substations (buses)
    substation_nodes = []
    for bus_idx in range(n_buses):
        substation_nodes.append(current_node_idx)
        node_to_bus[current_node_idx] = bus_idx
        node_to_feeder[current_node_idx] = -1  # Substation, not in feeder
        node_types.append(0)
        current_node_idx += 1
    
    # Create feeders for each bus
    for bus_idx in range(n_buses):
        substation_node = substation_nodes[bus_idx]
        
        for feeder_idx in range(n_feeders_per_bus):
            feeder_id = bus_idx * n_feeders_per_bus + feeder_idx
            prev_node = substation_node
            
            # Create radial feeder chain
            for node_in_feeder in range(nodes_per_feeder):
                curr_node = current_node_idx
                node_to_bus[curr_node] = bus_idx
                node_to_feeder[curr_node] = feeder_id
                node_types.append(1)
                
                # Add edge from previous node to current
                edge_src.append(prev_node)
                edge_dst.append(curr_node)
                edge_types.append(0)  # Regular feeder line
                
                # Random resistance (typical distribution line)
                edge_resistance.append(np.random.uniform(0.1, 0.5))
                edge_switch_state.append(1)  # Feeder lines are typically on
                
                prev_node = curr_node
                current_node_idx += 1
    
    # Add tie-switches between buses (connect random nodes from different buses)
    for _ in range(tie_switches):
        # Select random feeder nodes from different buses
        bus1_nodes = [n for n in range(n_total) if node_to_bus.get(n, 0) == 0 and node_types[n] == 1]
        bus2_nodes = [n for n in range(n_total) if node_to_bus.get(n, 0) == 1 and node_types[n] == 1]
        
        if bus1_nodes and bus2_nodes:
            node1 = random.choice(bus1_nodes)
            node2 = random.choice(bus2_nodes)
            
            edge_src.append(node1)
            edge_dst.append(node2)
            edge_types.append(1)  # Tie-switch
            edge_resistance.append(np.random.uniform(0.1, 0.5))
            edge_switch_state.append(0)  # Tie-switches typically start open
    
    # Create bidirectional edges
    edge_index = []
    for src, dst in zip(edge_src, edge_dst):
        edge_index.append([src, dst])
        edge_index.append([dst, src])  # Bidirectional
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Duplicate edge attributes for bidirectional edges
    edge_types_full = edge_types + edge_types
    edge_resistance_full = edge_resistance + edge_resistance
    edge_switch_state_full = edge_switch_state + edge_switch_state
    
    # Node features: [node_type, voltage_nominal, power_demand, has_converter]
    node_features = []
    for i in range(n_total):
        node_type = node_types[i]
        voltage_nominal = 1.0  # Per-unit voltage
        
        if node_type == 0:  # Substation
            power_demand = 0.0
            has_converter = 0.0
        else:  # Feeder node
            power_demand = np.random.uniform(0.1, 0.8)  # Random load
            has_converter = np.random.binomial(1, 0.3)  # 30% have converters (EVs, solar)
        
        node_features.append([node_type, voltage_nominal, power_demand, has_converter])
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Edge features: [edge_type, resistance, switch_state]
    edge_attr = torch.tensor(
        [[et, r, ss] for et, r, ss in zip(edge_types_full, edge_resistance_full, edge_switch_state_full)],
        dtype=torch.float
    )
    
    # Create PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=n_total
    )
    
    # Add metadata
    data.node_to_bus = node_to_bus
    data.node_to_feeder = node_to_feeder
    data.substation_nodes = substation_nodes
    data.seed = seed
    data.n_buses = n_buses
    data.n_feeders_per_bus = n_feeders_per_bus
    data.nodes_per_feeder = nodes_per_feeder
    
    return data


def batch_sample_topologies(
    n_samples: int,
    params: Optional[Dict] = None,
    base_seed: int = 42
) -> list:
    """
    Generate a batch of topology samples.
    
    Args:
        n_samples: Number of graphs to generate
        params: Parameters for topology generation
        base_seed: Base seed for reproducibility
        
    Returns:
        List of PyG Data objects
    """
    graphs = []
    for i in range(n_samples):
        seed = base_seed + i
        graph = sample_dual_bus_topology(seed=seed, params=params)
        graphs.append(graph)
    return graphs


if __name__ == "__main__":
    # Test the generator
    print("Testing topology generator...")
    graph = sample_dual_bus_topology(seed=42)
    print(f"Generated graph with {graph.num_nodes} nodes and {graph.edge_index.shape[1]} edges")
    print(f"Node features shape: {graph.x.shape}")
    print(f"Edge features shape: {graph.edge_attr.shape}")
    print(f"Substations: {graph.substation_nodes}")
    print("\nGraph structure:")
    print(graph)

