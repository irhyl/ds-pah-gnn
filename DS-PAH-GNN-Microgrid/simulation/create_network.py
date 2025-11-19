"""
Network creation utilities for building microgrid graphs.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional


def create_microgrid(
    num_nodes: int = 10,
    topology: str = "mesh",
    seed: Optional[int] = None
) -> Tuple[nx.Graph, Dict]:
    """
    Create a microgrid network graph.
    
    Parameters
    ----------
    num_nodes : int
        Number of nodes in the microgrid
    topology : str
        Network topology type ('mesh', 'ring', 'tree')
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    graph : nx.Graph
        NetworkX graph representing the microgrid
    metadata : dict
        Metadata about the network (nodes, edges, attributes)
    """
    if seed is not None:
        np.random.seed(seed)
    
    if topology == "mesh":
        graph = nx.complete_graph(num_nodes)
    elif topology == "ring":
        graph = nx.cycle_graph(num_nodes)
    elif topology == "tree":
        graph = nx.balanced_tree(2, np.log2(num_nodes).astype(int))
    else:
        raise ValueError(f"Unknown topology: {topology}")
    
    # Add node attributes
    for node in graph.nodes():
        graph.nodes[node]['voltage'] = np.random.uniform(0.95, 1.05)
        graph.nodes[node]['frequency'] = 50.0 + np.random.normal(0, 0.1)
        graph.nodes[node]['generation'] = np.random.uniform(0, 100) if np.random.random() > 0.5 else 0
        graph.nodes[node]['load'] = np.random.uniform(10, 50)
    
    # Add edge attributes
    for edge in graph.edges():
        graph.edges[edge]['resistance'] = np.random.uniform(0.01, 0.1)
        graph.edges[edge]['reactance'] = np.random.uniform(0.02, 0.2)
        graph.edges[edge]['capacity'] = np.random.uniform(50, 200)
    
    metadata = {
        'num_nodes': num_nodes,
        'num_edges': graph.number_of_edges(),
        'topology': topology,
        'nodes': list(graph.nodes()),
        'edges': list(graph.edges())
    }
    
    return graph, metadata


def add_nodes(graph: nx.Graph, num_new_nodes: int) -> nx.Graph:
    """Add new nodes to an existing microgrid."""
    current_num = graph.number_of_nodes()
    for i in range(num_new_nodes):
        node_id = current_num + i
        graph.add_node(node_id)
        graph.nodes[node_id]['voltage'] = np.random.uniform(0.95, 1.05)
        graph.nodes[node_id]['frequency'] = 50.0
        graph.nodes[node_id]['generation'] = 0
        graph.nodes[node_id]['load'] = np.random.uniform(10, 50)
    return graph


def connect_nodes(
    graph: nx.Graph,
    node1: int,
    node2: int,
    resistance: float = 0.05,
    reactance: float = 0.1,
    capacity: float = 100
) -> nx.Graph:
    """Add an edge between two nodes with electrical parameters."""
    graph.add_edge(node1, node2)
    graph.edges[node1, node2]['resistance'] = resistance
    graph.edges[node1, node2]['reactance'] = reactance
    graph.edges[node1, node2]['capacity'] = capacity
    return graph
