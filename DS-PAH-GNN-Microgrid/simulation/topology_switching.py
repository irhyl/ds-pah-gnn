"""
Dynamic topology switching and reconfiguration for microgrids.
"""

import networkx as nx
import numpy as np
from typing import List, Tuple, Optional, Dict
from copy import deepcopy


class TopologySwitcher:
    """
    Manages dynamic topology reconfiguration of microgrids.
    """
    
    def __init__(self, base_graph: nx.Graph):
        """
        Initialize with a base microgrid topology.
        
        Parameters
        ----------
        base_graph : nx.Graph
            Base microgrid network
        """
        self.base_graph = deepcopy(base_graph)
        self.current_graph = deepcopy(base_graph)
        self.switch_history = []
    
    def get_possible_reconfigurations(self) -> List[nx.Graph]:
        """
        Get all valid topology reconfigurations.
        
        Returns connected subgraphs that maintain system connectivity.
        """
        reconfigurations = []
        
        # Try removing each edge
        for edge in self.base_graph.edges():
            reconfigured = deepcopy(self.base_graph)
            reconfigured.remove_edge(*edge)
            
            # Check if still connected
            if nx.is_connected(reconfigured):
                reconfigurations.append(reconfigured)
        
        return reconfigurations
    
    def switch_to_topology(self, new_graph: nx.Graph) -> bool:
        """
        Switch to a new topology.
        
        Parameters
        ----------
        new_graph : nx.Graph
            New topology to switch to
            
        Returns
        -------
        success : bool
            Whether the switch was successful
        """
        if not nx.is_connected(new_graph):
            return False
        
        switch_info = {
            'from_edges': set(self.current_graph.edges()),
            'to_edges': set(new_graph.edges()),
            'edges_removed': set(self.current_graph.edges()) - set(new_graph.edges()),
            'edges_added': set(new_graph.edges()) - set(self.current_graph.edges())
        }
        
        self.current_graph = deepcopy(new_graph)
        self.switch_history.append(switch_info)
        
        return True
    
    def island_on_fault(self, failed_edge: Tuple[int, int]) -> bool:
        """
        Reconfigure to island the network after a line outage.
        
        Parameters
        ----------
        failed_edge : tuple
            Edge that has failed (node1, node2)
            
        Returns
        -------
        success : bool
            Whether islanding was successful
        """
        reconfigured = deepcopy(self.current_graph)
        reconfigured.remove_edge(*failed_edge)
        
        if nx.is_connected(reconfigured):
            # Still connected, no need to island
            return True
        
        # Find connected components
        components = list(nx.connected_components(reconfigured))
        
        if len(components) > 1:
            # Keep only the largest component
            largest = max(components, key=len)
            nodes_to_keep = largest
            nodes_to_remove = set(reconfigured.nodes()) - nodes_to_keep
            reconfigured.remove_nodes_from(nodes_to_remove)
            
            self.switch_to_topology(reconfigured)
            return True
        
        return False
    
    def restore_after_fault(self, repaired_edges: List[Tuple[int, int]]) -> bool:
        """
        Restore connectivity after component repair.
        
        Parameters
        ----------
        repaired_edges : list of tuple
            Edges that have been repaired
            
        Returns
        -------
        success : bool
            Whether restoration was successful
        """
        restored = deepcopy(self.current_graph)
        
        for edge in repaired_edges:
            if edge in self.base_graph.edges():
                restored.add_edge(edge[0], edge[1])
                # Copy attributes
                for key, value in self.base_graph.edges[edge].items():
                    restored.edges[edge][key] = value
        
        if nx.is_connected(restored):
            self.switch_to_topology(restored)
            return True
        
        return False
    
    def optimize_for_losses(self) -> nx.Graph:
        """
        Find topology that minimizes transmission losses.
        
        Returns
        -------
        optimal_graph : nx.Graph
            Topology with minimized losses
        """
        candidates = self.get_possible_reconfigurations()
        
        min_loss = float('inf')
        optimal_topo = self.current_graph
        
        for candidate in candidates:
            # Calculate total resistance loss
            loss = 0
            for edge in candidate.edges():
                r = candidate.edges[edge].get('resistance', 0.05)
                loss += r
            
            if loss < min_loss:
                min_loss = loss
                optimal_topo = candidate
        
        return optimal_topo
    
    def get_switch_history(self) -> List[Dict]:
        """Get history of all topology switches."""
        return self.switch_history
    
    def reset_to_base(self) -> None:
        """Reset to base topology."""
        self.current_graph = deepcopy(self.base_graph)
        self.switch_history = []
