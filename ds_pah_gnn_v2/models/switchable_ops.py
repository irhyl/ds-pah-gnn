"""
Switchable Edge Operators (SEO) for dynamic power grid topology.

These operators learn different message passing functions for edges
that can be switched on/off (e.g., circuit breakers, tie-switches).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwitchableEdgeOperator(nn.Module):
    """
    Switchable Edge Operator that learns separate transformations
    for 'on' and 'off' states and blends them based on learned gating.
    """
    
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64):
        """
        Args:
            in_dim: Input edge feature dimension
            out_dim: Output edge feature dimension
            hidden_dim: Hidden dimension for gating network
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Operator for 'on' state (closed switch)
        self.O_on = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        
        # Operator for 'off' state (open switch)
        self.O_off = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        
        # Gating network to compute mixing weight
        self.gating = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        """
        Apply switchable operator to edge features.
        
        Args:
            edge_features: [num_edges, in_dim]
            
        Returns:
            Transformed edge features: [num_edges, out_dim]
        """
        # Compute transformations for both states
        h_on = self.O_on(edge_features)
        h_off = self.O_off(edge_features)
        
        # Compute gating coefficient alpha ∈ [0, 1]
        alpha = self.gating(edge_features)  # [num_edges, 1]
        
        # Blend: O_eff = alpha * O_on + (1 - alpha) * O_off
        h_eff = alpha * h_on + (1 - alpha) * h_off
        
        return h_eff


class SimpleSwitchableOp(nn.Module):
    """
    Simplified switchable operator (linear version for quick prototyping).
    """
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        
        self.O_on = nn.Linear(in_dim, out_dim)
        self.O_off = nn.Linear(in_dim, out_dim)
        self.gating = nn.Sequential(
            nn.Linear(in_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        h_on = self.O_on(edge_features)
        h_off = self.O_off(edge_features)
        alpha = self.gating(edge_features)
        return alpha * h_on + (1 - alpha) * h_off


if __name__ == "__main__":
    # Test the switchable operator
    print("Testing Switchable Edge Operator...")
    
    batch_size = 10
    in_dim = 16
    out_dim = 32
    
    op = SwitchableEdgeOperator(in_dim, out_dim)
    edge_feats = torch.randn(batch_size, in_dim)
    
    output = op(edge_feats)
    print(f"Input shape: {edge_feats.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Switchable operator test passed!")

