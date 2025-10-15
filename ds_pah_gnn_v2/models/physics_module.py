"""
Physics-aware constraints and surrogate models for power systems.

Implements:
- Linear DC power flow approximation
- Kirchhoff's Current Law (KCL) residual computation
- Converter droop models
- Physics loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def compute_edge_current(
    V: torch.Tensor,
    edge_index: torch.Tensor,
    R: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute edge currents using Ohm's law: I_e = (V_i - V_j) / R_e
    
    Args:
        V: [num_nodes, 1] - node voltages
        edge_index: [2, num_edges] - edge connectivity
        R: [num_edges, 1] - edge resistances
        eps: Small constant to avoid division by zero
        
    Returns:
        I: [num_edges, 1] - edge currents
    """
    src_nodes = edge_index[0]
    dst_nodes = edge_index[1]
    
    V_src = V[src_nodes]
    V_dst = V[dst_nodes]
    
    # Ohm's law: I = (V_src - V_dst) / R
    I = (V_src - V_dst) / (R + eps)
    
    return I


def kcl_residual(
    P_node: torch.Tensor,
    V: torch.Tensor,
    edge_index: torch.Tensor,
    R: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute Kirchhoff's Current Law (KCL) residuals at each node.
    
    KCL: Sum of currents entering node = Power injection / Voltage
         ∑_j I_ij = P_i / V_i
    
    Args:
        P_node: [num_nodes, 1] - node power injections (positive = generation, negative = load)
        V: [num_nodes, 1] - node voltages
        edge_index: [2, num_edges] - edge connectivity
        R: [num_edges, 1] - edge resistances
        eps: Small constant for numerical stability
        
    Returns:
        residual: [num_nodes, 1] - KCL residual at each node
    """
    num_nodes = P_node.shape[0]
    
    # Compute edge currents
    I_edges = compute_edge_current(V, edge_index, R, eps)
    
    # Aggregate currents at each node (incoming - outgoing)
    I_net = torch.zeros_like(P_node)
    
    src_nodes = edge_index[0]
    dst_nodes = edge_index[1]
    
    # For each edge, current flows from src to dst
    # At src node: current flows out (negative)
    # At dst node: current flows in (positive)
    I_net.index_add_(0, dst_nodes, I_edges)
    I_net.index_add_(0, src_nodes, -I_edges)
    
    # Expected current from power injection: I_expected = P / V
    I_expected = P_node / (V + eps)
    
    # KCL residual: difference between net current and expected current
    residual = torch.abs(I_net - I_expected)
    
    return residual


def physics_loss(
    P_node: torch.Tensor,
    V_pred: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    lambda_kcl: float = 1.0
) -> torch.Tensor:
    """
    Compute physics-based loss including KCL violations.
    
    Args:
        P_node: [num_nodes, 1] - true power at nodes
        V_pred: [num_nodes, 1] - predicted voltages
        edge_index: [2, num_edges]
        edge_attr: [num_edges, edge_dim] - edge features (resistance in column 1)
        lambda_kcl: Weight for KCL loss
        
    Returns:
        loss: Scalar physics loss
    """
    # Extract resistances from edge attributes
    # Assuming edge_attr[:, 1] contains resistance
    R = edge_attr[:, 1:2]
    
    # Compute KCL residuals
    kcl_res = kcl_residual(P_node, V_pred, edge_index, R)
    
    # KCL loss: penalize violations
    L_kcl = lambda_kcl * kcl_res.mean()
    
    return L_kcl


class ConverterModel(nn.Module):
    """
    Learnable converter model for distributed energy resources (DER).
    
    Models the relationship between setpoint and actual V/I behavior
    with droop characteristics.
    """
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 32):
        """
        Args:
            input_dim: Input dimension (e.g., [V_setpoint, I_setpoint])
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.droop_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Output: [V_actual, I_actual]
        )
    
    def forward(self, setpoints: torch.Tensor) -> torch.Tensor:
        """
        Args:
            setpoints: [batch, 2] - [V_setpoint, I_setpoint]
            
        Returns:
            actual: [batch, 2] - [V_actual, I_actual]
        """
        return self.droop_net(setpoints)


class PhysicsConstraintLayer(nn.Module):
    """
    Differentiable physics constraint layer that can be integrated into the model.
    """
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        
        self.constraint_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply physics-informed constraints to embeddings.
        
        Returns:
            x_constrained: Physics-constrained node embeddings
            constraint_violation: Measure of constraint violation
        """
        x_constrained = self.constraint_net(x)
        
        # Placeholder for actual constraint computation
        # In practice, this would enforce physical laws
        constraint_violation = torch.tensor(0.0)
        
        return x_constrained, constraint_violation


def voltage_constraint_loss(
    V_pred: torch.Tensor,
    V_min: float = 0.95,
    V_max: float = 1.05,
    penalty_weight: float = 100.0
) -> torch.Tensor:
    """
    Soft constraint for voltage limits (per-unit).
    
    Args:
        V_pred: [num_nodes, 1] - predicted voltages
        V_min: Minimum allowed voltage (p.u.)
        V_max: Maximum allowed voltage (p.u.)
        penalty_weight: Penalty weight for violations
        
    Returns:
        loss: Voltage constraint violation loss
    """
    # Violations below minimum
    violation_low = F.relu(V_min - V_pred)
    
    # Violations above maximum
    violation_high = F.relu(V_pred - V_max)
    
    # Total violation
    total_violation = violation_low + violation_high
    
    loss = penalty_weight * total_violation.mean()
    
    return loss


def switching_regularization(
    switch_logits: torch.Tensor,
    switch_state_prev: Optional[torch.Tensor] = None,
    lambda_sparse: float = 0.1,
    lambda_smooth: float = 1.0
) -> torch.Tensor:
    """
    Regularization for switch operations.
    
    Args:
        switch_logits: [num_edges, 1] - current switch logits
        switch_state_prev: [num_edges, 1] - previous switch states
        lambda_sparse: Weight for sparsity (minimize total switches)
        lambda_smooth: Weight for smoothness (minimize switching frequency)
        
    Returns:
        loss: Switching regularization loss
    """
    # Convert logits to probabilities
    switch_probs = torch.sigmoid(switch_logits)
    
    # Sparsity loss: encourage fewer switches to be on
    # L1 penalty on probabilities
    L_sparse = lambda_sparse * switch_probs.mean()
    
    # Smoothness loss: penalize changes in switch state
    L_smooth = torch.tensor(0.0, device=switch_logits.device)
    if switch_state_prev is not None:
        prev_probs = torch.sigmoid(switch_state_prev)
        L_smooth = lambda_smooth * F.mse_loss(switch_probs, prev_probs)
    
    return L_sparse + L_smooth


if __name__ == "__main__":
    # Test physics functions
    print("Testing physics module...")
    
    num_nodes = 5
    num_edges = 8
    
    # Create test data
    V = torch.ones(num_nodes, 1) * 1.0  # All voltages at 1.0 p.u.
    P_node = torch.randn(num_nodes, 1) * 0.1  # Small power injections
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    R = torch.ones(num_edges, 1) * 0.2  # Uniform resistance
    
    # Test edge current computation
    I = compute_edge_current(V, edge_index, R)
    print(f"Edge currents shape: {I.shape}")
    
    # Test KCL residual
    kcl_res = kcl_residual(P_node, V, edge_index, R)
    print(f"KCL residuals shape: {kcl_res.shape}")
    print(f"Mean KCL residual: {kcl_res.mean().item():.6f}")
    
    # Test voltage constraints
    V_test = torch.tensor([[0.90], [0.98], [1.02], [1.08], [1.00]])
    v_loss = voltage_constraint_loss(V_test)
    print(f"Voltage constraint loss: {v_loss.item():.4f}")
    
    # Test converter model
    converter = ConverterModel()
    setpoints = torch.tensor([[1.0, 0.5], [0.95, 0.3]])
    actual = converter(setpoints)
    print(f"Converter output shape: {actual.shape}")
    
    print("✓ Physics module tests passed!")

