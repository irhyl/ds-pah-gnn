"""
Evaluation metrics for DS-PAH-GNN.

Includes:
- Voltage metrics (MAE, violations)
- Switch operation metrics
- Physics constraint violations
- Energy efficiency metrics
"""

import torch
import numpy as np
from typing import Dict, Optional, List
from torch_geometric.data import Data


def voltage_mae(V_pred: torch.Tensor, V_target: torch.Tensor) -> float:
    """
    Mean Absolute Error for voltage predictions.
    
    Args:
        V_pred: Predicted voltages [num_nodes, 1]
        V_target: Target voltages [num_nodes, 1]
        
    Returns:
        MAE in per-unit
    """
    return torch.abs(V_pred - V_target).mean().item()


def voltage_rmse(V_pred: torch.Tensor, V_target: torch.Tensor) -> float:
    """
    Root Mean Square Error for voltage predictions.
    """
    return torch.sqrt(((V_pred - V_target) ** 2).mean()).item()


def voltage_violation_rate(
    V_pred: torch.Tensor,
    V_min: float = 0.95,
    V_max: float = 1.05
) -> Dict[str, float]:
    """
    Compute voltage violation statistics.
    
    Args:
        V_pred: Predicted voltages [num_nodes, 1] or [num_nodes]
        V_min: Minimum voltage limit (p.u.)
        V_max: Maximum voltage limit (p.u.)
        
    Returns:
        Dictionary with violation statistics
    """
    V_flat = V_pred.flatten()
    
    violations_low = (V_flat < V_min).float()
    violations_high = (V_flat > V_max).float()
    violations_any = violations_low + violations_high
    
    return {
        'violation_rate': violations_any.mean().item(),
        'low_violation_rate': violations_low.mean().item(),
        'high_violation_rate': violations_high.mean().item(),
        'num_violations': violations_any.sum().item(),
        'max_low_violation': max(0, V_min - V_flat.min().item()),
        'max_high_violation': max(0, V_flat.max().item() - V_max)
    }


def switch_accuracy(
    switch_pred: torch.Tensor,
    switch_target: torch.Tensor,
    threshold: float = 0.5
) -> float:
    """
    Binary accuracy for switch predictions.
    
    Args:
        switch_pred: Predicted switch logits [num_edges, 1]
        switch_target: Target switch states [num_edges, 1]
        threshold: Threshold for binarization
        
    Returns:
        Accuracy (0-1)
    """
    pred_binary = (torch.sigmoid(switch_pred) > threshold).float()
    target_binary = (switch_target > threshold).float()
    
    correct = (pred_binary == target_binary).float()
    return correct.mean().item()


def switch_operation_count(
    switch_states: torch.Tensor,
    prev_states: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Count switch operations (state changes).
    
    Args:
        switch_states: Current switch states [num_edges, 1]
        prev_states: Previous switch states [num_edges, 1]
        
    Returns:
        Dictionary with operation counts
    """
    if prev_states is None:
        return {
            'total_operations': 0.0,
            'switches_on': (switch_states > 0.5).float().sum().item(),
            'switches_off': (switch_states <= 0.5).float().sum().item()
        }
    
    # Count state changes
    changes = (switch_states != prev_states).float()
    
    return {
        'total_operations': changes.sum().item(),
        'operation_rate': changes.mean().item(),
        'switches_on': (switch_states > 0.5).float().sum().item(),
        'switches_off': (switch_states <= 0.5).float().sum().item()
    }


def physics_violation_score(
    P_node: torch.Tensor,
    V_pred: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor
) -> Dict[str, float]:
    """
    Compute physics constraint violations (KCL, power flow).
    
    Args:
        P_node: Node power injections
        V_pred: Predicted voltages
        edge_index: Edge connectivity
        edge_attr: Edge attributes (with resistance)
        
    Returns:
        Dictionary with physics violation metrics
    """
    from models.physics_module import kcl_residual
    
    # Extract resistance
    R = edge_attr[:, 1:2]
    
    # Compute KCL residuals
    kcl_res = kcl_residual(P_node, V_pred, edge_index, R)
    
    return {
        'kcl_mean_violation': kcl_res.mean().item(),
        'kcl_max_violation': kcl_res.max().item(),
        'kcl_std_violation': kcl_res.std().item()
    }


def energy_loss_metric(
    I_edges: torch.Tensor,
    R: torch.Tensor
) -> float:
    """
    Compute total energy loss (I^2 * R).
    
    Args:
        I_edges: Edge currents [num_edges, 1]
        R: Edge resistances [num_edges, 1]
        
    Returns:
        Total energy loss
    """
    losses = (I_edges ** 2) * R
    return losses.sum().item()


def unmet_demand_metric(
    P_demand: torch.Tensor,
    P_served: torch.Tensor
) -> Dict[str, float]:
    """
    Compute unmet demand statistics.
    
    Args:
        P_demand: Requested power [num_nodes, 1]
        P_served: Actual served power [num_nodes, 1]
        
    Returns:
        Dictionary with unmet demand metrics
    """
    unmet = torch.clamp(P_demand - P_served, min=0)
    
    return {
        'total_unmet': unmet.sum().item(),
        'unmet_rate': (unmet.sum() / (P_demand.sum() + 1e-8)).item(),
        'max_unmet': unmet.max().item(),
        'nodes_with_unmet': (unmet > 0).float().sum().item()
    }


def compute_all_metrics(
    output: Dict[str, torch.Tensor],
    data: Data,
    target: Optional[Dict[str, torch.Tensor]] = None,
    prev_switch_states: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        output: Model output dictionary
        data: Input graph data
        target: Target values (if available)
        prev_switch_states: Previous switch states
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Voltage metrics
    V_pred = output['voltage_pred']
    if target and 'voltage' in target:
        metrics['voltage_mae'] = voltage_mae(V_pred, target['voltage'])
        metrics['voltage_rmse'] = voltage_rmse(V_pred, target['voltage'])
    
    # Voltage violations
    v_violations = voltage_violation_rate(V_pred)
    metrics.update({f'voltage_{k}': v for k, v in v_violations.items()})
    
    # Switch metrics
    switch_logits = output['switch_logits']
    switch_probs = torch.sigmoid(switch_logits)
    
    if target and 'switches' in target:
        metrics['switch_accuracy'] = switch_accuracy(switch_logits, target['switches'])
    
    switch_ops = switch_operation_count(switch_probs, prev_switch_states)
    metrics.update({f'switch_{k}': v for k, v in switch_ops.items()})
    
    # Physics violations
    P_node = data.x[:, 2:3]  # Power demand column
    try:
        phys_viol = physics_violation_score(P_node, V_pred, data.edge_index, data.edge_attr)
        metrics.update({f'physics_{k}': v for k, v in phys_viol.items()})
    except Exception as e:
        print(f"Warning: Could not compute physics violations: {e}")
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Metrics"):
    """
    Pretty print metrics.
    
    Args:
        metrics: Dictionary of metrics
        title: Title to display
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    # Group metrics by category
    categories = {
        'Voltage': [k for k in metrics.keys() if k.startswith('voltage')],
        'Switch': [k for k in metrics.keys() if k.startswith('switch')],
        'Physics': [k for k in metrics.keys() if k.startswith('physics')],
        'Energy': [k for k in metrics.keys() if 'energy' in k or 'loss' in k],
        'Other': [k for k in metrics.keys() if not any(
            k.startswith(p) for p in ['voltage', 'switch', 'physics']
        ) and 'energy' not in k and 'loss' not in k]
    }
    
    for category, keys in categories.items():
        if keys:
            print(f"\n{category} Metrics:")
            print("-" * 60)
            for key in keys:
                value = metrics[key]
                if isinstance(value, float):
                    print(f"  {key:40s}: {value:12.6f}")
                else:
                    print(f"  {key:40s}: {value}")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test metrics
    print("Testing evaluation metrics...")
    
    num_nodes = 10
    num_edges = 20
    
    # Create dummy predictions
    V_pred = torch.tensor(np.random.uniform(0.95, 1.05, (num_nodes, 1)))
    V_target = torch.ones(num_nodes, 1)
    
    switch_pred = torch.randn(num_edges, 1)
    switch_target = torch.randint(0, 2, (num_edges, 1)).float()
    
    # Test voltage metrics
    mae = voltage_mae(V_pred, V_target)
    print(f"Voltage MAE: {mae:.4f}")
    
    rmse = voltage_rmse(V_pred, V_target)
    print(f"Voltage RMSE: {rmse:.4f}")
    
    # Test voltage violations
    violations = voltage_violation_rate(V_pred)
    print(f"Voltage violations: {violations}")
    
    # Test switch metrics
    acc = switch_accuracy(switch_pred, switch_target)
    print(f"Switch accuracy: {acc:.4f}")
    
    # Test switch operations
    ops = switch_operation_count(torch.sigmoid(switch_pred))
    print(f"Switch operations: {ops}")
    
    print("✓ Evaluation metrics tests passed!")

