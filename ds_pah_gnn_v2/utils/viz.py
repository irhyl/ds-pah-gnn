"""
Visualization utilities for DS-PAH-GNN.

Includes:
- Network topology visualization
- Attention heatmaps
- Training curves
- Voltage/current profiles
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import Optional, List, Dict
import networkx as nx
from torch_geometric.utils import to_networkx


def plot_network_topology(
    data,
    node_colors: Optional[np.ndarray] = None,
    edge_colors: Optional[np.ndarray] = None,
    title: str = "Power Distribution Network",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8)
):
    """
    Plot network topology with optional coloring.
    
    Args:
        data: PyG Data object
        node_colors: Node colors (e.g., voltages)
        edge_colors: Edge colors (e.g., switch states)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Convert to NetworkX
    G = to_networkx(data, to_undirected=True)
    
    # Layout
    pos = nx.spring_layout(G, seed=42, k=1, iterations=50)
    
    # Node colors
    if node_colors is None:
        if hasattr(data, 'x') and data.x.shape[1] > 0:
            node_colors = data.x[:, 0].cpu().numpy()
        else:
            node_colors = 'lightblue'
    
    # Draw network
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=500,
        cmap='viridis',
        alpha=0.9
    )
    
    nx.draw_networkx_edges(
        G, pos,
        edge_color='gray' if edge_colors is None else edge_colors,
        width=2,
        alpha=0.6
    )
    
    nx.draw_networkx_labels(
        G, pos,
        font_size=8,
        font_color='white'
    )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_attention_heatmap(
    attention_weights: torch.Tensor,
    node_labels: Optional[List[str]] = None,
    title: str = "Attention Weights",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8)
):
    """
    Plot attention weight heatmap.
    
    Args:
        attention_weights: [num_nodes, 1] or [num_nodes, num_nodes]
        node_labels: Node labels
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    attn_np = attention_weights.detach().cpu().numpy()
    
    if attn_np.ndim == 2 and attn_np.shape[1] == 1:
        # Single attention vector
        attn_np = attn_np.squeeze()
        plt.bar(range(len(attn_np)), attn_np, color='steelblue', alpha=0.7)
        plt.xlabel('Node Index')
        plt.ylabel('Attention Weight')
        plt.title(title)
        
    else:
        # Attention matrix
        sns.heatmap(
            attn_np,
            cmap='viridis',
            annot=False,
            cbar=True,
            xticklabels=node_labels,
            yticklabels=node_labels
        )
        plt.title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    metrics: Optional[Dict[str, List[float]]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5)
):
    """
    Plot training and validation curves.
    
    Args:
        train_losses: Training losses
        val_losses: Validation losses
        metrics: Additional metrics to plot
        save_path: Path to save figure
        figsize: Figure size
    """
    n_plots = 2 if metrics is None else 2 + len(metrics)
    fig, axes = plt.subplots(1, min(n_plots, 3), figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]
    
    # Loss curves
    ax = axes[0]
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Train', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Additional metrics
    if metrics:
        for idx, (metric_name, values) in enumerate(metrics.items()):
            if idx + 1 < len(axes):
                ax = axes[idx + 1]
                ax.plot(epochs[:len(values)], values, linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric_name)
                ax.set_title(metric_name.replace('_', ' ').title())
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_voltage_profile(
    voltages: torch.Tensor,
    node_indices: Optional[List[int]] = None,
    V_min: float = 0.95,
    V_max: float = 1.05,
    title: str = "Voltage Profile",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6)
):
    """
    Plot voltage profile across nodes.
    
    Args:
        voltages: [num_nodes] or [num_nodes, timesteps]
        node_indices: Node indices to plot
        V_min: Minimum voltage limit
        V_max: Maximum voltage limit
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    V_np = voltages.detach().cpu().numpy()
    
    if V_np.ndim == 1:
        # Single time step
        if node_indices is None:
            node_indices = range(len(V_np))
        
        plt.plot(node_indices, V_np, 'bo-', linewidth=2, markersize=6, label='Voltage')
        plt.axhline(y=V_min, color='r', linestyle='--', label=f'Min ({V_min} p.u.)')
        plt.axhline(y=V_max, color='r', linestyle='--', label=f'Max ({V_max} p.u.)')
        plt.fill_between(node_indices, V_min, V_max, alpha=0.2, color='green', label='Safe zone')
        
        plt.xlabel('Node Index')
        plt.ylabel('Voltage (p.u.)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    else:
        # Multiple time steps
        num_nodes, timesteps = V_np.shape
        if node_indices is None:
            node_indices = list(range(min(10, num_nodes)))  # Plot first 10 nodes
        
        for node_idx in node_indices:
            plt.plot(range(timesteps), V_np[node_idx, :], label=f'Node {node_idx}')
        
        plt.axhline(y=V_min, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=V_max, color='r', linestyle='--', alpha=0.5)
        
        plt.xlabel('Time Step')
        plt.ylabel('Voltage (p.u.)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_switch_operations(
    switch_states: torch.Tensor,
    timesteps: Optional[List[int]] = None,
    title: str = "Switch Operations Over Time",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6)
):
    """
    Plot switch operations over time.
    
    Args:
        switch_states: [num_edges, timesteps] - binary switch states
        timesteps: Time step labels
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    states_np = switch_states.detach().cpu().numpy()
    
    if timesteps is None:
        timesteps = range(states_np.shape[1])
    
    # Heatmap of switch states
    plt.imshow(states_np, aspect='auto', cmap='RdYlGn', interpolation='nearest')
    plt.colorbar(label='Switch State (0=Off, 1=On)')
    plt.xlabel('Time Step')
    plt.ylabel('Switch Index')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization utilities...")
    
    from torch_geometric.data import Data
    
    # Create dummy data
    num_nodes = 10
    x = torch.randn(num_nodes, 4)
    edge_index = torch.randint(0, num_nodes, (2, 20))
    edge_attr = torch.randn(20, 3)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    # Test network plot
    print("Plotting network topology...")
    plot_network_topology(data, title="Test Network")
    
    # Test attention heatmap
    print("Plotting attention weights...")
    attention = torch.softmax(torch.randn(num_nodes, 1), dim=0)
    plot_attention_heatmap(attention, title="Test Attention")
    
    # Test voltage profile
    print("Plotting voltage profile...")
    voltages = torch.tensor(np.random.uniform(0.95, 1.05, num_nodes))
    plot_voltage_profile(voltages, title="Test Voltage Profile")
    
    print("✓ Visualization tests passed!")

