"""
Visualization utilities for microgrid networks and analysis results.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from typing import Dict, Optional, List, Tuple


def plot_network(
    graph: nx.Graph,
    title: str = "Microgrid Topology",
    node_size: int = 500,
    figsize: Tuple[int, int] = (12, 8),
    show_labels: bool = True,
    layout: str = "spring"
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot microgrid network topology.
    
    Parameters
    ----------
    graph : nx.Graph
        Microgrid network
    title : str
        Plot title
    node_size : int
        Node size for visualization
    figsize : tuple
        Figure size
    show_labels : bool
        Whether to show node labels
    layout : str
        Layout algorithm ('spring', 'circular', 'kamada_kawai')
        
    Returns
    -------
    fig : matplotlib.Figure
    ax : matplotlib.Axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Choose layout
    if layout == "spring":
        pos = nx.spring_layout(graph, k=2, iterations=50)
    elif layout == "circular":
        pos = nx.circular_layout(graph)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(graph)
    else:
        pos = nx.spring_layout(graph)
    
    # Node colors based on generation
    node_colors = []
    for node in graph.nodes():
        gen = graph.nodes[node].get('generation', 0)
        if gen > 50:
            node_colors.append('red')  # Generator
        elif gen > 0:
            node_colors.append('orange')  # Renewable
        else:
            node_colors.append('lightblue')  # Load
    
    # Draw network
    nx.draw_networkx_nodes(
        graph, pos,
        node_color=node_colors,
        node_size=node_size,
        ax=ax,
        alpha=0.8
    )
    
    nx.draw_networkx_edges(
        graph, pos,
        width=2,
        ax=ax,
        alpha=0.6
    )
    
    if show_labels:
        nx.draw_networkx_labels(
            graph, pos,
            font_size=8,
            ax=ax
        )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    
    return fig, ax


def plot_results(
    results: Dict,
    figsize: Tuple[int, int] = (15, 10)
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Plot power flow analysis results.
    
    Parameters
    ----------
    results : dict
        Power flow results from run_powerflow_analysis
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : matplotlib.Figure
    axes : list of matplotlib.Axes
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Plot 1: Voltage profile
    voltages = results['voltages']
    axes[0].plot(voltages, 'o-', linewidth=2, markersize=8)
    axes[0].axhline(y=1.0, color='r', linestyle='--', label='Nominal (1.0 pu)')
    axes[0].axhline(y=0.95, color='orange', linestyle='--', label='Lower limit')
    axes[0].axhline(y=1.05, color='orange', linestyle='--', label='Upper limit')
    axes[0].set_xlabel('Node')
    axes[0].set_ylabel('Voltage (pu)')
    axes[0].set_title('Bus Voltage Profile')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Branch power flows
    flows = [flow['power'] for flow in results['branch_flows']]
    axes[1].bar(range(len(flows)), flows, color='steelblue', alpha=0.7)
    axes[1].set_xlabel('Branch')
    axes[1].set_ylabel('Power Flow (MW)')
    axes[1].set_title('Branch Power Flows')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Branch losses
    losses = [flow['loss'] for flow in results['branch_flows']]
    axes[2].bar(range(len(losses)), losses, color='coral', alpha=0.7)
    axes[2].set_xlabel('Branch')
    axes[2].set_ylabel('Loss (MW)')
    axes[2].set_title(f'Branch Losses (Total: {results["total_loss"]:.3f} MW)')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Convergence info
    axes[3].text(0.5, 0.7, f"Convergence Status", ha='center', fontsize=12, fontweight='bold')
    axes[3].text(0.5, 0.5, f"Converged: {results['converged']}", ha='center', fontsize=11)
    axes[3].text(0.5, 0.35, f"Iterations: {results['iterations']}", ha='center', fontsize=11)
    axes[3].text(0.5, 0.2, f"Max Mismatch: {np.max(results['voltage_mismatch']):.6f}", ha='center', fontsize=11)
    axes[3].axis('off')
    
    plt.tight_layout()
    return fig, axes


def plot_constraints(
    violations: Dict,
    figsize: Tuple[int, int] = (14, 5)
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Plot constraint violations.
    
    Parameters
    ----------
    violations : dict
        Constraint violations from check_constraints
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : matplotlib.Figure
    axes : list of matplotlib.Axes
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Voltage violations
    if violations['voltage_violations']:
        nodes = [v['node'] for v in violations['voltage_violations']]
        voltages = [v['voltage'] for v in violations['voltage_violations']]
        axes[0].bar(range(len(nodes)), voltages, color='red', alpha=0.7)
        axes[0].set_xlabel('Node')
        axes[0].set_ylabel('Voltage (pu)')
        axes[0].set_title(f"Voltage Violations ({len(nodes)})")
        axes[0].axhline(y=0.95, color='orange', linestyle='--')
        axes[0].axhline(y=1.05, color='orange', linestyle='--')
    else:
        axes[0].text(0.5, 0.5, 'No Violations', ha='center', va='center', fontsize=12)
        axes[0].set_title('Voltage Violations')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Thermal violations
    if violations['thermal_violations']:
        edges = [str(v['edge']) for v in violations['thermal_violations']]
        powers = [v['power'] for v in violations['thermal_violations']]
        axes[1].bar(range(len(edges)), powers, color='red', alpha=0.7)
        axes[1].set_xlabel('Edge')
        axes[1].set_ylabel('Power (MW)')
        axes[1].set_title(f"Thermal Violations ({len(edges)})")
        axes[1].set_xticklabels(edges, rotation=45, ha='right')
    else:
        axes[1].text(0.5, 0.5, 'No Violations', ha='center', va='center', fontsize=12)
        axes[1].set_title('Thermal Violations')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Summary
    total_violations = (
        len(violations['voltage_violations']) +
        len(violations['thermal_violations']) +
        len(violations['frequency_violations'])
    )
    axes[2].text(0.5, 0.7, 'Constraint Violations Summary', ha='center', fontsize=12, fontweight='bold')
    axes[2].text(0.5, 0.5, f"Voltage: {len(violations['voltage_violations'])}", ha='center', fontsize=11)
    axes[2].text(0.5, 0.35, f"Thermal: {len(violations['thermal_violations'])}", ha='center', fontsize=11)
    axes[2].text(0.5, 0.2, f"Frequency: {len(violations['frequency_violations'])}", ha='center', fontsize=11)
    axes[2].text(0.5, 0.02, f"Total: {total_violations}", ha='center', fontsize=11, fontweight='bold', color='red' if total_violations > 0 else 'green')
    axes[2].axis('off')
    axes[2].set_title('Summary')
    
    plt.tight_layout()
    return fig, axes


def plot_scenario_comparison(
    scenario_results: List[Dict],
    metric: str = "total_loss",
    figsize: Tuple[int, int] = (10, 6)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Compare a metric across multiple scenarios.
    
    Parameters
    ----------
    scenario_results : list of dict
        Results from multiple power flow runs
    metric : str
        Metric to compare ('total_loss', 'convergence', etc.)
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : matplotlib.Figure
    ax : matplotlib.Axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    values = [result.get(metric, 0) for result in scenario_results]
    
    ax.bar(range(len(values)), values, color='steelblue', alpha=0.7)
    ax.set_xlabel('Scenario')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'{metric.replace("_", " ").title()} Across Scenarios')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig, ax
