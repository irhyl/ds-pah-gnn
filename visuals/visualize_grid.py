import os
import networkx as nx
import matplotlib.pyplot as plt

def visualize_topology(graph, title="Microgrid Topology"):
    """
    Visualizes the microgrid topology with color-coded components
    and saves the output to the artifacts directory.
    """
    # --- Setup Professional Plotting Style ---
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'axes.grid': False,
        'lines.linewidth': 2
    })

    plt.figure(figsize=(12, 8))
    
    # 1. Define Layout (Hierarchical)
    pos = {}
    
    roots = [n for n, d in graph.nodes(data=True) if d['type'] == 'root']
    stations = [n for n, d in graph.nodes(data=True) if d['type'] == 'station']
    
    # Root at top center
    for i, node in enumerate(roots):
        pos[node] = (0.5, 1.0)
        
    # Stations in middle row
    for i, node in enumerate(stations):
        pos[node] = ((i + 1) / (len(stations) + 1), 0.6)
        
    # Use spring layout for devices, but keep root/stations fixed
    fixed_nodes = roots + stations
    fixed_pos = {n: pos[n] for n in fixed_nodes}
    
    # Calculate positions for the rest
    pos = nx.spring_layout(graph, pos=fixed_pos, fixed=fixed_nodes, k=0.15, iterations=50)
    
    # 2. Draw Nodes
    colors = []
    for n, d in graph.nodes(data=True):
        if d['type'] == 'root': colors.append('gold')
        elif d['type'] == 'station': colors.append('skyblue')
        elif d['type'] == 'charger': colors.append('salmon')
        elif d['type'] == 'pv': colors.append('lightgreen')
        elif d['type'] == 'storage': colors.append('orchid')
        else: colors.append('lightgrey')
        
    nx.draw_networkx_nodes(graph, pos, node_color=colors, node_size=400, edgecolors='black')
    
    # 3. Draw Edges (Solid = Closed/Active, Dashed = Open/Inactive)
    active_edges = [(u, v) for u, v, d in graph.edges(data=True) if d['status'] == 1]
    inactive_edges = [(u, v) for u, v, d in graph.edges(data=True) if d['status'] == 0]
    
    nx.draw_networkx_edges(graph, pos, edgelist=active_edges, width=2.0, edge_color='black')
    nx.draw_networkx_edges(graph, pos, edgelist=inactive_edges, width=1.0, edge_color='grey', style='dashed', alpha=0.5)
    
    # 4. Save
    plt.title(title)
    plt.axis('off')
    
    save_path = os.path.join(os.path.dirname(__file__), "artifacts", "optimal_topology.png")
    plt.savefig(save_path)
    print(f"Topology visualization saved to {save_path}")