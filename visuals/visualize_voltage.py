import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from microgrid import DualBusMicrogrid
from model import MicrogridGNN
from converter import raw_to_pyg
from optimize import generate_valid_topologies, get_snapshot_from_graph

def main():
    # Setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(script_dir, "artifacts")
    config_path = os.path.join(script_dir, "config.yaml")
    model_path = os.path.join(artifacts_dir, "best_model.pth")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MicrogridGNN(hidden_channels=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 1. Create Stressed Scenario
    env = DualBusMicrogrid(config)
    env.update_stochastic_loads()
    
    # Apply heavy load to Station 1 (Node 1)
    print("Applying stress test (Heavy Load at Station 1)...")
    for u, v, attr in env.graph.edges(data=True):
        if u == 1 and env.graph.nodes[v]['type'] == 'charger':
            env.graph.nodes[v]['power'] = -2.0
        elif v == 1 and env.graph.nodes[u]['type'] == 'charger':
            env.graph.nodes[u]['power'] = -2.0
            
    # 2. Solve Baseline (Default Radial Topology)
    # Reset switches
    for u, v, d in env.graph.edges(data=True):
        if d['type'] == 'feeder': d['status'] = 1
        if d['type'] == 'tie': d['status'] = 0
        
    nodes_base, _, _ = env.solve_power_flow()
    voltages_base = [n['v'] for n in nodes_base]
    ids = [n['id'] for n in nodes_base]
    
    # 3. Find Optimized Topology via GNN
    candidates = generate_valid_topologies(env.graph)
    best_pred = float('inf')
    best_cand = None
    
    for cand in candidates:
        snapshot = get_snapshot_from_graph(cand['graph'])
        data = raw_to_pyg(snapshot)
        with torch.no_grad():
            x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
            edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
            edge_attr_dict = {k: data[k].edge_attr.to(device) for k in data.edge_types}
            pred = model(x_dict, edge_index_dict, edge_attr_dict, batch=None).item()
        
        if pred < best_pred:
            best_pred = pred
            best_cand = cand
            
    # 4. Solve Optimized Topology (Physics Ground Truth)
    temp_env = DualBusMicrogrid(config)
    temp_env.graph = best_cand['graph']
    # Copy loads
    for n, d in env.graph.nodes(data=True):
        temp_env.graph.nodes[n]['power'] = d['power']
        if 'soc' in d: temp_env.graph.nodes[n]['soc'] = d['soc']
    
    # Update cache
    temp_env.nodes = list(temp_env.graph.nodes(data=True))
    temp_env.edges = list(temp_env.graph.edges(data=True))
    
    nodes_opt, _, _ = temp_env.solve_power_flow()
    voltages_opt = [n['v'] for n in nodes_opt]

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
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2
    })

    # 5. Plot Comparison
    plt.figure(figsize=(12, 6))
    plt.plot(ids, voltages_base, 'o--', color='#d62728', label='Baseline (Passive)', markersize=6)
    plt.plot(ids, voltages_opt, 's-', color='#2ca02c', label='DS-PAH-GNN (Optimized)', markersize=8)
    
    plt.axhline(1.0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='Nominal (1.0 p.u.)')
    plt.axhline(0.95, color='grey', linestyle=':', linewidth=1.5, label='Min Limit (0.95 p.u.)')
    
    plt.xlabel('Node ID')
    plt.ylabel('Voltage Magnitude (p.u.)')
    plt.title('Figure 7: Nodal Voltage Profile Improvement')
    plt.legend()
    
    save_path = os.path.join(artifacts_dir, "paper_fig_voltage.png")
    plt.savefig(save_path)
    print(f"Saved {save_path}")

if __name__ == "__main__":
    main()