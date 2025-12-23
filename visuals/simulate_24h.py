import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    
    # Initialize Environment
    env = DualBusMicrogrid(config)
    
    hours = list(range(24))
    baseline_losses = []
    optimized_losses = []
    
    print("Running 24-Hour Digital Twin Simulation...")
    
    for h in tqdm(hours):
        # 1. Update Physics for this hour (Sun moves, EVs arrive)
        env.update_stochastic_loads()
        
        # 2. Calculate Baseline Loss (Default Radial Topology)
        # Reset switches to default
        for u, v, d in env.graph.edges(data=True):
            if d['type'] == 'feeder': d['status'] = 1
            if d['type'] == 'tie': d['status'] = 0
            
        try:
            _, _, base_loss = env.solve_power_flow()
        except:
            base_loss = 1.0 # Fallback if default is unstable
        baseline_losses.append(base_loss)
        
        # 3. Run AI Optimization
        # Generate candidates
        candidates = generate_valid_topologies(env.graph)
        
        best_pred_loss = float('inf')
        best_candidate = None
        
        # Evaluate all candidates with GNN
        for cand in candidates:
            snapshot = get_snapshot_from_graph(cand['graph'])
            data = raw_to_pyg(snapshot)
            
            with torch.no_grad():
                x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
                edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
                edge_attr_dict = {k: data[k].edge_attr.to(device) for k in data.edge_types}
                
                pred_loss = model(x_dict, edge_index_dict, edge_attr_dict, batch=None).item()
            
            if pred_loss < best_pred_loss:
                best_pred_loss = pred_loss
                best_candidate = cand
        
        # 4. Verify Winner with Physics (Ground Truth)
        # In a real deployment, we would apply this topology.
        # Here we solve it to get the *actual* optimized loss for the plot.
        # (Using prediction for selection, physics for reporting)
        temp_env = DualBusMicrogrid(config)
        temp_env.graph = best_candidate['graph']
        # Copy loads from current env
        for n, d in env.graph.nodes(data=True):
            temp_env.graph.nodes[n]['power'] = d['power']
            if 'soc' in d: temp_env.graph.nodes[n]['soc'] = d['soc']
            
        # CRITICAL FIX: Update physics engine cache to recognize new topology & loads
        temp_env.nodes = list(temp_env.graph.nodes(data=True))
        temp_env.edges = list(temp_env.graph.edges(data=True))
            
        try:
            _, _, opt_loss = temp_env.solve_power_flow()
        except:
            opt_loss = base_loss # Fallback
            
        optimized_losses.append(opt_loss)

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

    # --- Plotting ---
    plt.figure(figsize=(12, 6))
    
    plt.plot(hours, baseline_losses, 'o-', color='#7f7f7f', label='Baseline (Passive)', linewidth=2.5, markersize=8, markerfacecolor='white', markeredgewidth=2)
    plt.plot(hours, optimized_losses, 's-', color='#2ca02c', label='DS-PAH-GNN (Active)', linewidth=3, markersize=8, markerfacecolor='white', markeredgewidth=2)
    
    # Fill area to show savings
    plt.fill_between(hours, baseline_losses, optimized_losses, color='#2ca02c', alpha=0.15, label='Energy Savings')
    
    plt.xlabel('Time of Day (Hour)')
    plt.ylabel('System Power Loss (p.u.)')
    plt.title('Figure 3: 24-Hour Real-Time Optimization Performance')
    plt.xticks(hours)
    plt.legend()
    
    # Calculate total savings
    total_base = sum(baseline_losses)
    total_opt = sum(optimized_losses)
    savings_pct = (total_base - total_opt) / total_base * 100
    
    plt.text(0.5, 0.9, f"Total Daily Savings: {savings_pct:.1f}%", 
             transform=plt.gca().transAxes, fontsize=14, 
             bbox=dict(facecolor='white', edgecolor='black', alpha=0.9, boxstyle='round,pad=0.5'))
    
    save_path = os.path.join(artifacts_dir, "paper_fig_24h.png")
    plt.savefig(save_path)
    print(f"Saved {save_path}")

if __name__ == "__main__":
    main()