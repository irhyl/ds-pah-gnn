import os
import time
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from microgrid import DualBusMicrogrid
from model import MicrogridGNN
from converter import raw_to_pyg

def main():
    # Setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(script_dir, "artifacts")
    config_path = os.path.join(script_dir, "config.yaml")
    model_path = os.path.join(artifacts_dir, "best_model.pth")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cpu') # Benchmark on CPU for fair comparison
    model = MicrogridGNN(hidden_channels=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    env = DualBusMicrogrid(config)
    env.update_stochastic_loads()
    
    iterations = 100
    print(f"Benchmarking over {iterations} iterations...")
    
    # 1. Benchmark Physics Solver
    start_phys = time.time()
    for _ in range(iterations):
        env.solve_power_flow()
    end_phys = time.time()
    avg_phys = (end_phys - start_phys) / iterations * 1000 # ms
    
    # 2. Benchmark GNN Inference
    # Pre-convert one sample to avoid benchmarking data loading time
    nodes, edges, loss = env.solve_power_flow()
    snapshot = {'nodes': nodes, 'edges': edges, 'loss': loss}
    data = raw_to_pyg(snapshot)
    
    start_gnn = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            x_dict = data.x_dict
            edge_index_dict = data.edge_index_dict
            edge_attr_dict = {k: data[k].edge_attr for k in data.edge_types}
            _ = model(x_dict, edge_index_dict, edge_attr_dict, batch=None)
    end_gnn = time.time()
    avg_gnn = (end_gnn - start_gnn) / iterations * 1000 # ms
    
    speedup = avg_phys / avg_gnn
    print(f"Physics: {avg_phys:.4f} ms/sample")
    print(f"GNN:     {avg_gnn:.4f} ms/sample")
    print(f"Speedup: {speedup:.2f}x")

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

    # 3. Plot
    plt.figure(figsize=(8, 6))
    labels = ['Physics Solver\n(Newton-Raphson)', 'DS-PAH-GNN\n(Inference)']
    times = [avg_phys, avg_gnn]
    colors = ['#7f7f7f', '#1f77b4']
    
    bars = plt.bar(labels, times, color=colors, edgecolor='black', alpha=0.8, width=0.5)
    
    plt.ylabel('Computation Time per Sample (ms)')
    plt.title(f'Figure 8: Computational Speedup ({speedup:.1f}x)')
    plt.yscale('log') # Log scale is crucial because GNN is so much faster
    plt.grid(True, axis='y', which='both', alpha=0.3)
    
    save_path = os.path.join(artifacts_dir, "paper_fig_speedup.png")
    plt.savefig(save_path)
    print(f"Saved {save_path}")

if __name__ == "__main__":
    main()