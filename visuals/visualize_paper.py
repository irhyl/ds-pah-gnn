import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import re

def main():
    # Setup Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    artifacts_dir = os.path.join(script_dir, "artifacts")
    csv_path = os.path.join(results_dir, "optimization_results.csv")
    
    # Ensure artifacts directory exists
    os.makedirs(artifacts_dir, exist_ok=True)
    
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}. Run optimize.py first.")
        return

    # Load Data
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse "Switches: 010101" from the Name field
            match = re.search(r'Switches: ([01]+)', row['Name'])
            if match:
                switches = [int(c) for c in match.group(1)]
                loss = float(row['Predicted_Loss'])
                data.append({'switches': switches, 'loss': loss})
    
    if not data:
        print("No data found in optimization_results.csv")
        return

    # Sort by loss (Lowest/Best first)
    data.sort(key=lambda x: x['loss'])
    
    # Extract for plotting
    losses = [d['loss'] for d in data]
    
    # Prepare Heatmap Data (Top 50 Candidates)
    top_n = min(50, len(data))
    top_data = data[:top_n]
    switch_matrix = np.array([d['switches'] for d in top_data])
    
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

    # --- Figure 4: Optimization Landscape (Histogram) ---
    plt.figure(figsize=(10, 6))
    plt.hist(losses, bins=30, color='#1f77b4', edgecolor='black', alpha=0.7)
    
    best_loss = losses[0]
    worst_loss = max(losses)
    
    plt.axvline(best_loss, color='#d62728', linestyle='--', linewidth=2, label=f'Optimal ({best_loss:.4f})')
    plt.axvline(worst_loss, color='grey', linestyle=':', linewidth=2, label=f'Worst ({worst_loss:.4f})')
    
    plt.title('Figure 4: Optimization Search Space')
    plt.xlabel('System Power Loss (p.u.)')
    plt.ylabel('Frequency (Number of Topologies)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    landscape_path = os.path.join(artifacts_dir, "paper_fig_landscape.png")
    plt.savefig(landscape_path, bbox_inches='tight')
    print(f"Saved {landscape_path}")

    # --- Figure 5: Topology DNA (Heatmap) ---
    plt.figure(figsize=(12, 8))
    
    # 0 = Open (Light), 1 = Closed (Dark Blue)
    plt.imshow(switch_matrix, aspect='auto', cmap='Blues', interpolation='nearest')
    
    cbar = plt.colorbar()
    cbar.set_label('Switch State (0=Open, 1=Closed)', rotation=270, labelpad=15)
    
    plt.title(f'Figure 5: Topology DNA (Top {top_n} Configurations)')
    plt.xlabel('Switch ID')
    plt.ylabel('Candidate Rank (0 = Optimal)')
    plt.xticks(range(switch_matrix.shape[1]))
    
    heatmap_path = os.path.join(artifacts_dir, "paper_fig_heatmap.png")
    plt.savefig(heatmap_path, bbox_inches='tight')
    print(f"Saved {heatmap_path}")

if __name__ == "__main__":
    main()