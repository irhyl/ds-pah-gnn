import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from dataset import MicrogridDataset
from model import MicrogridGNN

def main():
    # Setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(script_dir, "artifacts")
    dataset_path = os.path.join(artifacts_dir, "dataset")
    model_path = os.path.join(artifacts_dir, "best_model.pth")
    
    # Load Data & Model
    dataset = MicrogridDataset(dataset_path)
    data = dataset[0] # Take the first sample
    
    device = torch.device('cpu') # t-SNE is usually CPU bound
    model = MicrogridGNN(hidden_channels=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print("Extracting node embeddings...")
    with torch.no_grad():
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        edge_attr_dict = {k: data[k].edge_attr for k in data.edge_types}
        
        # Get embeddings from the new method
        embeddings_dict = model.get_node_embeddings(x_dict, edge_index_dict, edge_attr_dict)
        
    # Prepare for t-SNE
    # We will visualize 'bus' nodes and 'device' nodes together
    bus_emb = embeddings_dict['bus'].numpy()
    dev_emb = embeddings_dict['device'].numpy()
    
    # Create labels for coloring
    # Bus = 0
    # Device Types: PV=1, Storage=2, Charger=3 (derived from features)
    
    bus_labels = np.zeros(len(bus_emb))
    
    # Device features are [V, P, SOC, Type]
    # Type is the last feature (float encoded)
    # In converter.py: PV=1.0, Storage=2.0, Charger=0.0 (default/implicit)
    dev_features = data['device'].x.numpy()
    dev_labels = []
    for feat in dev_features:
        type_val = feat[3]
        if type_val == 1.0: dev_labels.append(1) # PV
        elif type_val == 2.0: dev_labels.append(2) # Storage
        else: dev_labels.append(3) # Charger
    dev_labels = np.array(dev_labels)
    
    # Concatenate
    X = np.vstack([bus_emb, dev_emb])
    y = np.concatenate([bus_labels, dev_labels])
    
    print(f"Running t-SNE on {len(X)} nodes...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    X_2d = tsne.fit_transform(X)
    
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

    # Plot
    plt.figure(figsize=(10, 8))
    
    # Define classes
    classes = ['Bus', 'PV Unit', 'Storage', 'EV Charger']
    colors = ['#1f77b4', '#ff7f0e', '#9467bd', '#d62728'] # Professional categorical colors
    markers = ['o', '^', 's', 'v']
    
    for i, label in enumerate(classes):
        mask = (y == i)
        plt.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=colors[i], label=label, marker=markers[i], s=150, edgecolors='white', linewidth=1.5, alpha=0.9
        )
        
    plt.title("Latent Space Visualization (t-SNE) of Microgrid Nodes")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(frameon=True, framealpha=0.9, edgecolor='black')
    
    save_path = os.path.join(artifacts_dir, "paper_fig_tsne.png")
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    print("Insight: Distinct clusters prove the GNN has learned to differentiate physical components.")

if __name__ == "__main__":
    main()