import os
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import MicrogridDataset
from model import MicrogridGNN

def train():
    # --- Configuration ---
    BATCH_SIZE = 32
    LR = 0.0005
    EPOCHS = 10
    
    # --- Setup Paths ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(script_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    dataset_path = os.path.join(artifacts_dir, "dataset")
    
    # --- Load Data ---
    print("Initializing Dataset...")
    dataset = MicrogridDataset(dataset_path)
    
    # Split: 80% Train, 20% Val
    # We slice sequentially to preserve disk locality (avoid thrashing shards)
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    
    # Loaders (shuffle=False is critical for sharded dataset performance)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- Initialize Model ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MicrogridGNN(hidden_channels=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.HuberLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    
    print(f"Training on {device} with {len(train_dataset)} training samples.")
    
    train_losses = []
    val_losses = []
    
    # --- Training Loop ---
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            edge_attr_dict = {k: batch[k].edge_attr for k in batch.edge_types}
            # Pass the batch vector for 'bus' nodes so the model knows which nodes belong to which graph
            out = model(batch.x_dict, batch.edge_index_dict, edge_attr_dict, batch['bus'].batch)
            
            target = batch.y_loss.view(-1, 1)
            loss = criterion(out, target)
            
            loss.backward()
            
            # Gradient Clipping to prevent instability/exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
            
        avg_train_loss = total_loss / len(train_dataset)
        train_losses.append(avg_train_loss)
        
        # --- Validation ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                edge_attr_dict = {k: batch[k].edge_attr for k in batch.edge_types}
                out = model(batch.x_dict, batch.edge_index_dict, edge_attr_dict, batch['bus'].batch)
                target = batch.y_loss.view(-1, 1)
                loss = criterion(out, target)
                total_val_loss += loss.item() * batch.num_graphs
                
        avg_val_loss = total_val_loss / len(val_dataset)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.8f}, Val Loss={avg_val_loss:.8f}")
        
        # Adjust Learning Rate if loss plateaus or spikes
        scheduler.step(avg_val_loss)
        
        # Save Best Model
        if epoch == 0 or avg_val_loss < min(val_losses[:-1]):
            torch.save(model.state_dict(), os.path.join(artifacts_dir, "best_model.pth"))
            
    # --- Plotting Training Curve ---
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
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss (p.u.)')
    plt.title('GNN Training Convergence')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(artifacts_dir, "training_loss.png"))
    print("Training Complete. Best model saved to 'best_model.pth'.")

if __name__ == "__main__":
    train()