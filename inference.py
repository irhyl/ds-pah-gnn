import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from dataset import MicrogridDataset
from model import MicrogridGNN

def main():
    # 1. Setup Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(script_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    dataset_path = os.path.join(artifacts_dir, "dataset")
    model_path = os.path.join(artifacts_dir, "best_model.pth")
    
    # 2. Load Data
    print("Loading dataset...")
    dataset = MicrogridDataset(dataset_path)
    
    # Split to get the validation set, same as in train.py
    train_size = int(0.8 * len(dataset))
    val_dataset = dataset[train_size:]
    
    if not val_dataset:
        print("Error: Validation set is empty. Ensure dataset is large enough.")
        return

    # 3. Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MicrogridGNN(hidden_channels=64).to(device)
    
    if not os.path.exists(model_path):
        print("Error: best_model.pth not found. Run train.py first.")
        return
        
    print(f"Loading model from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 4. Full Validation Evaluation
    print(f"Evaluating on validation set ({len(val_dataset)} samples)...")
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for data in tqdm(val_dataset):
            
            # Move data to device
            x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
            edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
            edge_attr_dict = {k: data[k].edge_attr.to(device) for k in data.edge_types}
            
            # Run Forward Pass
            prediction = model(x_dict, edge_index_dict, edge_attr_dict, batch=None)
            
            y_true.append(data.y_loss.item())
            y_pred.append(prediction.item())

    # 5. Scientific Reporting and Saving
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Scientific Baseline (Mean Predictor)
    # "How well would we do if we just guessed the average loss every time?"
    baseline_preds = np.full_like(y_true, np.mean(y_true))
    baseline_rmse = np.sqrt(mean_squared_error(y_true, baseline_preds))

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

    # --- 6. Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot A: Scatter
    ax1.scatter(y_true, y_pred, alpha=0.6, color='#1f77b4', edgecolors='w', s=50)
    ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2.5, label='Ideal Fit')
    ax1.set_xlabel('Actual Loss (p.u.)')
    ax1.set_ylabel('Predicted Loss (p.u.)')
    ax1.set_title(f'Prediction Accuracy (R²={r2:.4f})')
    ax1.legend()
    
    # Plot B: Residuals
    residuals = y_true - y_pred
    ax2.hist(residuals, bins=50, color='#9467bd', edgecolor='black', alpha=0.8)
    ax2.axvline(0, color='black', linestyle='--', lw=2, label='Zero Error')
    ax2.set_xlabel('Residual Error (True - Pred)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Error Distribution (Residuals)')
    ax2.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(artifacts_dir, "inference_plot.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    # --- Create Report String ---
    report_lines = []
    report_lines.append("="*40)
    report_lines.append("   DS-PAH-GNN PERFORMANCE REPORT")
    report_lines.append("="*40)
    report_lines.append(f"Samples Evaluated: {len(val_dataset)}")
    report_lines.append(f"R² Score:          {r2:.4f} (Target: > 0.90)")
    report_lines.append(f"MAE (Mean Abs Err):{mae:.6f} p.u.")
    report_lines.append(f"RMSE:              {rmse:.6f} p.u.")
    report_lines.append(f"Baseline RMSE:     {baseline_rmse:.6f} p.u. (Naive Mean Predictor)")
    report_lines.append(f"Improvement:       {(1 - rmse/baseline_rmse)*100:.2f}% over baseline")
    report_lines.append("-" * 40)
    report_lines.append("Interpretation:")
    if r2 > 0.9:
        report_lines.append("Model has learned the physics of power flow.")
    else:
        report_lines.append("Model is not capturing the physics accurately.")
    report_lines.append("="*40)

    report_string = "\n".join(report_lines)
    
    # --- Print to Console and Save to File ---
    print("\n" + report_string)
    output_path = os.path.join(artifacts_dir, "inference_report.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_string)
    print(f"\nReport saved to {output_path}")

if __name__ == "__main__":
    main()