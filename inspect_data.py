import pickle
import os
import numpy as np

def main():
    # Path to the first shard
    # Use absolute path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "artifacts", "dataset", "shard_1.pkl")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        print("Make sure you have run 'generate.py' and that the dataset was created.")
        return

    print(f"Loading {file_path}...")
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    
    print(f"Loaded {len(data)} samples.")
    
    # Inspect the first sample
    sample = data[0]
    print("\n--- Sample 0 Analysis ---")
    print(f"ID: {sample['id']}")
    print(f"Total Loss: {sample['loss']:.6f} p.u.")
    
    voltages = [n['v'] for n in sample['nodes']]
    print(f"Voltage Range: {min(voltages):.4f} - {max(voltages):.4f} p.u.")
    
    powers = [n['p'] for n in sample['nodes']]
    print(f"Total Load: {sum(powers):.4f} p.u.")

if __name__ == "__main__":
    main()