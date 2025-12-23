import os
import yaml
import numpy as np
import pickle
from tqdm import tqdm
from microgrid import DualBusMicrogrid

def main():
    # 1. Load Config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # 2. Setup Output
    if not os.path.isabs(config['output_dir']):
        config['output_dir'] = os.path.join(script_dir, config['output_dir'])
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Set seed for reproducibility (Research Standard)
    np.random.seed(42)
    
    # 3. Initialize Physics Engine
    env = DualBusMicrogrid(config)
    
    print(f"Starting Simulation: {config['total_samples']} samples...")
    
    dataset = []
    
    # 4. Simulation Loop
    for i in tqdm(range(config['total_samples'])):
        # A. Randomize Environment
        env.update_stochastic_loads()
        env.randomize_topology()
        
        # B. Solve Physics (Ground Truth)
        try:
            nodes, edges, loss = env.solve_power_flow()
        except Exception as e:
            # Skip invalid topologies (e.g., singular matrix/disconnected grid)
            continue
        
        # C. Store Snapshot
        # We store a dictionary for every time step
        snapshot = {
            'id': i,
            'nodes': nodes,   # Contains Voltages (Features)
            'edges': edges,   # Contains Connectivity & Resistance
            'loss': loss      # Label (for regression/reward)
        }
        dataset.append(snapshot)
        
        # D. Periodic Save (Sharding)
        # Save every 1000 samples to avoid memory overflow
        if (i + 1) % 1000 == 0:
            shard_id = (i + 1) // 1000
            save_path = os.path.join(config['output_dir'], f"shard_{shard_id}.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(dataset, f)
            dataset = [] # Clear memory

    # Save any remaining data that didn't fill a complete chunk
    if len(dataset) > 0:
        shard_id = (config['total_samples'] // 1000) + 1
        save_path = os.path.join(config['output_dir'], f"shard_{shard_id}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(dataset, f)

    print("Data Generation Complete.")

if __name__ == "__main__":
    main()