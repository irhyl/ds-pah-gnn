import pickle
import os
import csv
import json
import glob

def main():
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "artifacts", "dataset")
    csv_dir = os.path.join(dataset_dir, "csv")
    
    # Create output directory
    os.makedirs(csv_dir, exist_ok=True)
    
    # Find all pickle shards
    shard_pattern = os.path.join(dataset_dir, "shard_*.pkl")
    shard_files = glob.glob(shard_pattern)
    
    if not shard_files:
        print(f"No .pkl shard files found in {dataset_dir}")
        print("Make sure you have generated the dataset first.")
        return

    print(f"Found {len(shard_files)} shards. Converting to CSV format...")

    output_csv_path = os.path.join(csv_dir, "full_dataset.csv")
    
    # Define CSV headers
    # We flatten key metrics and keep complex structures as JSON strings
    headers = [
        'sample_id', 
        'total_loss_pu', 
        'min_voltage_pu', 
        'max_voltage_pu', 
        'total_load_pu', 
        'node_count',
        'edge_count',
        'nodes_json', 
        'edges_json'
    ]

    print(f"Writing to {output_csv_path} ...")

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=headers)
        writer.writeheader()

        for pkl_path in shard_files:
            filename = os.path.basename(pkl_path)
            print(f"Processing {filename} ...")
            
            try:
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
                    
                if not data:
                    print(f"  Skipping empty shard: {filename}")
                    continue
                
                for sample in data:
                    # Extract basic stats
                    nodes = sample.get('nodes', [])
                    edges = sample.get('edges', [])
                    
                    voltages = [n.get('v', 1.0) for n in nodes]
                    powers = [n.get('p', 0.0) for n in nodes]
                    
                    min_v = min(voltages) if voltages else 0.0
                    max_v = max(voltages) if voltages else 0.0
                    total_p = sum(powers)
                    
                    row = {
                        'sample_id': sample.get('id'),
                        'total_loss_pu': sample.get('loss'),
                        'min_voltage_pu': f"{min_v:.6f}",
                        'max_voltage_pu': f"{max_v:.6f}",
                        'total_load_pu': f"{total_p:.6f}",
                        'node_count': len(nodes),
                        'edge_count': len(edges),
                        'nodes_json': json.dumps(nodes),
                        'edges_json': json.dumps(edges)
                    }
                    writer.writerow(row)
                        
            except Exception as e:
                print(f"  Error converting {filename}: {e}")

    print(f"\nConversion complete! CSV file is located at:\n{output_csv_path}")

if __name__ == "__main__":
    main()