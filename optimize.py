import torch
import os
import yaml
import numpy as np
import networkx as nx
import time
import csv
from itertools import product
from microgrid import DualBusMicrogrid
from model import MicrogridGNN
from converter import raw_to_pyg
from visualize_grid import visualize_topology

def get_snapshot_from_graph(graph):
    """
    Extracts the current state of a graph into a dictionary format
    compatible with converter.py, without running the full physics solver.
    """
    nodes = []
    for n, d in graph.nodes(data=True):
        nodes.append({'id': n, 'v': d['voltage'], 'p': d['power'], 'type': d['type'], 'soc': d.get('soc', 0.0)})
        
    edges = []
    for u, v, d in graph.edges(data=True):
        edges.append({'u': u, 'v': v, 'r': d['r'], 'status': d['status']})
        
    return {'nodes': nodes, 'edges': edges, 'loss': 0.0} # Loss is unknown/dummy

def generate_valid_topologies(base_graph):
    """
    Generates all valid, connected topologies by flipping switchable edges.
    A topology is valid if the core bus network remains connected.
    """
    candidates = []
    
    # 1. Identify all switchable edges (feeders and ties)
    switchable_edges = []
    for u, v, d in base_graph.edges(data=True):
        if d.get('type') in ['feeder', 'tie']:
            switchable_edges.append((u, v))
            
    num_switches = len(switchable_edges)
    print(f"Found {num_switches} switchable edges. Evaluating {2**num_switches} possible configurations...")
    
    # 2. Iterate through all 2^N combinations of switch states
    for i, states in enumerate(product([0, 1], repeat=num_switches)):
        G_candidate = base_graph.copy()
        
        # Apply the new switch states
        for j, edge in enumerate(switchable_edges):
            u, v = edge
            G_candidate[u][v]['status'] = states[j]
            
        # 3. CRITICAL: Check for connectivity on the bus network
        bus_nodes = [n for n, d in G_candidate.nodes(data=True) if d['type'] in ['root', 'station']]
        
        # Build a temporary graph of just bus nodes and ACTIVE edges between them
        bus_subgraph = nx.Graph()
        bus_subgraph.add_nodes_from(bus_nodes)
        for u, v, d in G_candidate.edges(data=True):
            if d['status'] == 1 and u in bus_nodes and v in bus_nodes:
                bus_subgraph.add_edge(u, v)
                
        if nx.is_connected(bus_subgraph):
            name = f"Candidate {len(candidates)} (Switches: {''.join(map(str, states))})"
            candidates.append({'name': name, 'graph': G_candidate})
            
    print(f"Generated {len(candidates)} valid, connected topologies.")
    return candidates

def main():
    # 1. Setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(script_dir, "artifacts")
    config_path = os.path.join(script_dir, "config.yaml")
    model_path = os.path.join(artifacts_dir, "best_model.pth")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Load Model
    print("Loading GNN Model...")
    model = MicrogridGNN(hidden_channels=64).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("⚠️ Warning: No trained model found. Predictions will be random.")
    model.eval()

    # 3. Create Scenario: "Station 1 Overload"
    print("\n--- Creating Scenario: Station 1 Overload ---")
    base_env = DualBusMicrogrid(config)
    
    # Find station 1 ID dynamically
    station_nodes = [n for n, d in base_env.graph.nodes(data=True) if d['type'] == 'station']
    stn1 = station_nodes[0]
    
    chargers_at_station_1 = [n for n in base_env.graph.neighbors(stn1) if base_env.graph.nodes[n]['type'] == 'charger']
    
    total_load = 0
    for c_id in chargers_at_station_1:
        load = -2.0 # Very high load (normal is ~0.5)
        base_env.graph.nodes[c_id]['power'] = load
        total_load += load
        
    print(f"Injected {total_load} p.u. load at Station 1.")

    # 4. Generate ALL valid topology candidates
    candidates = generate_valid_topologies(base_env.graph)
    
    if not candidates:
        print("Error: No valid topologies could be generated.")
        return

    # 5. Evaluate Candidates
    print("\n--- Evaluating Topologies ---")
    results_data = []
    
    best_loss = float('inf')
    best_candidate_graph = None
    best_name = ""

    start_time = time.time()
    for i, cand in enumerate(candidates):
        env_graph = cand['graph']
        
        # Convert to PyG
        snapshot = get_snapshot_from_graph(env_graph)
        data = raw_to_pyg(snapshot)
        
        # Inference
        with torch.no_grad():
            x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
            edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
            edge_attr_dict = {k: data[k].edge_attr.to(device) for k in data.edge_types}
            pred_loss = model(x_dict, edge_index_dict, edge_attr_dict, batch=None).item()
            
        results_data.append({
            'Candidate_ID': i,
            'Name': cand['name'],
            'Predicted_Loss': pred_loss
        })
        
        if pred_loss < best_loss:
            print(f"New Best -> {cand['name']:<35} | Predicted Loss: {pred_loss:.6f}")
            best_loss = pred_loss
            best_name = cand['name']
            best_candidate_graph = env_graph
    
    end_time = time.time()
    duration = end_time - start_time

    print("-" * 50)
    print(f"⚡ Optimization Time: {duration*1000:.2f} ms ({len(candidates)} candidates)")
    print(f"🏆 Optimal Configuration: {best_name} (Loss: {best_loss:.6f})")

    # Save full results to CSV
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "optimization_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Candidate_ID', 'Name', 'Predicted_Loss'])
        writer.writeheader()
        writer.writerows(results_data)
    print(f"Full optimization results saved to {csv_path}")

    # 6. Visualize the winning topology
    if best_candidate_graph:
        print("\n--- Visualizing Optimal Topology ---")
        visualize_topology(best_candidate_graph, title=f"Optimal Topology: {best_name}")
    else:
        print("Could not determine an optimal topology.")

if __name__ == "__main__":
    main()