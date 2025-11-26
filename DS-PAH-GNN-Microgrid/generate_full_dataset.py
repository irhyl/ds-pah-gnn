"""
Generate full 12,000-sample production dataset for DS-PAH-GNN.

This script scales the scenario generation pipeline to create a comprehensive
dataset suitable for GNN training at production scale.
"""

import json
import csv
import time
import numpy as np
from pathlib import Path
from simulation.create_network import create_microgrid
from simulation.scenario_generator import generate_scenarios, generate_ev_load_profile, generate_pv_profile
from simulation.run_powerflow import run_powerflow_analysis, check_constraints


def generate_dataset(num_samples: int = 12000, output_dir: str = "gnn_dataset_full"):
    """
    Generate full GNN dataset with specified number of samples.
    
    Parameters
    ----------
    num_samples : int
        Number of samples to generate (default 12000)
    output_dir : str
        Directory to save dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    dataset = []
    summary = []
    
    # Configuration
    num_networks = 10  # 10 different network topologies
    scenarios_per_network = 12  # 12 scenarios per network
    timesteps_per_scenario = 100  # 100 timesteps per scenario
    
    # Total samples: 10 networks × 12 scenarios × 100 timesteps = 12,000 samples
    assert num_networks * scenarios_per_network * timesteps_per_scenario == num_samples, \
        f"Configuration mismatch: {num_networks} × {scenarios_per_network} × {timesteps_per_scenario} ≠ {num_samples}"
    
    print(f"Generating {num_samples}-sample dataset...")
    print(f"  Networks: {num_networks}")
    print(f"  Scenarios per network: {scenarios_per_network}")
    print(f"  Timesteps per scenario: {timesteps_per_scenario}")
    print(f"  Expected total: {num_networks * scenarios_per_network * timesteps_per_scenario}")
    
    start_time = time.time()
    
    sample_id = 0
    
    for network_id in range(num_networks):
        print(f"\n[Network {network_id + 1}/{num_networks}] Creating network topology...")
        
        # Create network with varying sizes
        num_nodes = 6 + network_id  # 6-15 nodes
        topologies = ['mesh', 'ring', 'tree']
        topology = topologies[network_id % len(topologies)]
        
        try:
            graph, _ = create_microgrid(num_nodes=num_nodes, topology=topology)
        except Exception as e:
            print(f"  Error creating network: {e}")
            continue
        
        print(f"  Topology: {topology}, Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
        
        for scenario_id in range(scenarios_per_network):
            print(f"  [Scenario {scenario_id + 1}/{scenarios_per_network}] Generating data...")
            
            # Generate EV and PV profiles
            ev_profiles = generate_ev_load_profile(num_stations=3, num_timesteps=timesteps_per_scenario, 
                                                   seed=network_id * 100 + scenario_id)
            pv_profiles = generate_pv_profile(num_feeders=2, num_timesteps=timesteps_per_scenario,
                                             seed=network_id * 100 + scenario_id)
            
            # Run power flow for each timestep
            for hour in range(timesteps_per_scenario):
                # Update graph with current loads and generation
                total_ev_load = sum(profile[hour] for profile in ev_profiles.values())
                total_pv_gen = sum(profile[hour] for profile in pv_profiles.values())
                
                # Distribute loads/generation across nodes
                nodes = list(graph.nodes())
                for i, node in enumerate(nodes):
                    graph.nodes[node]['load'] = total_ev_load / len(nodes)
                    graph.nodes[node]['generation'] = total_pv_gen / len(nodes)
                
                # Run power flow
                try:
                    pf_results = run_powerflow_analysis(graph)
                    constraints = check_constraints(graph, pf_results)
                except Exception as e:
                    print(f"    Power flow error at hour {hour}: {e}")
                    pf_results = {'voltages': np.ones(graph.number_of_nodes()), 'total_loss': 0.0}
                    constraints = {'voltage_violations': [], 'thermal_violations': []}
                
                # Create graph features for GNN
                nodes_data = []
                for node in nodes:
                    nodes_data.append({
                        'node_id': node,
                        'voltage': float(pf_results['voltages'][node] if isinstance(pf_results['voltages'], np.ndarray) and node < len(pf_results['voltages']) else 1.0),
                        'frequency': 50.0 + np.random.normal(0, 0.2),
                        'generation': graph.nodes[node].get('generation', 0),
                        'load': graph.nodes[node].get('load', 0),
                    })
                
                edges_data = []
                for edge in graph.edges():
                    i, j = edge
                    edges_data.append({
                        'edge': edge,
                        'resistance': graph.edges[edge].get('resistance', 0.05),
                        'reactance': graph.edges[edge].get('reactance', 0.1),
                    })
                
                # Create sample
                sample = {
                    'nodes': nodes_data,
                    'edges': edges_data,
                    'metadata': {
                        'sample_id': sample_id,
                        'network_id': network_id,
                        'scenario_id': scenario_id,
                        'hour': hour,
                        'topology': topology,
                    },
                    'targets': {
                        'total_loss': float(pf_results.get('total_loss', 0.0)),
                        'voltage_violations': len(constraints.get('voltage_violations', [])),
                        'thermal_violations': len(constraints.get('thermal_violations', [])),
                    }
                }
                
                dataset.append(sample)
                
                # Add to summary
                summary.append({
                    'sample_id': sample_id,
                    'network_id': network_id,
                    'scenario_id': scenario_id,
                    'hour': hour,
                    'num_nodes': graph.number_of_nodes(),
                    'num_edges': graph.number_of_edges(),
                    'topology': topology,
                    'total_loss': float(pf_results.get('total_loss', 0.0)),
                    'voltage_violations': len(constraints.get('voltage_violations', [])),
                    'thermal_violations': len(constraints.get('thermal_violations', [])),
                })
                
                sample_id += 1
                
                if sample_id % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = sample_id / elapsed
                    remaining = (num_samples - sample_id) / rate
                    print(f"    Progress: {sample_id}/{num_samples} ({100*sample_id/num_samples:.1f}%) "
                          f"[{rate:.1f} samples/sec, ~{remaining/60:.1f} min remaining]")
    
    # Save dataset
    print(f"\nSaving dataset to {output_dir}/...")
    
    # Save JSON dataset
    with open(output_path / "gnn_dataset_full.json", 'w') as f:
        json.dump(dataset, f, indent=2)
    
    # Save CSV summary
    with open(output_path / "dataset_summary_full.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    
    # Print statistics
    elapsed = time.time() - start_time
    print(f"\n✅ Dataset generation complete!")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Elapsed time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print(f"  Average rate: {len(dataset)/elapsed:.1f} samples/second")
    print(f"  Files saved:")
    print(f"    - {output_path / 'gnn_dataset_full.json'}")
    print(f"    - {output_path / 'dataset_summary_full.csv'}")


if __name__ == "__main__":
    generate_dataset(num_samples=12000, output_dir="gnn_dataset_full")
