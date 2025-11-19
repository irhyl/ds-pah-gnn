#!/usr/bin/env python3
"""
Smoke test for DS-PAH-GNN Microgrid simulation pipeline.

Validates: network creation, scenarios, power flow, constraints, time-series, export.
"""

import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import networkx as nx

from simulation.create_network import create_microgrid
from simulation.scenario_generator import (
    generate_scenarios,
    create_time_series_scenarios,
    create_adversarial_scenarios,
    generate_ev_load_profile,
    generate_pv_profile,
)
from simulation.run_powerflow import run_powerflow_analysis, check_constraints, export_results_json


def test_1_network_creation():
    """Test network creation with different topologies."""
    print("\n" + "="*70)
    print("TEST 1: Network Creation")
    print("="*70)
    
    for topo in ["mesh", "ring", "tree"]:
        graph, meta = create_microgrid(num_nodes=8, topology=topo, seed=42)
        print(f"\n{topo.upper()}: {meta['num_nodes']} nodes, {meta['num_edges']} edges")
        assert graph.number_of_nodes() == meta['num_nodes']
        assert graph.number_of_edges() == meta['num_edges']
        for node in graph.nodes():
            assert 'voltage' in graph.nodes[node]
            assert 'generation' in graph.nodes[node]
    print("\n✓ PASSED")
    return graph


def test_2_ev_pv_profiles():
    """Test EV and PV profile generation."""
    print("\n" + "="*70)
    print("TEST 2: EV and PV Profile Generation")
    print("="*70)
    
    rng = np.random.default_rng(42)
    ev_profiles = generate_ev_load_profile(num_stations=4, num_timesteps=24, rng=rng)
    print(f"\nEV profiles: {len(ev_profiles)} stations")
    for station_id, profile in list(ev_profiles.items())[:2]:
        print(f"  {station_id}: max={max(profile):.1f} kW")
        assert len(profile) == 24
        assert all(p >= 0 for p in profile)
    
    pv_profiles = generate_pv_profile(num_feeders=3, num_timesteps=24, day_of_year=180, rng=rng)
    print(f"\nPV profiles: {len(pv_profiles)} feeders")
    for feeder_id, profile in list(pv_profiles.items())[:2]:
        print(f"  {feeder_id}: max={max(profile):.1f} kW")
        assert len(profile) == 24
        assert all(p >= 0 for p in profile)
    
    # Test reproducibility
    ev_2 = generate_ev_load_profile(num_stations=4, num_timesteps=24, seed=42)
    assert all(np.allclose(ev_profiles[k], ev_2[k]) for k in ev_profiles)
    
    print("\n✓ PASSED")


def test_3_scenario_generation():
    """Test scenario generation."""
    print("\n" + "="*70)
    print("TEST 3: Scenario Generation")
    print("="*70)
    
    base_graph, _ = create_microgrid(num_nodes=10, topology="mesh", seed=42)
    
    for scenario_type in ["renewable_variation", "load_variation", "contingency"]:
        scenarios = generate_scenarios(base_graph, num_scenarios=3, scenario_type=scenario_type, seed=42)
        print(f"\n{scenario_type}: {len(scenarios)} scenarios")
        for idx, (graph, meta) in enumerate(scenarios[:1]):
            print(f"  Scenario {idx}: nodes={graph.number_of_nodes()}")
        assert len(scenarios) == 3
    
    print("\n✓ PASSED")


def test_4_powerflow_analysis():
    """Test power flow analysis."""
    print("\n" + "="*70)
    print("TEST 4: Power Flow Analysis")
    print("="*70)
    
    graph, _ = create_microgrid(num_nodes=8, topology="mesh", seed=42)
    
    results = run_powerflow_analysis(graph, max_iterations=100, tolerance=1e-6)
    
    print(f"\nPower Flow Results:")
    print(f"  Converged: {results['converged']}")
    print(f"  Iterations: {results['iterations']}")
    print(f"  Voltage range: {np.min(results['voltages']):.4f} - {np.max(results['voltages']):.4f} pu")
    print(f"  Total loss: {results['total_loss']:.6f} MW")
    print(f"  Branch flows: {len(results['branch_flows'])}")
    
    assert results['converged']
    assert len(results['voltages']) == graph.number_of_nodes()
    assert results['total_loss'] >= 0
    
    print("\n✓ PASSED")
    return graph, results


def test_5_constraints(graph, results):
    """Test constraint checking."""
    print("\n" + "="*70)
    print("TEST 5: Constraint Checking")
    print("="*70)
    
    defaults = {'vmin': 0.95, 'vmax': 1.05, 'default_line_capacity': 100.0}
    violations = check_constraints(graph, results, defaults)
    
    print(f"\nViolations:")
    print(f"  Voltage: {len(violations['voltage_violations'])}")
    print(f"  Thermal: {len(violations['thermal_violations'])}")
    print(f"  Frequency: {len(violations['frequency_violations'])}")
    
    assert isinstance(violations, dict)
    assert 'voltage_violations' in violations
    
    print("\n✓ PASSED")


def test_6_time_series():
    """Test time-series scenario generation."""
    print("\n" + "="*70)
    print("TEST 6: Time-Series Generation")
    print("="*70)
    
    base_graph, _ = create_microgrid(num_nodes=8, topology="mesh", seed=42)
    time_series = create_time_series_scenarios(
        base_graph, num_timesteps=24, include_ev=True, include_pv=True, seed=42
    )
    print(f"\nTime-series: {len(time_series)} timesteps")
    for t, graph, meta in time_series[::8]:  # Every 8 hours
        print(f"  Hour {meta['hour_of_day']}: {graph.number_of_nodes()} nodes")
    
    assert len(time_series) == 24
    print("\n✓ PASSED")


def test_7_adversarial():
    """Test adversarial scenario generation."""
    print("\n" + "="*70)
    print("TEST 7: Adversarial Scenarios")
    print("="*70)
    
    base_graph, _ = create_microgrid(num_nodes=10, topology="mesh", seed=42)
    adversarial = create_adversarial_scenarios(base_graph, num_scenarios=5, seed=42)
    
    print(f"\nAdversarial: {len(adversarial)} scenarios")
    for idx, (graph, meta) in enumerate(adversarial[:2]):
        print(f"  Scenario {idx}: outages={len(meta.get('outages', []))}")
    
    assert len(adversarial) == 5
    print("\n✓ PASSED")


def test_8_results_export():
    """Test results export."""
    print("\n" + "="*70)
    print("TEST 8: Results Export")
    print("="*70)
    
    base_graph, _ = create_microgrid(num_nodes=8, topology="mesh", seed=42)
    results = run_powerflow_analysis(base_graph, max_iterations=100, tolerance=1e-6)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "results.json"
        export_results_json(results, str(export_path))
        
        assert export_path.exists(), "Export file not created"
        
        with open(export_path, 'r') as f:
            exported = json.load(f)
        
        print(f"\nExported to: {export_path.name}")
        print(f"  Keys: {list(exported.keys())[:5]}")
        print(f"  Converged: {exported['converged']}")
        print(f"  Total loss: {exported['total_loss']:.6f} MW")
    
    print("\n✓ PASSED")


def main():
    """Run all smoke tests."""
    print("\n" + "="*70)
    print("DS-PAH-GNN MICROGRID SIMULATION - SMOKE TEST SUITE")
    print("="*70)
    
    try:
        # Test 1: Network creation
        graph = test_1_network_creation()
        
        # Test 2: EV/PV profiles
        test_2_ev_pv_profiles()
        
        # Test 3: Scenario generation
        test_3_scenario_generation()
        
        # Test 4 & 5: Power flow and constraints
        graph, results = test_4_powerflow_analysis()
        test_5_constraints(graph, results)
        
        # Test 6: Time-series
        test_6_time_series()
        
        # Test 7: Adversarial
        test_7_adversarial()
        
        # Test 8: Export
        test_8_results_export()
        
        print("\n" + "="*70)
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("="*70 + "\n")
        return 0
        
    except Exception as e:
        print(f"\n✗ Test FAILED")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

