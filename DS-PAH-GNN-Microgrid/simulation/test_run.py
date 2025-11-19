"""Small smoke test to validate scenario generation and powerflow fallback."""

from simulation.create_network import create_microgrid
from simulation.scenario_generator import create_time_series_scenarios, generate_scenarios
from simulation.run_powerflow import run_powerflow_analysis, check_constraints


def main():
    g, meta = create_microgrid(num_nodes=6, topology='ring', seed=42)
    print('Created graph:', meta)

    # Generate a few scenarios
    scenarios = generate_scenarios(g, num_scenarios=3, scenario_type='renewable_variation', seed=42)
    print(f'Generated {len(scenarios)} renewable variation scenarios')

    ts = create_time_series_scenarios(g, num_timesteps=6, rng=None)
    print(f'Generated {len(ts)} timesteps')

    # Run powerflow on first scenario using lightweight fallback
    result = run_powerflow_analysis(scenarios[0])
    print('Powerflow converged:', result['converged'])
    print('Total loss (approx):', result['total_loss'])

    violations = check_constraints(scenarios[0], result)
    print('Violations:', {k: len(v) for k, v in violations.items()})


if __name__ == '__main__':
    main()
