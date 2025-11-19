"""
Power flow analysis helper utilities.

Provides two execution paths:
 - pandapower-backed power flow (preferred when pandapower is installed)
 - lightweight NetworkX-based approximate solver (fallback for quick tests)

The main entrypoint `run_powerflow_analysis` accepts either a pandapower `net`
or a `networkx.Graph` with node/edge attributes and returns a consistent
results dictionary.
"""

from typing import Dict, Any
import logging
import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)


def _run_simple_powerflow(graph: nx.Graph, max_iterations: int = 100, tolerance: float = 1e-6) -> Dict[str, Any]:
    """Lightweight approximate powerflow for NetworkX graphs.

    This is a simple resistive/admittance based iterative solver intended for
    quick checks and unit tests. It is NOT a replacement for a full pandapower
    solution but preserves the original project's lightweight behaviour.
    """
    num_nodes = graph.number_of_nodes()
    node_list = list(graph.nodes())
    node_idx = {node: i for i, node in enumerate(node_list)}

    # Initialize voltage magnitudes and angles
    voltages = np.ones(num_nodes, dtype=float) + np.random.normal(0, 0.01, num_nodes)
    angles = np.random.normal(0, 0.01, num_nodes)

    # Build admittance matrix (simplified)
    Y = np.zeros((num_nodes, num_nodes), dtype=complex)
    for edge in graph.edges():
        i, j = node_idx[edge[0]], node_idx[edge[1]]
        r = float(graph.edges[edge].get('resistance', 0.05))
        x = float(graph.edges[edge].get('reactance', 0.1))
        z = r + 1j * x
        y = 1 / z if z != 0 else 0
        Y[i, j] = -y
        Y[j, i] = -y
        Y[i, i] += y
        Y[j, j] += y

    converged = False
    mismatch = np.zeros(num_nodes, dtype=float)
    for iteration in range(1, max_iterations + 1):
        power_injections = np.zeros(num_nodes, dtype=complex)
        for i, node in enumerate(node_list):
            gen = float(graph.nodes[node].get('generation', 0))
            load = float(graph.nodes[node].get('load', 0))
            # normalize to MW -> per-unit-ish scale used in this lightweight solver
            power_injections[i] = (gen - load) / 100.0

        V = voltages * np.exp(1j * angles)
        S_calc = V * np.conj(Y @ V)
        mismatch = np.abs(power_injections - S_calc)

        if np.nanmax(mismatch) < tolerance:
            converged = True
            break

        # simple heuristic updates
        angles += 0.01 * np.imag(mismatch)
        voltages *= (1 + 0.001 * np.real(mismatch))
        voltages = np.clip(voltages, 0.5, 1.5)

    # Branch flows and losses
    branch_flows = []
    total_loss = 0.0
    V = voltages * np.exp(1j * angles)
    for edge in graph.edges():
        i, j = node_idx[edge[0]], node_idx[edge[1]]
        r = float(graph.edges[edge].get('resistance', 0.05))
        x = float(graph.edges[edge].get('reactance', 0.1))
        z = r + 1j * x
        I = (V[i] - V[j]) / z if z != 0 else 0
        S = V[i] * np.conj(I)
        loss = float((abs(I) ** 2) * r)

        branch_flows.append({
            'edge': edge,
            'power': float(abs(S)),
            'current': float(abs(I)),
            'loss': loss
        })
        total_loss += loss

    results = {
        'converged': converged,
        'iterations': iteration,
        'voltages': voltages,
        'angles': angles,
        'branch_flows': branch_flows,
        'total_loss': float(total_loss),
        'voltage_mismatch': mismatch,
        'pandapower_net': None
    }
    return results


def _run_pandapower(net: Any, max_iterations: int = 100, tolerance: float = 1e-6) -> Dict[str, Any]:
    """Run pandapower power flow and extract a normalized results dict.

    This function is defensive about available result fields and works with
    both AC and DC-style runs where pandapower provides the appropriate
    `res_bus` and `res_line` tables.
    """
    try:
        import pandapower as pp  # type: ignore
    except Exception as e:
        raise RuntimeError("pandapower is required for pandapower-backed runs") from e

    # Run power flow. Prefer a DC-style init when available; fall back to default.
    try:
        pp.runpp(net, init='dc', calculate_voltage_angles=False)
    except Exception:
        pp.runpp(net)

    # Extract bus voltages (prefer pu if present)
    voltages = None
    if hasattr(net, 'res_bus') and 'vm_pu' in net.res_bus.columns:
        voltages = net.res_bus['vm_pu'].to_numpy()
    elif hasattr(net, 'res_bus') and 'vm_kv' in net.res_bus.columns and 'vn_kv' in net.bus.columns:
        base_kv = net.bus['vn_kv'].to_numpy()
        vm_kv = net.res_bus['vm_kv'].to_numpy()
        if base_kv.shape == vm_kv.shape:
            voltages = (vm_kv / base_kv)
        else:
            voltages = vm_kv
    else:
        voltages = np.zeros(len(net.bus.index), dtype=float)

    # Lines: currents, loading, losses
    branch_flows = []
    total_loss = 0.0
    if hasattr(net, 'res_line') and not net.res_line.empty:
        rl = net.res_line
        for idx, row in rl.iterrows():
            # Map to bus indices where possible
            i = int(net.line.at[idx, 'from_bus']) if 'from_bus' in net.line.columns else None
            j = int(net.line.at[idx, 'to_bus']) if 'to_bus' in net.line.columns else None
            edge = (i, j) if (i is not None and j is not None) else int(idx)

            current = None
            loading = None
            loss = 0.0
            power = None

            if 'i_ka' in rl.columns:
                current = float(row['i_ka'])
            if 'loading_percent' in rl.columns:
                loading = float(row['loading_percent'])
            if 'pl_mw' in rl.columns:
                power = float(row['pl_mw'])
                loss = abs(power)
            elif 'p_loss_kw' in rl.columns:
                power = float(row['p_loss_kw']) / 1000.0
                loss = abs(power)

            branch_flows.append({
                'edge': edge,
                'power': power,
                'current': current,
                'loss': loss,
                'loading_percent': loading
            })
            total_loss += loss

    results = {
        'converged': True,
        'iterations': None,
        'voltages': np.asarray(voltages, dtype=float),
        'angles': None,
        'branch_flows': branch_flows,
        'total_loss': float(total_loss),
        'voltage_mismatch': None,
        'pandapower_net': net
    }
    return results


def run_powerflow_analysis(net_or_graph: Any, max_iterations: int = 100, tolerance: float = 1e-6) -> Dict[str, Any]:
    """Run power flow analysis on either a pandapower network or a NetworkX graph.

    Parameters
    ----------
    net_or_graph : pandapower net or networkx.Graph
        If a pandapower network is provided, the pandapower solver will be used.
        Otherwise, the function expects a NetworkX graph with node/edge
        attributes similar to the original implementation.
    """
    # Detect pandapower availability and net-like objects
    try:
        import pandapower as pp  # type: ignore
        has_pandapower = True
    except Exception:
        has_pandapower = False

    # Heuristic: pandapower net has attribute 'bus' (pandas.DataFrame)
    if has_pandapower and hasattr(net_or_graph, 'bus'):
        try:
            return _run_pandapower(net_or_graph, max_iterations=max_iterations, tolerance=tolerance)
        except Exception as e:
            logger.warning("pandapower run failed (%s). Falling back to lightweight solver.", e)

    if isinstance(net_or_graph, nx.Graph):
        return _run_simple_powerflow(net_or_graph, max_iterations=max_iterations, tolerance=tolerance)

    # Last resort: try to treat it as a NetworkX-like object
    try:
        return _run_simple_powerflow(net_or_graph, max_iterations=max_iterations, tolerance=tolerance)
    except Exception as e:
        raise RuntimeError("Unsupported network object for powerflow analysis") from e


def check_constraints(graph_or_net: Any, powerflow_results: Dict[str, Any], defaults: Dict[str, float] = None) -> Dict[str, Any]:
    """Check thermal and voltage constraints for results.

    Returns a dictionary with lists of violations.
    """
    if defaults is None:
        defaults = {'vmin': 0.95, 'vmax': 1.05, 'default_line_capacity': 100.0}

    violations = {
        'thermal_violations': [],
        'voltage_violations': [],
        'frequency_violations': []
    }

    # Voltage checks
    voltages = powerflow_results.get('voltages')
    if voltages is not None:
        vmin = defaults.get('vmin', 0.95)
        vmax = defaults.get('vmax', 1.05)
        for idx, V in enumerate(np.asarray(voltages, dtype=float)):
            if np.isnan(V):
                continue
            if V < vmin or V > vmax:
                violations['voltage_violations'].append({
                    'index': int(idx),
                    'voltage': float(V),
                    'limits': (vmin, vmax)
                })

    # Thermal checks from branch flows
    for flow in powerflow_results.get('branch_flows', []):
        edge = flow.get('edge')
        power = flow.get('power') if flow.get('power') is not None else 0.0
        capacity = defaults.get('default_line_capacity', 100.0)

        # If the graph/net exposes capacity information, attempt to use it
        try:
            if hasattr(graph_or_net, 'line') and hasattr(graph_or_net, 'res_line'):
                # pandapower case: try to read line max_loading or a custom field
                pass
            elif isinstance(graph_or_net, nx.Graph):
                # networkx graph: try to look up capacity from edge attributes
                if isinstance(edge, tuple) and graph_or_net.has_edge(edge[0], edge[1]):
                    cap = graph_or_net.edges[edge].get('capacity')
                    if cap is not None:
                        capacity = float(cap)
        except Exception:
            pass

        try:
            if power is not None and float(power) > capacity:
                violations['thermal_violations'].append({
                    'edge': edge,
                    'power': float(power),
                    'limit': float(capacity)
                })
        except Exception:
            continue

    return violations


def export_results_json(results: Dict[str, Any], path: str) -> None:
    """Export results dict to JSON-friendly file.

    This function converts numpy arrays to lists and writes a compact JSON file.
    """
    import json

    def _clean(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    with open(path, 'w', encoding='utf8') as f:
        json.dump(_clean(results), f, indent=2)

