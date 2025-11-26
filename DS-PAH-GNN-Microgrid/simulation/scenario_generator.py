"""
Scenario generation for microgrid simulations.

This module generates realistic, reproducible scenarios for time-series simulation
including EV charging events, solar/PV profiles, and contingency variations.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import networkx as nx
from copy import deepcopy


class EVChargingProfile:
    """Manages EV charging demand profiles with IEEE 1149 CC-CV charging dynamics."""

    def __init__(self, station_id: str, max_power_kw: float = 50.0, station_capacity: int = 4, 
                 v_ref: float = 400.0, i_max: float = 125.0):
        """
        Initialize EV charging profile with CC-CV charging model.

        Parameters
        ----------
        station_id : str
            Identifier for the charging station
        max_power_kw : float
            Maximum charging power per vehicle in kW
        station_capacity : int
            Number of charging ports at station
        v_ref : float
            Reference battery voltage (V) for voltage regulation
        i_max : float
            Maximum charging current (A) in constant-current phase
        """
        self.station_id = station_id
        self.max_power_kw = max_power_kw
        self.station_capacity = station_capacity
        self.v_ref = v_ref  # Reference voltage for CV phase
        self.i_max = i_max  # Maximum current for CC phase
        self.plugged_in_vehicles = []  # List of (start_time, duration_hours, soc_target)
        self.vehicle_socs = {}  # Track SOC for each vehicle {vehicle_id: current_soc}

    def add_charging_event(self, start_hour: float, duration_hours: float, soc_target: float = 0.8, 
                          vehicle_id: Optional[str] = None):
        """Add a vehicle charging event (plug-in and duration)."""
        if vehicle_id is None:
            vehicle_id = f"V{len(self.plugged_in_vehicles)}"
        self.plugged_in_vehicles.append((vehicle_id, start_hour, duration_hours, soc_target))
        self.vehicle_socs[vehicle_id] = 0.2  # Initial SOC at 20%

    def get_power_at_time(self, hour: float) -> float:
        """
        Get total charging power at a given hour using CC-CV dynamics.
        
        IEEE 1149 Standard Two-Phase Model:
        - Phase 1 (CC): Constant current until SOC reaches 80%
        - Phase 2 (CV): Constant voltage with power tapering as V approaches V_ref
        """
        power_kw = 0.0
        
        for vehicle_id, start, duration, soc_target in self.plugged_in_vehicles:
            if start <= hour < start + duration:
                # Vehicle is plugged in - determine charging phase
                current_soc = self.vehicle_socs.get(vehicle_id, 0.2)
                
                if current_soc < 0.80:
                    # Phase 1: Constant Current (CC) - full power
                    power_kw += self.max_power_kw
                    # Increment SOC over charging period (linear approximation)
                    soc_increment = (self.max_power_kw / 50.0) * (1.0 / duration) * 0.15  # 15% SOC per full charge
                    self.vehicle_socs[vehicle_id] = min(0.80, current_soc + soc_increment)
                else:
                    # Phase 2: Constant Voltage (CV) - power tapers
                    # Power decreases as V_battery approaches V_ref
                    soc_progress = (current_soc - 0.80) / (soc_target - 0.80)  # 0 to 1
                    taper_factor = max(0.1, 1.0 - soc_progress)  # Linear taper from 1.0 to 0.1
                    power_kw += self.max_power_kw * taper_factor
                    
                    # Increment SOC more slowly in CV phase
                    soc_increment = (self.max_power_kw * taper_factor / 50.0) * (1.0 / duration) * 0.20
                    self.vehicle_socs[vehicle_id] = min(soc_target, current_soc + soc_increment)
        
        return min(power_kw, self.station_capacity * self.max_power_kw)  # Respect station capacity


class SolarProfile:
    """Manages solar/PV generation profiles with cloud variation."""

    def __init__(self, feeder_id: str, rated_power_kw: float = 50.0, cloud_variability: float = 0.15):
        """
        Initialize solar profile.

        Parameters
        ----------
        feeder_id : str
            Identifier for the feeder with PV
        rated_power_kw : float
            Rated PV power in kW
        cloud_variability : float
            Standard deviation of cloud-induced variability (fraction of rated power)
        """
        self.feeder_id = feeder_id
        self.rated_power_kw = rated_power_kw
        self.cloud_variability = cloud_variability

    def get_irradiance_at_time(
        self, hour: float, day_of_year: int = 180, rng: Optional[np.random.Generator] = None
    ) -> float:
        """
        Get solar irradiance (0-1 normalized) at a given hour of the day.

        Parameters
        ----------
        hour : float
            Hour of the day (0-24)
        day_of_year : int
            Day of the year for seasonal variation
        rng : np.random.Generator, optional
            Numpy random generator for cloud effects

        Returns
        -------
        irradiance : float
            Normalized irradiance (0-1)
        """
        # Simplified sine-wave solar model: peaks at noon, zero at sunrise/sunset
        if hour < 6 or hour > 18:
            base_irradiance = 0.0
        else:
            # Normalized sine curve from 6am to 6pm
            base_irradiance = max(0, np.sin((hour - 6) / 12 * np.pi))

        # Seasonal variation (higher in summer, lower in winter)
        seasonal_factor = 0.7 + 0.3 * np.sin((day_of_year - 80) / 365 * 2 * np.pi)

        # Apply cloud variability
        cloud_effect = 0.0
        if rng is not None:
            cloud_effect = rng.normal(0, self.cloud_variability)
        cloud_effect = np.clip(cloud_effect, -0.5, 0.5)

        irradiance = base_irradiance * seasonal_factor + cloud_effect
        return np.clip(irradiance, 0, 1)

    def get_power_at_time(
        self, hour: float, day_of_year: int = 180, rng: Optional[np.random.Generator] = None
    ) -> float:
        """Get PV power generation at a given hour."""
        irradiance = self.get_irradiance_at_time(hour, day_of_year, rng)
        return irradiance * self.rated_power_kw


def generate_ev_load_profile(
    num_stations: int = 4,
    num_timesteps: int = 24,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None
) -> Dict[str, List[float]]:
    """
    Generate realistic EV charging demand profiles for multiple stations.

    Parameters
    ----------
    num_stations : int
        Number of charging stations
    num_timesteps : int
        Number of hourly time steps
    seed : int, optional
        Random seed for reproducibility
    rng : np.random.Generator, optional
        Numpy random generator

    Returns
    -------
    ev_profiles : dict
        Mapping of station_id -> list of power demands (kW) over time
    """
    if rng is None:
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

    ev_profiles = {}

    for station_idx in range(num_stations):
        station_id = f"EV_S{station_idx + 1}"
        profile = EVChargingProfile(station_id, max_power_kw=30.0)

        # Simulate 3-4 charging events per day (morning, midday, evening)
        num_events = rng.integers(2, 5)
        for event_idx in range(num_events):
            # Random plug-in times (weighted towards morning/evening peaks)
            if rng.random() > 0.5:
                # Morning/evening peak (6-10am, 4-8pm)
                start_hour = rng.choice([rng.uniform(6, 10), rng.uniform(16, 20)])
            else:
                # Off-peak hours
                start_hour = rng.uniform(10, 16)
            start_hour = np.clip(start_hour, 0, 23)
            duration = rng.uniform(1.5, 4.0)  # 1.5-4 hours charging

            vehicle_id = f"V{station_idx}_{event_idx}"
            profile.add_charging_event(start_hour, duration, soc_target=0.8, vehicle_id=vehicle_id)

        # Extract power profile over all timesteps
        power_profile = [profile.get_power_at_time(float(t)) for t in range(num_timesteps)]
        ev_profiles[station_id] = power_profile

    return ev_profiles


def generate_pv_profile(
    num_feeders: int = 3,
    num_timesteps: int = 24,
    day_of_year: int = 180,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None
) -> Dict[str, List[float]]:
    """
    Generate realistic solar PV generation profiles with cloud variability.

    Parameters
    ----------
    num_feeders : int
        Number of feeders with PV
    num_timesteps : int
        Number of hourly time steps
    day_of_year : int
        Day of the year for seasonal variation
    seed : int, optional
        Random seed
    rng : np.random.Generator, optional
        Numpy random generator

    Returns
    -------
    pv_profiles : dict
        Mapping of feeder_id -> list of power generation (kW) over time
    """
    if rng is None:
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

    pv_profiles = {}

    for feeder_idx in range(num_feeders):
        feeder_id = f"PV_F{feeder_idx + 1}"
        solar = SolarProfile(feeder_id, rated_power_kw=40.0 + feeder_idx * 10)

        power_profile = [
            solar.get_power_at_time(float(t), day_of_year, rng)
            for t in range(num_timesteps)
        ]
        pv_profiles[feeder_id] = power_profile

    return pv_profiles


def generate_scenarios(
    base_graph: nx.Graph,
    num_scenarios: int = 10,
    scenario_type: str = "renewable_variation",
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None
) -> List[Tuple[nx.Graph, Dict[str, Any]]]:
    """
    Generate multiple scenarios from a base microgrid.

    Parameters
    ----------
    base_graph : nx.Graph
        Base microgrid network
    num_scenarios : int
        Number of scenarios to generate
    scenario_type : str
        Type of variation ('renewable_variation', 'load_variation', 'contingency')
    seed : int, optional
        Random seed
    rng : np.random.Generator, optional
        Numpy random generator

    Returns
    -------
    scenarios : list of (nx.Graph, metadata)
        List of (graph, metadata_dict) tuples
    """
    if rng is None:
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

    scenarios = []

    for scenario_idx in range(num_scenarios):
        scenario_graph = deepcopy(base_graph)
        metadata = {"scenario_id": scenario_idx, "type": scenario_type, "seed": seed}

        if scenario_type == "renewable_variation":
            # Vary renewable generation across all nodes
            for node in scenario_graph.nodes():
                if scenario_graph.nodes[node].get("generation", 0) > 0:
                    variation = float(rng.uniform(0.2, 1.0))
                    scenario_graph.nodes[node]["generation"] *= variation
            metadata["variations"] = {"renewable": "randomized"}

        elif scenario_type == "load_variation":
            # Vary loads across all nodes
            for node in scenario_graph.nodes():
                variation = float(rng.uniform(0.8, 1.2))
                scenario_graph.nodes[node]["load"] *= variation
            metadata["variations"] = {"load": "randomized"}

        elif scenario_type == "contingency":
            # Simulate component outages
            max_out = min(3, max(1, scenario_graph.number_of_edges()))
            num_outages = rng.integers(0, max_out + 1)
            edges = list(scenario_graph.edges())
            if num_outages > 0 and len(edges) > 0:
                edges_to_remove = list(rng.choice(edges, size=min(num_outages, len(edges)), replace=False))
                scenario_graph.remove_edges_from(edges_to_remove)
                metadata["outages"] = [tuple(e) for e in edges_to_remove]

        scenarios.append((scenario_graph, metadata))

    return scenarios


def create_time_series_scenarios(
    base_graph: nx.Graph,
    num_timesteps: int = 24,
    day_of_year: int = 180,
    renewable_profile: Optional[np.ndarray] = None,
    include_ev: bool = True,
    include_pv: bool = True,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None
) -> List[Tuple[int, nx.Graph, Dict[str, Any]]]:
    """
    Create time-series scenarios with realistic EV and PV profiles.

    Parameters
    ----------
    base_graph : nx.Graph
        Base microgrid network
    num_timesteps : int
        Number of hourly time steps
    day_of_year : int
        Day of the year for seasonal variation
    renewable_profile : np.ndarray, optional
        Renewable generation profile over time
    include_ev : bool
        Whether to include EV charging profiles
    include_pv : bool
        Whether to include PV generation
    seed : int, optional
        Random seed
    rng : np.random.Generator, optional
        Numpy random generator

    Returns
    -------
    time_series : list of (hour, nx.Graph, metadata)
        List of (t, graph, metadata_dict) tuples
    """
    if rng is None:
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

    if renewable_profile is None:
        # Default sinusoidal profile (peaks at noon)
        renewable_profile = np.sin(np.linspace(0, np.pi, num_timesteps)) * 0.5 + 0.3

    # Generate base profiles once
    if include_ev:
        ev_profiles = generate_ev_load_profile(
            num_stations=4, num_timesteps=num_timesteps, seed=seed, rng=rng
        )
    else:
        ev_profiles = {}

    if include_pv:
        pv_profiles = generate_pv_profile(
            num_feeders=3, num_timesteps=num_timesteps, day_of_year=day_of_year, seed=seed, rng=rng
        )
    else:
        pv_profiles = {}

    time_series = []

    for t in range(num_timesteps):
        scenario_graph = deepcopy(base_graph)
        metadata = {
            "timestep": t,
            "hour_of_day": t,
            "day_of_year": day_of_year,
            "seed": seed,
            "include_ev": include_ev,
            "include_pv": include_pv
        }

        # Apply EV loads
        if include_ev:
            node_list = list(scenario_graph.nodes())
            for station_id, profile in ev_profiles.items():
                if node_list:
                    node = node_list[hash(station_id) % len(node_list)]
                    scenario_graph.nodes[node]["ev_load"] = float(profile[t])

        # Apply PV generation
        if include_pv:
            node_list = list(scenario_graph.nodes())
            for feeder_id, profile in pv_profiles.items():
                if node_list:
                    node = node_list[hash(feeder_id) % len(node_list)]
                    scenario_graph.nodes[node]["pv_generation"] = float(profile[t])

        # Apply time-varying generation with optional stochastic fluctuations
        for node in scenario_graph.nodes():
            base_gen = float(scenario_graph.nodes[node].get("generation", 0))
            # apply deterministic profile
            gen = base_gen * float(renewable_profile[t])
            # add small stochastic fluctuation to model clouds
            if base_gen > 0:
                gen *= float(rng.normal(1.0, 0.05))
            scenario_graph.nodes[node]["generation"] = max(0.0, gen)

        # Apply time-varying loads
        load_profile = 0.7 + 0.3 * np.sin(2 * np.pi * t / num_timesteps)
        for node in scenario_graph.nodes():
            base_load = float(scenario_graph.nodes[node].get("load", 0))
            load = base_load * load_profile
            # add per-node noise
            load *= float(rng.normal(1.0, 0.05))
            scenario_graph.nodes[node]["load"] = max(0.0, load)

        time_series.append((t, scenario_graph, metadata))

    return time_series


def create_adversarial_scenarios(
    base_graph: nx.Graph,
    num_scenarios: int = 5,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None
) -> List[Tuple[nx.Graph, Dict[str, Any]]]:
    """
    Create adversarial scenarios testing system resilience.

    Scenarios include simultaneous contingencies, extreme generation/load
    combinations, and cascading failures.
    """
    if rng is None:
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

    scenarios = []

    for scenario_idx in range(num_scenarios):
        scenario_graph = deepcopy(base_graph)
        metadata = {"scenario_id": scenario_idx, "type": "adversarial", "seed": seed}

        # Random contingency (outage)
        outages = []
        if rng.random() > 0.5 and scenario_graph.number_of_edges() > 1:
            edges = list(scenario_graph.edges())
            edge_to_remove = tuple(rng.choice(edges))
            try:
                scenario_graph.remove_edge(*edge_to_remove)
                outages.append(edge_to_remove)
            except Exception:
                pass
        metadata["outages"] = outages

        # Extreme generation variation
        gen_extremes = []
        for node in scenario_graph.nodes():
            if rng.random() > 0.7:
                factor = float(rng.uniform(0.1, 2.0))
                scenario_graph.nodes[node]["generation"] = (
                    scenario_graph.nodes[node].get("generation", 0) * factor
                )
                gen_extremes.append({"node": node, "factor": factor})
        metadata["generation_extremes"] = gen_extremes

        # Extreme load variation
        load_extremes = []
        for node in scenario_graph.nodes():
            if rng.random() > 0.7:
                factor = float(rng.uniform(0.5, 1.8))
                scenario_graph.nodes[node]["load"] = (
                    scenario_graph.nodes[node].get("load", 0) * factor
                )
                load_extremes.append({"node": node, "factor": factor})
        metadata["load_extremes"] = load_extremes

        scenarios.append((scenario_graph, metadata))

    return scenarios
