"""
Load profile generation for power distribution systems.

Generates realistic load patterns for:
- Residential loads
- Commercial loads
- Electric vehicle (EV) charging
- Distributed Energy Resources (DER)
"""

import numpy as np
import torch
from typing import Tuple, Optional, Dict
import math


def generate_residential_load(
    num_nodes: int,
    timesteps: int = 24,
    base_load: float = 0.5,
    peak_factor: float = 1.5,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Generate residential load profile with daily pattern.
    
    Args:
        num_nodes: Number of nodes
        timesteps: Number of time steps (default: 24 hours)
        base_load: Base load in per-unit
        peak_factor: Peak to base ratio
        seed: Random seed
        
    Returns:
        Load profile: [num_nodes, timesteps]
    """
    if seed is not None:
        np.random.seed(seed)
    
    loads = np.zeros((num_nodes, timesteps))
    
    for i in range(num_nodes):
        # Daily pattern: low at night, peak in evening
        time_pattern = np.array([
            math.sin(2 * math.pi * (t - 6) / 24) for t in range(timesteps)
        ])
        time_pattern = (time_pattern + 1) / 2  # Normalize to [0, 1]
        
        # Scale to base and peak
        profile = base_load + (peak_factor - base_load) * time_pattern
        
        # Add random variation
        noise = np.random.normal(0, 0.1, timesteps)
        profile = profile + noise
        profile = np.clip(profile, 0.1, 2.0)  # Clip to reasonable range
        
        loads[i, :] = profile
    
    return torch.tensor(loads, dtype=torch.float32)


def generate_commercial_load(
    num_nodes: int,
    timesteps: int = 24,
    base_load: float = 0.3,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Generate commercial load profile (higher during business hours).
    
    Args:
        num_nodes: Number of nodes
        timesteps: Number of time steps
        base_load: Base load in per-unit
        seed: Random seed
        
    Returns:
        Load profile: [num_nodes, timesteps]
    """
    if seed is not None:
        np.random.seed(seed)
    
    loads = np.zeros((num_nodes, timesteps))
    
    for i in range(num_nodes):
        # Business hours pattern (8 AM - 6 PM)
        profile = np.ones(timesteps) * base_load
        for t in range(timesteps):
            if 8 <= t < 18:  # Business hours
                profile[t] = base_load * 3.0
        
        # Add random variation
        noise = np.random.normal(0, 0.05, timesteps)
        profile = profile + noise
        profile = np.clip(profile, 0.1, 2.0)
        
        loads[i, :] = profile
    
    return torch.tensor(loads, dtype=torch.float32)


def generate_ev_charging_profile(
    num_evs: int,
    timesteps: int = 24,
    charging_rate: float = 0.8,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Generate EV charging profile (typically evening charging).
    
    Args:
        num_evs: Number of EVs
        timesteps: Number of time steps
        charging_rate: Charging power in per-unit
        seed: Random seed
        
    Returns:
        Charging profile: [num_evs, timesteps]
    """
    if seed is not None:
        np.random.seed(seed)
    
    profiles = np.zeros((num_evs, timesteps))
    
    for i in range(num_evs):
        # Random arrival time (typically evening)
        arrival_time = int(np.random.normal(18, 2))  # Around 6 PM
        arrival_time = np.clip(arrival_time, 0, timesteps - 4)
        
        # Charging duration (3-4 hours)
        duration = int(np.random.uniform(3, 5))
        
        # Set charging period
        for t in range(arrival_time, min(arrival_time + duration, timesteps)):
            profiles[i, t] = charging_rate * np.random.uniform(0.8, 1.0)
    
    return torch.tensor(profiles, dtype=torch.float32)


def generate_solar_generation(
    num_panels: int,
    timesteps: int = 24,
    peak_generation: float = 1.0,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Generate solar PV generation profile (daytime generation).
    
    Args:
        num_panels: Number of solar installations
        timesteps: Number of time steps
        peak_generation: Peak generation in per-unit (negative = generation)
        seed: Random seed
        
    Returns:
        Generation profile: [num_panels, timesteps] (negative values)
    """
    if seed is not None:
        np.random.seed(seed)
    
    profiles = np.zeros((num_panels, timesteps))
    
    for i in range(num_panels):
        # Solar irradiance pattern (sunrise to sunset)
        for t in range(timesteps):
            if 6 <= t <= 18:  # Daylight hours
                # Bell curve centered at noon
                solar_angle = math.sin(math.pi * (t - 6) / 12)
                generation = -peak_generation * solar_angle  # Negative = generation
                
                # Add cloud variability
                cloud_factor = np.random.uniform(0.7, 1.0)
                generation *= cloud_factor
                
                profiles[i, t] = generation
    
    return torch.tensor(profiles, dtype=torch.float32)


def generate_mixed_load_profile(
    num_nodes: int,
    timesteps: int = 24,
    residential_ratio: float = 0.6,
    commercial_ratio: float = 0.3,
    ev_ratio: float = 0.1,
    solar_ratio: float = 0.2,
    seed: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """
    Generate mixed load profile with multiple sources.
    
    Args:
        num_nodes: Number of nodes
        timesteps: Number of time steps
        residential_ratio: Fraction of residential loads
        commercial_ratio: Fraction of commercial loads
        ev_ratio: Fraction with EV charging
        solar_ratio: Fraction with solar generation
        seed: Random seed
        
    Returns:
        Dictionary with load components and total net load
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate number of each type
    n_residential = int(num_nodes * residential_ratio)
    n_commercial = int(num_nodes * commercial_ratio)
    n_ev = int(num_nodes * ev_ratio)
    n_solar = int(num_nodes * solar_ratio)
    
    # Generate components
    res_load = generate_residential_load(n_residential, timesteps, seed=seed)
    com_load = generate_commercial_load(n_commercial, timesteps, seed=seed+1 if seed else None)
    ev_load = generate_ev_charging_profile(n_ev, timesteps, seed=seed+2 if seed else None)
    solar_gen = generate_solar_generation(n_solar, timesteps, seed=seed+3 if seed else None)
    
    # Combine into total load profile for all nodes
    total_load = torch.zeros(num_nodes, timesteps)
    
    idx = 0
    # Residential
    for i in range(min(n_residential, num_nodes - idx)):
        total_load[idx + i, :] = res_load[i, :]
    idx += n_residential
    
    # Commercial
    for i in range(min(n_commercial, num_nodes - idx)):
        if idx + i < num_nodes:
            total_load[idx + i, :] = com_load[i, :]
    idx += n_commercial
    
    # Add EV charging to random nodes
    ev_nodes = np.random.choice(num_nodes, size=min(n_ev, num_nodes), replace=False)
    for i, node_idx in enumerate(ev_nodes):
        if i < ev_load.shape[0]:
            total_load[node_idx, :] += ev_load[i, :]
    
    # Add solar generation to random nodes
    solar_nodes = np.random.choice(num_nodes, size=min(n_solar, num_nodes), replace=False)
    for i, node_idx in enumerate(solar_nodes):
        if i < solar_gen.shape[0]:
            total_load[node_idx, :] += solar_gen[i, :]
    
    return {
        'residential': res_load,
        'commercial': com_load,
        'ev_charging': ev_load,
        'solar_generation': solar_gen,
        'total_net_load': total_load
    }


if __name__ == "__main__":
    # Test load generators
    print("Testing load generators...")
    
    num_nodes = 10
    timesteps = 24
    
    # Test residential
    res_load = generate_residential_load(num_nodes, timesteps, seed=42)
    print(f"Residential load shape: {res_load.shape}")
    print(f"Residential load range: [{res_load.min():.2f}, {res_load.max():.2f}]")
    
    # Test commercial
    com_load = generate_commercial_load(num_nodes, timesteps, seed=42)
    print(f"Commercial load shape: {com_load.shape}")
    
    # Test EV
    ev_load = generate_ev_charging_profile(5, timesteps, seed=42)
    print(f"EV charging shape: {ev_load.shape}")
    
    # Test solar
    solar_gen = generate_solar_generation(5, timesteps, seed=42)
    print(f"Solar generation shape: {solar_gen.shape}")
    print(f"Solar generation range: [{solar_gen.min():.2f}, {solar_gen.max():.2f}]")
    
    # Test mixed
    mixed = generate_mixed_load_profile(num_nodes, timesteps, seed=42)
    print(f"Total net load shape: {mixed['total_net_load'].shape}")
    print(f"Total net load range: [{mixed['total_net_load'].min():.2f}, {mixed['total_net_load'].max():.2f}]")
    
    print("✓ Load generator tests passed!")

