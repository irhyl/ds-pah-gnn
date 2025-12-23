# DC Microgrid Design Specification

## System Overview
The simulation environment models a **Hierarchical Dual-Bus DC Microgrid** designed for EV fast-charging applications. The system is modeled as a directed graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ where nodes represent electrical buses/components and edges represent transmission lines.

## Topology Structure
The network follows a 3-tier hierarchy:

1.  **Level 0 (Root):** Point of Common Coupling (PCC). Acts as the slack bus (Voltage Source) with fixed voltage $V_{ref} = 1.0$ p.u.
2.  **Level 1 (Stations):** Distribution substations connected to the root via **Feeder Lines**.
3.  **Level 2 (Devices):** Leaf nodes connected to stations via **Branch Lines**.
    *   **EV Chargers:** Stochastic loads following CC-CV charging curves.
    *   **PV Inverters:** Stochastic generation sources.
    *   **Battery Storage (BESS):** Bidirectional energy buffers.

Additionally, **Tie-Lines** connect adjacent stations in a ring configuration, allowing for topology reconfiguration (load transfer) during congestion.

## Physics Engine
The system state is resolved using a **Linearized DC Power Flow** approximation ($P \approx V(V_i - V_j)/R$), which reduces the non-linear power flow equations to a linear system of the form:

$$ \mathbf{Y} \mathbf{V} = \mathbf{I} $$

Where:
- $\mathbf{Y}$: Nodal Admittance Matrix (constructed from $1/R_{ij}$).
- $\mathbf{V}$: Vector of nodal voltages (unknowns).
- $\mathbf{I}$: Vector of current injections ($I \approx P/V_{nom}$).

### Component Modeling

| Component | Type | Model Dynamics |
| :--- | :--- | :--- |
| **Grid (PCC)** | Slack Bus | Infinite source, maintains $V=1.0$ p.u. |
| **EV Charger** | Load ($P < 0$) | Stochastic arrival + CC-CV profile ($P_{max} \to P_{decay}$). |
| **PV Array** | Gen ($P > 0$) | Diurnal sine wave + Gaussian cloud noise. |
| **Storage** | Buffer | State-of-Charge (SOC) tracking: $SOC_{t+1} = SOC_t + \eta P_t \Delta t$. |

## Configuration Parameters

Default electrical parameters used in `microgrid.py`:

| Parameter | Value (p.u.) | Description |
| :--- | :--- | :--- |
| `v_source` | 1.00 | Nominal grid voltage. |
| `r_feeder` | 0.01 | Resistance of main feeder lines. |
| `r_branch` | 0.05 | Resistance of local branch lines. |
| `r_tie` | 0.03 | Resistance of reconfiguration tie-lines. |
| `load_scale`| 0.50 | Base scaling factor for loads. |

## Optimization Objective
The goal is to find the optimal set of switch states $S_{ij} \in \{0, 1\}$ for feeders and tie-lines that minimizes total system loss:

$$ \min \sum_{(i,j) \in \mathcal{E}} S_{ij} \cdot \frac{(V_i - V_j)^2}{R_{ij}} $$

Subject to:
1.  **Radial/Connected Constraint:** All stations must be energized.
2.  **Voltage Constraints:** $0.95 \le V_i \le 1.05$.
3.  **Thermal Limits:** $|I_{ij}| \le I_{max}$.
