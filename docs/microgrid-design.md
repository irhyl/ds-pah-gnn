# DC Microgrid Design Specification

## 1. System Overview

The simulation environment models a **Hierarchical Dual-Bus DC Microgrid** designed for EV fast-charging applications. The system is modeled as a directed graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ where nodes represent electrical buses/components and edges represent transmission lines.

## 2. Topology Structure
The network follows a 3-tier hierarchy:

1.  **Level 0 (Root):** Point of Common Coupling (PCC). Acts as the slack bus (Voltage Source) with fixed voltage $V_{ref} = 1.0$ p.u.
2.  **Level 1 (Stations):** Distribution substations connected to the root via **Feeder Lines**.
3.  **Level 2 (Devices):** Leaf nodes connected to stations via **Branch Lines**.
    *   **EV Chargers:** Stochastic loads following Constant-Current Constant-Voltage (CC-CV) charging curves.
    *   **PV Inverters:** Stochastic generation sources.
    *   **Battery Storage (BESS):** Bidirectional energy buffers.

Additionally, **Tie-Lines** connect adjacent stations in a ring configuration, allowing for topology reconfiguration (load transfer) during congestion.

## 3. Physics Engine
The system state is resolved using a **Linearized DC Power Flow** approximation ($P \approx V(V_i - V_j)/R$), which reduces the non-linear power flow equations to a linear system of the form:

    $$ \mathbf{Y} \mathbf{V} = \mathbf{I} $$

Where:
- $\mathbf{Y}$: Nodal Admittance Matrix (constructed from $1/R_{ij}$).
- $\mathbf{V}$: Vector of nodal voltages (unknowns).
- $\mathbf{I}$: Vector of current injections ($I \approx P/V_{nom}$).

    *   **Rationale:** Linearization simplifies the power flow equations, enabling faster computation and dataset generation. This is crucial for training the GNN.

### Component Modeling

| Component | Type | Model Dynamics |
| :--- | :--- | :--- |
| **Grid (PCC)** | Slack Bus | Infinite source, maintains $V=1.0$ p.u. |
| **EV Charger** | Load ($P < 0$) | Stochastic arrival + CC-CV profile ($P_{max} \to P_{decay}$). |
|  |   | *Note: CC-CV charging models the real-world behavior of EV batteries, where charging occurs at a constant current until a voltage threshold, then maintains the voltage while current decreases.* |
| **PV Array** | Gen ($P > 0$) | Diurnal sine wave + Gaussian cloud noise. |
|  |   | *Note: This captures the typical daily solar generation profile with random fluctuations due to cloud cover, creating realistic conditions.* |
| **Storage** | Buffer | State-of-Charge (SOC) tracking: $SOC_{t+1} = SOC_t + \eta P_t \Delta t$. |
|  |   | *Note: Batteries are crucial for microgrid stability. The SOC-based model simulates charging/discharging behavior.* |

## 4. Graph Feature Engineering

To train the GNN, physical quantities are mapped to tensor features.

### Node Features ($X$)
| Index | Feature | Description |
| :--- | :--- | :--- |
| 0 | Voltage ($V$) | Current voltage magnitude (p.u.). |
| 1 | Power ($P$) | Net active power injection (+Gen, -Load). |
| 2 | SOC | State of Charge (0.0-1.0) for storage; 0 for others. |
| 3 | Type | One-hot encoding of component type. |


### Edge Features ($E$)
| Index | Feature | Description |
| :--- | :--- | :--- |
| 0 | Resistance ($R$) | Line resistance (p.u.). |
| 1 | Status ($S$) | Switch state (1=Closed, 0=Open). |
| 2 | Conductance ($G$) | Derived feature $G = S/R$. |


## 5. Configuration Parameters

Default electrical parameters used in `microgrid.py`:

| Parameter | Value (p.u.) | Description |

| :--- | :--- | :--- |
| `v_source` | 1.00 | Nominal grid voltage. |
| `r_feeder` | 0.01 | Resistance of main feeder lines. |
| `r_branch` | 0.05 | Resistance of local branch lines. |
| `r_tie` | 0.03 | Resistance of reconfiguration tie-lines. |
| `load_scale`| 0.50 | Base scaling factor for loads. |

## 6. Optimization Objective

The goal is to find the optimal set of switch states $S_{ij} \in \{0, 1\}$ for feeders and tie-lines that minimizes total system loss:

    $$ \min \sum_{(i,j) \in \mathcal{E}} S_{ij} \cdot \frac{(V_i - V_j)^2}{R_{ij}} $$

Subject to:
1.  **Radial/Connected Constraint:** All stations must be energized.
2.  **Voltage Constraints:** $0.95 \le V_i \le 1.05$.
3.  **Thermal Limits:** $|I_{ij}| \le I_{max}$.

## 7. Data Generation Strategy
To ensure robust generalization, the dataset covers:
1.  **Topology Variations:** Random switching of feeders and ties (keeping connectivity).
2.  **Load Scenarios:** 24-hour simulation with stochastic noise on PV/EV.
3.  **Battery States:** Randomized initial SOC to learn charging/discharging value.