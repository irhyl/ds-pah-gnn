# Hierarchical Dual-Bus DC Microgrid: System Modeling & Physics

## 1. Research Context & Motivation
The rapid electrification of transportation imposes significant stress on distribution networks. Fast-charging stations (XFC) introduce high-power, stochastic loads that can cause voltage instability and thermal overloading.

This research focuses on a **DC Microgrid** architecture. Unlike traditional AC grids, DC microgrids are particularly suited for modern energy systems because:
1.  **Native Compatibility:** PV panels, Batteries, and EVs are inherently DC devices.
2.  **Efficiency:** Eliminates redundant DC/AC and AC/DC conversion stages, reducing losses by 5-15%.
3.  **Controllability:** Power electronics allow for precise, rapid control of voltage and power flow.

## 2. Graph-Theoretic Topology Definition
We model the microgrid as a **Hierarchical Dual-Bus** system, represented formally as a graph $G = (\mathcal{V}, \mathcal{E})$.

### 2.1. The Hierarchy
The topology is structured in three distinct layers to ensure scalability and fault tolerance:

1.  **Level 0: The Main Bus (Root)**
    *   **Function:** Point of Common Coupling (PCC) to the utility grid. Acts as the slack bus (voltage reference).
    *   **Graph Node:** $v_{root} \in \mathcal{V}$.
    *   **Physics:** Infinite stiffness (constant voltage source).

2.  **Level 1: Station Buses (Aggregators)**
    *   **Function:** Local distribution hubs that aggregate Distributed Energy Resources (DERs).
    *   **Graph Nodes:** $\mathcal{V}_{station} \subset \mathcal{V}$.
    *   **Connectivity:** Connected to Root via *Feeder Lines* and to each other via *Tie-Lines*.

3.  **Level 2: Leaf Nodes (Devices)**
    *   **Function:** Sources and sinks of power.
    *   **Graph Nodes:** $\mathcal{V}_{device} = \mathcal{V}_{ev} \cup \mathcal{V}_{pv} \cup \mathcal{V}_{ess}$.

### 2.2. Edge Types & Switching
The edge set $\mathcal{E}$ contains static and dynamic elements:
*   **Static Edges:** Fixed connections (e.g., Station $\to$ Charger).
*   **Switchable Edges:** Reconfigurable lines used for topology optimization.
    *   *Feeders:* $\mathcal{E}_{feed} \subset \mathcal{E}$. Low resistance, high capacity.
    *   *Tie-Lines:* $\mathcal{E}_{tie} \subset \mathcal{E}$. Used for load balancing and redundancy.

## 3. Mathematical Modeling of Components

### 3.1. Electric Vehicles (Stochastic Loads)
EVs are modeled as negative power injections. The arrival process follows a non-homogeneous Poisson process $\lambda(t)$, and demand is stochastic:
$$ P_{ev}(t) = -\mathbb{I}(t) \cdot P_{rated} \cdot \eta $$
Where $\mathbb{I}(t)$ is the charging status (Bernoulli trial based on time-of-day probabilities).

### 3.2. Photovoltaic (PV) Generation
PV output is modeled as a function of solar irradiance $G(t)$ and temperature $T(t)$, approximated by a bell curve with Gaussian noise $\epsilon$:
$$ P_{pv}(t) = P_{max} \cdot \sin\left(\frac{\pi(t - t_{sunrise})}{t_{sunset} - t_{sunrise}}\right) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2) $$

### 3.3. Battery Energy Storage Systems (BESS)
Batteries are dynamic nodes with internal state (State of Charge - SOC).
$$ SOC(t+1) = SOC(t) + \frac{P_{batt}(t) \cdot \Delta t}{E_{cap}} $$
*   **Discharging ($P > 0$):** Supports the grid during peak load.
*   **Charging ($P < 0$):** Absorbs excess PV generation.

## 4. DC Power Flow Formulation
We employ a steady-state DC Power Flow model. In DC systems, reactive power ($Q$) is zero, and phase angles are irrelevant. The system is governed by Ohm's Law and Kirchhoff's Current Law (KCL).

### 4.1. The Nodal Admittance Matrix ($Y_{bus}$)
For a system with $N$ buses, the matrix $Y \in \mathbb{R}^{N \times N}$ is constructed as:
$$
Y_{ij} =
\begin{cases}
\sum_{k \neq i} \frac{1}{R_{ik}} & \text{if } i = j \text{ (Diagonal: Sum of connected conductances)} \\
-\frac{1}{R_{ij}} & \text{if } i \neq j \text{ and connected} \\
0 & \text{otherwise}
\end{cases}
$$
Note: $G_{ij} = \frac{1}{R_{ij}}$ is the conductance.

### 4.2. System of Linear Equations
The relationship between nodal voltages $V$ and current injections $I$ is linear:
$$ [Y] \cdot [V] = [I] $$

However, loads are typically Constant Power ($P$), not Constant Current ($I$). This creates a non-linearity ($I = P/V$).
**Approximation:** Since voltage deviations in microgrids are tightly regulated ($V \approx 1.0$ p.u.), we linearize by approximating current injections:
$$ I_i \approx \frac{P_i}{V_{ref}} $$
This allows us to solve for voltages using a fast linear solver (LU decomposition):
$$ V = Y^{-1} \cdot I $$

### 4.3. Loss Calculation (Objective Function)
The primary optimization objective is to minimize Ohmic distribution losses:
$$ \min J = \sum_{(i,j) \in \mathcal{E}} I_{ij}^2 R_{ij} = \sum_{(i,j) \in \mathcal{E}} \frac{(V_i - V_j)^2}{R_{ij}} $$

## 5. The Topology Optimization Problem
The problem of finding the optimal switch configuration is a **Mixed-Integer Non-Linear Programming (MINLP)** problem, which is NP-Hard.

$$
\begin{aligned}
\text{minimize} \quad & P_{loss}(x) \\
\text{subject to} \quad & V_{min} \le V_i \le V_{max} \quad \forall i \in \mathcal{V} \\
& |I_{ij}| \le I_{max} \quad \forall (i,j) \in \mathcal{E} \\
& g \in \mathcal{G}_{radial} \quad (\text{Radiality Constraint})
\end{aligned}
$$

Where $x$ is the binary vector of switch states. Our research replaces the computationally expensive iterative solvers required for this MINLP with a **Graph Neural Network** approximator.