# DS-PAH-GNN: Theoretical Framework & Architecture

## 1. Introduction
The **Data-Driven Physics-Aware Hierarchical Graph Neural Network (DS-PAH-GNN)** represents a paradigm shift in power system analysis. Instead of solving differential-algebraic equations (DAEs) iteratively (e.g., Newton-Raphson, which scales $\mathcal{O}(N^{3})$), we learn a mapping function $f_\theta: \mathcal{G} \to \mathbb{R}$ that predicts system state directly from topology.

This approach leverages the **Universal Approximation Theorem** for Graph Neural Networks (GNNs), positing that a sufficiently deep message-passing network can approximate any permutation-invariant function over graphs. This allows us to directly predict complex system behavior from the network's topological structure and component characteristics without manually deriving the system equations.

## 2. Heterogeneous Graph Representation
Power systems are inherently heterogeneous. A "Bus" is physically distinct from a "Battery". We formalize the microgrid as a Heterogeneous Graph $G = (\mathcal{V}, \mathcal{E}, \mathcal{R}, \mathcal{T})$.

### 2.1. Node Feature Engineering (X)
We define distinct feature spaces for different node types $\tau \in \mathcal{T}$:

1.  **Bus Nodes ($X_{bus} \in \mathbb{R}^{2}$):**
    *   $V$ (Voltage Magnitude): System state variable.
    *   $P$ (Active Power): Net injection (0 for transit buses).

2.  **Device Nodes ($X_{dev} \in \mathbb{R}^{4}$):**
    *   $V, P$: Standard electrical state.
    *   $SOC$ (State of Charge): Critical temporal state for batteries.
    *   $Type$: One-hot encoding or embedding of device class (PV vs Load).

### 2.2. Physics-Informed Edge Features ($E$)
Edges represent transmission lines. Embedding physical laws into edge attributes enhances the GNN's ability to learn and generalize. Instead of treating edges as simple binary connections, we enrich them with electrical characteristics:
$$ e_{ij} = \left[ G_{ij}, S_{ij} \right] = \left[ \frac{1}{R_{ij}}, \text{Status} \in \{0,1\} \right] $$
*   **Conductance ($G_{ij}$):** The inverse of resistance. This is crucial because electrical current flow is proportional to conductance ($I = V \cdot G$). By feeding $G$ to the network, we provide the "path of least resistance" explicitly.

    *   **Rationale:** Conductance provides a direct measure of how easily current flows between two nodes. Including it allows the GNN to directly model the electrical relationships dictated by Ohm's Law.

*   **Status (S\_ij)**: Binary indicator of switch state (1=Closed, 0=Open).

    *   **Rationale:** Indicates whether the connection is active, allowing the GNN to incorporate topology changes due to switching actions.

## 3. Physics-Aware Message Passing Mechanism
We utilize a modified **Graph Isomorphism Network (GIN)** convolution, adapted for continuous edge features.

### 3.1. The Convolution Operation
For a node $i$ and its neighbors $\mathcal{N}(i)$, the update rule at layer $k$ is:
$$ h_i^{(k)} = \text{MLP}^{(k)} \left( (1 + \epsilon^{(k)}) \cdot h_i^{(k-1)} + \sum_{j \in \mathcal{N}(i)} \text{ReLU}(h_j^{(k-1)} + \phi(e_{ij})) \right) $$

    *   **$h_i^{(k)}$**: The hidden state of node i at layer k. This represents the node's learned features after k message passing iterations.
    *   **$MLP^{(k)}$**: A Multi-Layer Perceptron (neural network) that transforms the aggregated information. This is the primary learning component, adapting the node features to capture relevant patterns.
    *   **$\epsilon^{(k)}$**: A learnable parameter that weights the node's own previous state. This is important for GINs to ensure they are as expressive as possible and can distinguish different graph structures.
    *   **ReLU**: Rectified Linear Unit activation function, introducing non-linearity to the message passing process.
    *   **$h_j^{(k-1)}$**: The hidden state of neighbor j at the previous layer (k-1). This is the information being passed *from* the neighbors.
    *   **$\phi(e_{ij})$**: A function (typically a linear transformation) applied to the edge features $e_{ij}$. This adapts the edge information to be compatible with the node features and incorporates the physical properties of the connection.

### 3.2. Alignment with Kirchhoff's Laws
This aggregation function structurally mimics **Kirchhoff's Current Law (KCL)**:
$$ \sum I_{in} = 0 \implies \sum_{j} (V_j - V_i) G_{ij} = 0 $$
By learning the weights $\phi$ applied to edge features $e_{ij}$ (which contain conductance), the GNN learns to weigh incoming messages (voltages/power) by their electrical connectivity, effectively "learning" Ohm's Law.

    *   **Implication:** This design choice makes the GNN "physics-aware," meaning it is predisposed to learn patterns that are consistent with the underlying electrical laws governing the microgrid.

## 4. Hierarchical Attention & Multi-Resolution Readout
Standard "Global Mean Pooling" destroys local structural information. A microgrid has critical local hotspots (e.g., an overloaded station). We employ **Self-Attention Graph Pooling (SAGPool)** to preserve this hierarchy.

### 4.1. Self-Attention Score Calculation
We compute a "criticality score" $Z \in \mathbb{R}^{N \times 1}$ for every node:
$$ Z = \sigma(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2} X \Theta_{att}) $$
Where $\tilde{A}$ is the adjacency matrix and $\Theta_{att}$ are learnable attention parameters.

### 4.2. Top-K Selection (The Masking)
We retain only the top $k$ nodes with the highest scores:
    $$ \text{idx} = \text{top-rank}(Z, \lceil k \cdot N \rceil) $$
    $$ X' = X_{\text{idx}, :}, \quad A' = A_{\text{idx}, \text{idx}} $$
This allows the network to focus on "stressed" nodes (e.g., buses with high power mismatch) while ignoring healthy parts of the grid.

    *   **$k$**: A hyperparameter that determines the fraction of nodes to retain.
    *   **$\lceil k \cdot N \rceil$**: The ceiling function ensures we retain at least one node, even if k is very small.
    *   **$X'$**: The reduced node feature matrix, containing only the features of the selected nodes.
    *   **$A'$**: The reduced adjacency matrix, representing the connections between the selected nodes.

### 4.3. Multi-Resolution Fusion
To make the final prediction, we combine two "views" of the grid:
1.  **Global View ($h_{global}$):** Mean pooling of *all* nodes. Captures total system load.
2.  **Local View ($h_{local}$):** Mean pooling of *only* the Top-K critical nodes. Captures local bottlenecks.

    $$ h_{final} = [ h_{global} \mathbin\Vert h_{local} ] $$

    *   **Rationale:** Combining global and local views allows the GNN to capture both overall system conditions and specific stress points, leading to more accurate predictions.

## 5. Computational Complexity & Scalability
*   **Traditional Solver:** $\mathcal{O}(N^{3})$ due to matrix inversion/factorization.
*   **DS-PAH-GNN:** $\mathcal{O}(|\mathcal{E}| + |\mathcal{V}|)$. Message passing is linear with respect to the number of edges.

This linear scaling allows for real-time topology optimization ($< 10$ ms) even as the microgrid scales to thousands of nodes, enabling sub-second control loops for grid stability.