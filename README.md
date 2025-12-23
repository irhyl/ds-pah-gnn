# DS-PAH-GNN: Physics-Aware Hierarchical GNN for Microgrid Topology Optimization

## Overview
This project implements a **Data-Driven Digital Twin** for a Dual-Bus DC Microgrid. It uses a **Physics-Aware Graph Neural Network (GNN)** to predict power losses and optimize grid topology in real-time, bypassing slow traditional physics solvers.

### Key Features
*   **Physics-Aware AI:** Uses `GINEConv` layers that explicitly incorporate electrical conductance ($1/R$) into message passing.
*   **Heterogeneous Modeling:** Distinctly models Main Buses, Local Stations, EV Chargers, PV Units, and Battery Storage.
*   **Topology Optimization:** Automatically identifies the optimal switch configuration to minimize losses during load spikes.


## Installation

1.  **Prerequisites:** Python 3.8+, CUDA (optional for GPU support).
2.  **Install Dependencies:**
    ```bash
    cd microgrid-design
    pip install -r requirements.txt
    ```

## Usage Pipeline

To replicate the research results, run the pipeline in this order:

### 1. Generate Data (The Physics Simulation)
Simulates 600,000 grid snapshots with stochastic EV loads and solar generation.
```bash
python generate.py
```

### 2. Train the Model (The Learning Phase)
Trains the GNN to predict power loss from graph topology. Generates `training_loss.png`.
```bash
python train.py
```

### 3. Evaluate Accuracy (The Validation Phase)
Calculates $R^2$, MAE, and RMSE on unseen data. Generates `inference_plot.png`.
```bash
python inference.py
```

### 4. Run Optimization (The Application)
Finds the optimal topology for a stressed grid scenario (e.g., Station Overload).
```bash
python optimize.py
```

## License
Research Use Only.