# DS-PAH-GNN: Physics-Aware Hierarchical GNN for Microgrid Topology Optimization

[![License: Research](https://img.shields.io/badge/License-Research-blue.svg)](#license)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-GNN-orange)](https://pytorch.org/)

## Overview

This project implements a **Data-Driven Digital Twin** for a Dual-Bus DC Microgrid. It uses a **Physics-Aware Graph Neural Network (GNN)** to predict power losses and optimize grid topology in real-time, bypassing slow traditional physics solvers. This approach allows for rapid grid reconfiguration and optimization, enhancing stability and efficiency.

### Key Features
*   **Physics-Aware AI:** Uses `GINEConv` layers that explicitly incorporate electrical conductance ($1/R$) into message passing.
*   **Heterogeneous Modeling:** Distinctly models Main Buses, Local Stations, EV Chargers, PV Units, and Battery Storage.
*   **Topology Optimization:** Automatically identifies the optimal switch configuration to minimize losses during load spikes.

## Table of Contents

1.  Installation
2.  Usage Pipeline
3.  Documentation
4.  File Structure
5.  Configuration
6.  License

## Installation

1.  **Prerequisites:** Python 3.8+, CUDA (optional for GPU support).
2.  **Install Dependencies:**
    ```bash
    cd ds-pah-gnn
    pip install -r requirements.txt
    ```

## Usage Pipeline

The core workflow involves simulating a microgrid environment, training a GNN to predict system states, and then using the trained model to optimize grid topology in real-time.

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