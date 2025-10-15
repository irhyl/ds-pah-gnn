# DS-PAH-GNN v2.0

**Dual-Stream Physics-Aware Hierarchical Graph Neural Network for Power Distribution System Optimization**

A state-of-the-art deep learning framework for power distribution network reconfiguration, integrating:
- ⚡ Physics-aware constraints (KCL, voltage bounds)
- 🔄 Switchable Edge Operators (SEO) for topology optimization
- 🌐 Dual-stream architecture (local + global dynamics)
- 🎯 Uncertainty quantification
- 🚀 Continuous-time modeling via Neural ODEs
- 🛡️ Robust training with adversarial perturbations
- 🤖 Meta-learning and RL fine-tuning support

---

## 📋 Features

### Core Architecture
- **LocalStream**: Node-level dynamics with GRU temporal updates
- **GlobalStream**: System-level coordination via hierarchical pooling
- **CrossHierarchyFusion**: Attention-based integration of local and global streams
- **Switchable Edge Operators**: Differentiable topology reconfiguration

### Advanced Capabilities
- Physics-based losses (power flow, KCL, voltage constraints)
- Convex feasibility projection (CVXPY)
- Temporal encoding for time-series data
- Multi-agent decomposition
- Explainability (GNNExplainer, integrated gradients)
- Knowledge distillation for edge deployment

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ds_pah_gnn_v2

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Smoke Test

Run a quick test to verify everything is working:

**Linux/Mac:**
```bash
bash scripts/run_smoke.sh
```

**Windows (PowerShell):**
```powershell
.\scripts\run_smoke.ps1
```

This will:
1. Test the topology generator
2. Verify model components
3. Run a short training loop (5 epochs)

### Training

**Basic training:**
```bash
python training/train.py --config experiments/configs/train_base.yaml
```

**Smoke test (fast):**
```bash
python training/train.py --config experiments/configs/smoke.yaml
```

---

## 📁 Project Structure

```
ds_pah_gnn_v2/
├── data/
│   ├── generators/          # Topology and data generation
│   │   ├── topology.py      # Power network graph generation
│   │   ├── loads.py         # Load profile generation
│   │   ├── augment_physics.py
│   │   └── adversarial.py
│   ├── real_traces/         # Real-world data traces
│   └── processed/           # Processed datasets
│
├── models/
│   ├── ds_pah_gnn.py       # Main model architecture
│   ├── switchable_ops.py   # Switchable Edge Operators
│   ├── physics_module.py   # Physics constraints
│   ├── temporal_encoder.py # Temporal modeling
│   ├── neural_ode.py       # Neural ODE wrapper
│   ├── multi_agent.py      # Multi-agent decomposition
│   ├── uncertainty_head.py # Uncertainty quantification
│   ├── explainability.py   # Explainability tools
│   └── distillation.py     # Knowledge distillation
│
├── training/
│   ├── train.py            # Main training script
│   ├── pretrain_contrastive.py
│   ├── maml_adapt.py       # Meta-learning
│   ├── rl_finetune.py      # RL fine-tuning
│   └── distill_train.py
│
├── eval/
│   ├── eval_metrics.py
│   ├── baseline_milp.py    # MILP baselines
│   ├── robustness_tests.py
│   ├── explainability_eval.py
│   └── uncertainty_eval.py
│
├── utils/
│   ├── graph_helpers.py    # Graph utilities
│   ├── projectors.py       # Convex projection
│   ├── safety_certifier.py
│   ├── hil_interface.py    # Hardware-in-loop
│   ├── viz.py              # Visualization
│   └── profiler.py
│
├── experiments/
│   ├── configs/            # Configuration files
│   └── runs/               # Experiment outputs
│
├── tests/                  # Unit tests
├── scripts/                # Utility scripts
├── notebooks/              # Analysis notebooks
└── docker/                 # Docker configuration
```

---

## 🔧 Configuration

Configuration files are in `experiments/configs/`. Key parameters:

### Model Architecture
```yaml
model:
  hidden_dim: 128           # Hidden dimension
  local_layers: 4           # Local stream layers
  global_layers: 3          # Global stream layers
  use_uncertainty: false    # Enable UQ
```

### Training
```yaml
training:
  batch_size: 16
  epochs: 200
  lr: 0.001
  robust_training: true     # Enable robust training
```

### Loss Weights
```yaml
loss_weights:
  lambda_task: 1.0          # Task loss
  lambda_phys: 10.0         # Physics loss
  lambda_voltage: 20.0      # Voltage constraints
  lambda_switch: 1.0        # Switch regularization
```

---

## 🧪 Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific test modules:
```bash
pytest tests/test_generators.py -v
pytest tests/test_model_shapes.py -v
pytest tests/test_physics_module.py -v
```

---

## 📊 Evaluation

### Metrics
- **Energy loss reduction**: Efficiency improvement
- **Voltage violations**: Constraint satisfaction
- **Switch operations**: Topology changes
- **Inference latency**: Real-time performance

### Run evaluation:
```bash
python eval/eval_metrics.py --checkpoint path/to/model.pt --split test
```

---

## 🎯 Roadmap

### Sprint 0 ✅ (Completed)
- [x] Repository setup
- [x] Topology generator
- [x] Model skeleton
- [x] Smoke tests

### Sprint 1 (In Progress)
- [ ] Complete physics module
- [ ] Implement projectors
- [ ] Local stream with GRU
- [ ] Unit tests for physics

### Sprint 2
- [ ] Global stream with pooling
- [ ] Temporal encoder
- [ ] Hierarchical architecture
- [ ] Visualization tools

### Sprint 3
- [ ] Neural ODE integration
- [ ] Robust training
- [ ] Dataset scaling
- [ ] Batch processing

### Sprint 4
- [ ] Multi-agent decomposition
- [ ] Uncertainty quantification
- [ ] Explainability modules
- [ ] Advanced metrics

### Sprint 5
- [ ] Contrastive pretraining
- [ ] MAML adaptation
- [ ] Baseline implementations
- [ ] Ablation studies

### Sprint 6
- [ ] RL fine-tuning (PPO)
- [ ] Knowledge distillation
- [ ] HIL interface
- [ ] Edge deployment

### Sprint 7
- [ ] Full experiments
- [ ] Paper figures
- [ ] Reproducibility artifacts
- [ ] Documentation

---

## 📚 Key Concepts

### Physics-Aware Learning
The model integrates physical constraints directly into the learning process:
- **Kirchhoff's Current Law (KCL)**: Ensures current conservation at nodes
- **Voltage bounds**: Maintains safe operating ranges
- **Power flow equations**: Linearized DC approximation

### Switchable Edge Operators (SEO)
Novel operators that learn different message passing functions for switches in on/off states:
```python
O_eff = α · O_on + (1 - α) · O_off
```
where α ∈ [0,1] is learned from edge features.

### Dual-Stream Architecture
- **LocalStream**: Captures node-level dynamics and local interactions
- **GlobalStream**: Models system-level coordination and global constraints
- **Fusion**: Cross-attention mechanism integrates both perspectives

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

## 📄 License

[Specify your license here]

---

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@software{ds_pah_gnn_v2,
  title={DS-PAH-GNN v2.0: Dual-Stream Physics-Aware Hierarchical GNN},
  author={[Your Name]},
  year={2025},
  url={[Repository URL]}
}
```

---

## 🐛 Troubleshooting

### CVXPY installation issues
If you encounter issues with CVXPY:
```bash
pip install cvxpy --no-build-isolation
```

### CUDA/GPU issues
Ensure PyTorch is installed with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Import errors
Make sure to run scripts from the project root:
```bash
cd ds_pah_gnn_v2
python training/train.py --config experiments/configs/smoke.yaml
```

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].

---

## 🙏 Acknowledgments

- PyTorch Geometric for graph neural network primitives
- CVXPY for convex optimization
- The power systems community for domain knowledge

---

**Built with ❤️ for sustainable and intelligent power systems**

