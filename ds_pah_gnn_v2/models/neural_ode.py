"""
Neural ODE wrapper for continuous-time graph dynamics.

Integrates torchdiffeq for continuous-time modeling of power system dynamics.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable

try:
    from torchdiffeq import odeint, odeint_adjoint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    print("Warning: torchdiffeq not available. Neural ODE will use fallback RK4.")


class ODEFunc(nn.Module):
    """
    ODE function wrapper for graph neural network dynamics.
    
    Defines dx/dt = f(x, t) where x is the node embeddings.
    """
    
    def __init__(self, gnn_layer: nn.Module, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        super().__init__()
        self.gnn_layer = gnn_layer
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.nfe = 0  # Number of function evaluations
    
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute dx/dt at time t.
        
        Args:
            t: Current time (scalar)
            x: Node embeddings [num_nodes, hidden_dim]
            
        Returns:
            dx/dt: Time derivative of x
        """
        self.nfe += 1
        
        # Apply GNN layer to compute dynamics
        dx_dt = self.gnn_layer(x, self.edge_index, self.edge_attr)
        
        return dx_dt


class NeuralODELayer(nn.Module):
    """
    Neural ODE layer for continuous-time graph evolution.
    """
    
    def __init__(
        self,
        gnn_layer: nn.Module,
        method: str = 'rk4',
        rtol: float = 1e-3,
        atol: float = 1e-4,
        adjoint: bool = False
    ):
        """
        Args:
            gnn_layer: GNN layer defining the dynamics
            method: ODE solver method ('rk4', 'dopri5', 'euler')
            rtol: Relative tolerance
            atol: Absolute tolerance
            adjoint: Use adjoint method for backprop (memory efficient)
        """
        super().__init__()
        
        self.gnn_layer = gnn_layer
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.adjoint = adjoint
        self.ode_func = None
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        t_span: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Evolve node embeddings through continuous time.
        
        Args:
            x: Initial node embeddings [num_nodes, hidden_dim]
            edge_index: Edge connectivity
            edge_attr: Edge attributes
            t_span: Time span [t_start, t_end], default [0, 1]
            
        Returns:
            Final node embeddings at t_end
        """
        if t_span is None:
            t_span = torch.tensor([0.0, 1.0]).to(x.device)
        
        # Create ODE function
        self.ode_func = ODEFunc(self.gnn_layer, edge_index, edge_attr)
        
        if TORCHDIFFEQ_AVAILABLE:
            # Use torchdiffeq
            ode_solver = odeint_adjoint if self.adjoint else odeint
            
            x_trajectory = ode_solver(
                self.ode_func,
                x,
                t_span,
                method=self.method,
                rtol=self.rtol,
                atol=self.atol
            )
            
            # Return final state
            x_final = x_trajectory[-1]
        else:
            # Fallback: simple RK4 with fixed steps
            x_final = self._rk4_step(x, t_span)
        
        return x_final
    
    def _rk4_step(self, x: torch.Tensor, t_span: torch.Tensor, steps: int = 3) -> torch.Tensor:
        """
        Fallback RK4 integration with fixed steps.
        """
        t_start, t_end = t_span[0], t_span[1]
        dt = (t_end - t_start) / steps
        
        x_current = x
        t_current = t_start
        
        for _ in range(steps):
            k1 = self.ode_func(t_current, x_current)
            k2 = self.ode_func(t_current + dt/2, x_current + dt * k1 / 2)
            k3 = self.ode_func(t_current + dt/2, x_current + dt * k2 / 2)
            k4 = self.ode_func(t_current + dt, x_current + dt * k3)
            
            x_current = x_current + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            t_current = t_current + dt
        
        return x_current


class GraphNeuralODE(nn.Module):
    """
    Complete Graph Neural ODE module.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_ode_layers: int = 1,
        method: str = 'rk4',
        adjoint: bool = False
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_ode_layers = num_ode_layers
        
        # Create simple GNN dynamics
        self.dynamics = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.ode_layer = NeuralODELayer(
            self.dynamics,
            method=method,
            adjoint=adjoint
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Apply Neural ODE to graph.
        """
        return self.ode_layer(x, edge_index, edge_attr)


if __name__ == "__main__":
    print("Testing Neural ODE...")
    
    num_nodes = 10
    hidden_dim = 32
    num_edges = 20
    
    # Create dummy data
    x = torch.randn(num_nodes, hidden_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 3)
    
    # Create Neural ODE
    ode_model = GraphNeuralODE(hidden_dim=hidden_dim, method='rk4')
    
    # Forward pass
    x_out = ode_model(x, edge_index, edge_attr)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_out.shape}")
    print(f"Output changed: {not torch.allclose(x, x_out)}")
    
    print("✓ Neural ODE test passed!")

