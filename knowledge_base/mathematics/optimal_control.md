# Optimal Control Theory in Cognitive Modeling

---
type: mathematical_concept
id: optimal_control_001
created: 2024-02-06
modified: 2024-02-06
tags: [mathematics, optimal-control, control-theory, optimization]
aliases: [control-theory, dynamic-optimization]
semantic_relations:
  - type: implements
    links: 
      - [[../../docs/research/research_documentation_index|Research Documentation]]
      - [[active_inference_theory]]
  - type: uses
    links:
      - [[variational_calculus]]
      - [[dynamic_programming]]
  - type: documented_by
    links:
      - [[../../docs/guides/implementation_guides_index|Implementation Guides]]
      - [[../../docs/api/api_documentation_index|API Documentation]]
---

## Overview

Optimal control theory provides the mathematical foundation for understanding action selection and policy optimization in cognitive systems. This document explores control theoretic principles and their applications in active inference.

## Dynamic Systems

### State Space Models
```python
class StateSpaceModel:
    """
    State space model implementation.
    
    Theory:
        - [[state_space]]
        - [[dynamical_systems]]
        - [[control_systems]]
    Mathematics:
        - [[differential_equations]]
        - [[control_theory]]
    """
    def __init__(self,
                 state_dim: int,
                 control_dim: int,
                 dynamics: Callable):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.f = dynamics
        
    def evolve_state(self,
                    state: np.ndarray,
                    control: np.ndarray,
                    dt: float) -> np.ndarray:
        """Evolve state forward in time."""
        return self._integrate_dynamics(state, control, dt)
    
    def linearize(self,
                 state: np.ndarray,
                 control: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Linearize dynamics around point."""
        A = self._compute_state_jacobian(state, control)
        B = self._compute_control_jacobian(state, control)
        return A, B
```

### Controlled Dynamics
```python
class ControlledSystem:
    """
    Controlled dynamical system implementation.
    
    Theory:
        - [[control_systems]]
        - [[feedback_control]]
        - [[stability_theory]]
    Mathematics:
        - [[control_theory]]
        - [[lyapunov_theory]]
    """
    def __init__(self,
                 model: StateSpaceModel,
                 controller: Controller):
        self.model = model
        self.controller = controller
        
    def closed_loop_dynamics(self,
                           state: np.ndarray,
                           reference: np.ndarray) -> np.ndarray:
        """Compute closed-loop dynamics."""
        # Get control input
        u = self.controller.compute_control(state, reference)
        
        # Apply dynamics
        return self.model.evolve_state(state, u)
    
    def simulate_trajectory(self,
                          initial_state: np.ndarray,
                          reference_trajectory: np.ndarray,
                          time_horizon: int) -> np.ndarray:
        """Simulate controlled system trajectory."""
        return self._simulate_system(
            initial_state, reference_trajectory, time_horizon
        )
```

## Optimal Control

### Hamilton-Jacobi-Bellman
```python
class HJBSolver:
    """
    Hamilton-Jacobi-Bellman equation solver.
    
    Theory:
        - [[optimal_control]]
        - [[dynamic_programming]]
        - [[hamilton_jacobi_bellman]]
    Mathematics:
        - [[partial_differential_equations]]
        - [[viscosity_solutions]]
    """
    def __init__(self,
                 model: StateSpaceModel,
                 cost: CostFunction):
        self.model = model
        self.cost = cost
        
    def value_function(self,
                      state: np.ndarray,
                      time: float) -> float:
        """Compute optimal value function."""
        return self._solve_hjb_equation(state, time)
    
    def optimal_control(self,
                       state: np.ndarray,
                       time: float) -> np.ndarray:
        """Compute optimal control."""
        # Value function gradient
        V_x = self._value_gradient(state, time)
        
        # Minimize Hamiltonian
        return self._minimize_hamiltonian(state, V_x)
```

### Pontryagin Maximum Principle
```python
class PontryaginSolver:
    """
    Pontryagin maximum principle solver.
    
    Theory:
        - [[maximum_principle]]
        - [[optimal_control]]
        - [[calculus_of_variations]]
    Mathematics:
        - [[hamiltonian_mechanics]]
        - [[boundary_value_problems]]
    """
    def __init__(self,
                 model: StateSpaceModel,
                 cost: CostFunction,
                 constraints: Constraints):
        self.model = model
        self.cost = cost
        self.constraints = constraints
        
    def solve_bvp(self,
                  initial_state: np.ndarray,
                  final_state: np.ndarray,
                  time_horizon: float) -> Tuple[np.ndarray, np.ndarray]:
        """Solve optimal control boundary value problem."""
        # Setup Hamiltonian system
        def hamiltonian_system(t, state_costate):
            return self._compute_hamiltonian_flow(t, state_costate)
        
        # Solve BVP
        solution = solve_bvp(
            hamiltonian_system,
            self._boundary_conditions,
            time_horizon,
            initial_state,
            final_state
        )
        
        return solution.y[:self.state_dim], solution.y[self.state_dim:]
```

## Policy Optimization

### Linear Quadratic Regulator
```python
class LQRController:
    """
    Linear quadratic regulator implementation.
    
    Theory:
        - [[lqr_control]]
        - [[optimal_control]]
        - [[feedback_control]]
    Mathematics:
        - [[riccati_equations]]
        - [[linear_systems]]
    """
    def __init__(self,
                 A: np.ndarray,
                 B: np.ndarray,
                 Q: np.ndarray,
                 R: np.ndarray):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        
    def compute_gains(self) -> np.ndarray:
        """Compute optimal feedback gains."""
        # Solve Riccati equation
        P = self._solve_care()
        
        # Compute gains
        K = np.linalg.solve(self.R, self.B.T @ P)
        
        return K
    
    def optimal_control(self,
                       state: np.ndarray) -> np.ndarray:
        """Compute optimal control input."""
        K = self.compute_gains()
        return -K @ state
```

### Model Predictive Control
```python
class MPController:
    """
    Model predictive control implementation.
    
    Theory:
        - [[model_predictive_control]]
        - [[receding_horizon]]
        - [[optimal_control]]
    Mathematics:
        - [[optimization]]
        - [[numerical_methods]]
    """
    def __init__(self,
                 model: StateSpaceModel,
                 cost: CostFunction,
                 horizon: int):
        self.model = model
        self.cost = cost
        self.N = horizon
        
    def optimize_trajectory(self,
                          initial_state: np.ndarray,
                          reference: np.ndarray) -> np.ndarray:
        """Optimize control trajectory."""
        # Setup optimization problem
        problem = self._setup_optimization(
            initial_state, reference
        )
        
        # Solve for optimal controls
        solution = solve_nonlinear_program(problem)
        
        return solution.x[:self.N * self.model.control_dim]
    
    def apply_control(self,
                     state: np.ndarray,
                     reference: np.ndarray) -> np.ndarray:
        """Apply first control input."""
        u_opt = self.optimize_trajectory(state, reference)
        return u_opt[:self.model.control_dim]
```

## Applications to Active Inference

### Free Energy Control
```python
class FreeEnergyController:
    """
    Free energy based control implementation.
    
    Theory:
        - [[active_inference]]
        - [[free_energy_principle]]
        - [[optimal_control]]
    Mathematics:
        - [[information_theory]]
        - [[control_theory]]
    """
    def __init__(self,
                 generative_model: GenerativeModel,
                 action_space: ActionSpace):
        self.model = generative_model
        self.actions = action_space
        
    def compute_expected_free_energy(self,
                                   belief_state: np.ndarray,
                                   policy: np.ndarray) -> float:
        """Compute expected free energy of policy."""
        # Epistemic value
        information_gain = self._compute_information_gain(
            belief_state, policy
        )
        
        # Pragmatic value
        expected_reward = self._compute_expected_reward(
            belief_state, policy
        )
        
        return information_gain + expected_reward
    
    def select_optimal_policy(self,
                            belief_state: np.ndarray,
                            policies: List[np.ndarray]) -> np.ndarray:
        """Select policy minimizing expected free energy."""
        # Compute EFE for all policies
        G = np.array([
            self.compute_expected_free_energy(belief_state, pi)
            for pi in policies
        ])
        
        # Softmax selection
        p = softmax(-G)
        return policies[np.argmax(p)]
```

### Active Inference Control
```python
class ActiveInferenceController:
    """
    Active inference control implementation.
    
    Theory:
        - [[active_inference]]
        - [[optimal_control]]
        - [[belief_updating]]
    Mathematics:
        - [[control_theory]]
        - [[information_geometry]]
    """
    def __init__(self,
                 model: GenerativeModel,
                 policy_space: PolicySpace,
                 temperature: float = 1.0):
        self.model = model
        self.policies = policy_space
        self.beta = temperature
        
    def infer_state(self,
                   observation: np.ndarray) -> np.ndarray:
        """Infer current state."""
        return self.model.infer_state(observation)
    
    def select_action(self,
                     belief_state: np.ndarray) -> np.ndarray:
        """Select action using active inference."""
        # Get available policies
        policies = self.policies.get_policies(belief_state)
        
        # Compute expected free energy
        G = np.array([
            self.model.compute_efe(belief_state, pi)
            for pi in policies
        ])
        
        # Policy selection
        p = softmax(-self.beta * G)
        selected_policy = policies[np.argmax(p)]
        
        # Return first action
        return selected_policy[0]
```

## Implementation Considerations

### Numerical Methods
```python
# @numerical_methods
numerical_implementations = {
    "optimal_control": {
        "shooting": "Single/multiple shooting",
        "collocation": "Direct collocation",
        "differential_dp": "Differential dynamic programming"
    },
    "trajectory_optimization": {
        "sqp": "Sequential quadratic programming",
        "ilqr": "Iterative LQR",
        "ddp": "Differential dynamic programming"
    },
    "mpc": {
        "condensed": "Condensed formulation",
        "sparse": "Sparse formulation",
        "rtc": "Real-time iteration"
    }
}
```

### Computational Efficiency
```python
# @efficiency_considerations
efficiency_methods = {
    "policy_optimization": {
        "warm_start": "Warm start from previous solution",
        "adaptive_horizon": "Adaptive horizon length",
        "parallel_rollouts": "Parallel policy evaluation"
    },
    "trajectory_optimization": {
        "matrix_caching": "Cache matrix computations",
        "constraint_handling": "Efficient constraint projection",
        "line_search": "Adaptive line search"
    },
    "real_time_control": {
        "partial_update": "Partial solution update",
        "early_termination": "Early termination criteria",
        "solution_reuse": "Solution reuse strategies"
    }
}
```

## Documentation Links
- [[../../docs/research/research_documentation_index|Research Documentation]]
- [[../../docs/guides/implementation_guides_index|Implementation Guides]]
- [[../../docs/api/api_documentation_index|API Documentation]]
- [[../../docs/examples/usage_examples_index|Usage Examples]]

## References
- [[bertsekas]] - Dynamic Programming and Optimal Control
- [[kirk]] - Optimal Control Theory
- [[lewis]] - Optimal Control
- [[rawlings]] - Model Predictive Control 