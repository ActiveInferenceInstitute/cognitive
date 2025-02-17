---
type: mathematical_concept
id: variational_free_energy_001
created: 2024-03-15
modified: 2024-03-15
tags: [free-energy, variational-methods, active-inference, statistical-mechanics, information-theory]
aliases: [VFE, evidence-lower-bound, ELBO]
complexity: advanced
processing_priority: 1
semantic_relations:
  - type: implements
    links: 
      - [[active_inference]]
      - [[free_energy_principle]]
      - [[policy_selection]]
  - type: mathematical_basis
    links:
      - [[variational_calculus]]
      - [[information_geometry]]
      - [[statistical_mechanics]]
  - type: relates
    links:
      - [[expected_free_energy]]
      - [[path_integral_free_energy]]
      - [[belief_updating]]
---

# Variational Free Energy

## Overview

Variational Free Energy (VFE) is a fundamental quantity in active inference that provides a measure of the discrepancy between an agent's beliefs and reality. It serves as a unified objective function for perception, learning, and action selection in both discrete and continuous time formulations.

## Mathematical Framework

### Core Definition

The Variational Free Energy is defined as:

```math
F[q] = \mathbb{E}_q[\ln q(s) - \ln p(o,s)]
```

where:
- $q(s)$ is the variational density over states
- $p(o,s)$ is the generative model
- $\mathbb{E}_q$ denotes expectation under $q$

### Discrete Time Formulation

In discrete time, VFE decomposes into:

```math
F_t = \underbrace{\text{KL}[q(s_t)||p(s_t|o_{1:t})]}_{\text{Accuracy}} - \underbrace{\ln p(o_t|o_{1:t-1})}_{\text{Complexity}}
```

### Continuous Time Extension

The continuous time formulation extends VFE to:

```math
F[q] = \int_t^{t+T} \left[\mathbb{E}_q[\ln q(s(\tau)) - \ln p(o(\tau),s(\tau))] + \frac{1}{2}\mathbb{E}_q[(\dot{s} - f(s,a))^T\Gamma(\dot{s} - f(s,a))]\right] d\tau
```

## Connection to Policy Selection

### Discrete Time Policy Selection

Policy selection in discrete time uses VFE through:

```math
P(\pi) = \sigma(-\gamma G(\pi))
```

where $G(\pi)$ is the expected free energy:

```math
G(\pi) = \sum_{\tau} \mathbb{E}_{q(o_\tau,s_\tau|\pi)}[\ln q(s_\tau|\pi) - \ln p(o_\tau,s_\tau|\pi)]
```

### Continuous Time Policy Selection

In continuous time, policy selection becomes:

```math
P(\pi) = \sigma(-\gamma \int_t^{t+T} \mathcal{L}_\pi(s(\tau), \dot{s}(\tau), a(\tau)) d\tau)
```

where $\mathcal{L}_\pi$ is the policy-specific Lagrangian.

## Implementation Framework

### 1. Variational Free Energy Computer

```python
class VFEComputer:
    """Computes Variational Free Energy in both discrete and continuous time"""
    def __init__(self):
        self.components = {
            'discrete': DiscreteVFE(),
            'continuous': ContinuousVFE(),
            'policy': PolicyVFE()
        }
        
    def compute_vfe(self, 
                   beliefs: Distribution,
                   observations: np.ndarray,
                   time_mode: str = 'discrete') -> float:
        """Compute VFE based on time mode"""
        if time_mode == 'discrete':
            return self.components['discrete'].compute(
                beliefs, observations)
        else:
            return self.components['continuous'].compute(
                beliefs, observations)
                
    def compute_policy_vfe(self,
                          policy: Policy,
                          horizon: int,
                          time_mode: str = 'discrete') -> float:
        """Compute policy-specific VFE"""
        return self.components['policy'].compute(
            policy, horizon, time_mode)
```

### 2. Belief Updating

```python
class BeliefUpdater:
    """Updates beliefs using VFE minimization"""
    def __init__(self):
        self.vfe = VFEComputer()
        self.optimizer = VariationalOptimizer()
        
    def update_beliefs(self,
                      current_beliefs: Distribution,
                      observation: np.ndarray,
                      time_mode: str = 'discrete') -> Distribution:
        """Update beliefs by minimizing VFE"""
        def objective(params):
            proposed_beliefs = self.construct_beliefs(params)
            return self.vfe.compute_vfe(
                proposed_beliefs, observation, time_mode)
                
        optimal_params = self.optimizer.minimize(objective)
        return self.construct_beliefs(optimal_params)
```

### 3. Policy Selection

```python
class PolicySelector:
    """Selects policies using VFE-based evaluation"""
    def __init__(self):
        self.vfe = VFEComputer()
        
    def select_policy(self,
                     policies: List[Policy],
                     horizon: int,
                     time_mode: str = 'discrete') -> Policy:
        """Select policy by minimizing expected VFE"""
        policy_vfes = []
        
        for policy in policies:
            vfe = self.vfe.compute_policy_vfe(
                policy, horizon, time_mode)
            policy_vfes.append(vfe)
            
        return policies[np.argmin(policy_vfes)]
```

## Advanced Concepts

### 1. Hierarchical VFE

The hierarchical extension of VFE:

```math
F_{\text{hierarchical}} = \sum_{l=1}^L F_l + \text{KL}[q_l(s_l)||p_l(s_l|s_{l+1})]
```

### 2. Amortized VFE

Using neural networks for efficient VFE computation:

```python
class AmortizedVFE:
    def __init__(self):
        self.encoder = ProbabilisticEncoder()
        self.decoder = ProbabilisticDecoder()
        
    def compute_amortized_vfe(self, x):
        """Compute VFE using amortized inference"""
        q_params = self.encoder(x)
        p_params = self.decoder(q_params)
        return self.compute_elbo(x, q_params, p_params)
```

### 3. Stochastic VFE

Extension to stochastic dynamics:

```math
F_{\text{stochastic}} = F + \mathbb{E}_q[\frac{1}{2}\text{tr}(D\nabla^2\ln q)]
```

## Planning with VFE

### 1. VFE-based Planning

```python
class VFEPlanner:
    """Plans actions using VFE minimization"""
    def __init__(self):
        self.vfe = VFEComputer()
        self.trajectory_optimizer = TrajectoryOptimizer()
        
    def plan_trajectory(self,
                       initial_state: np.ndarray,
                       goal_state: np.ndarray,
                       horizon: int) -> Trajectory:
        """Plan trajectory by minimizing VFE"""
        def objective(trajectory):
            return self.vfe.compute_trajectory_vfe(
                trajectory, goal_state)
                
        return self.trajectory_optimizer.optimize(
            objective, initial_state, horizon)
```

### 2. Active Inference Planning

```python
class ActiveInferencePlanner:
    """Implements active inference planning using VFE"""
    def __init__(self):
        self.vfe = VFEComputer()
        self.policy_selector = PolicySelector()
        
    def plan_actions(self,
                    beliefs: Distribution,
                    policies: List[Policy],
                    horizon: int) -> List[Action]:
        """Plan actions using active inference"""
        selected_policy = self.policy_selector.select_policy(
            policies, horizon)
            
        return self.extract_action_sequence(
            selected_policy, beliefs)
```

## Applications

### 1. Perception
- Belief updating
- State estimation
- Parameter learning

### 2. Action
- Policy selection
- Motor control
- Decision making

### 3. Learning
- Model learning
- Skill acquisition
- Habit formation

## References
- [[friston_2010]] - "The free-energy principle: a unified brain theory?"
- [[bogacz_2017]] - "A tutorial on the free-energy framework for modelling perception and learning"
- [[buckley_2017]] - "The free energy principle for action and perception: A mathematical review"

## See Also
- [[active_inference]]
- [[expected_free_energy]]
- [[path_integral_free_energy]]
- [[belief_updating]]
- [[policy_selection]]

## Theoretical Foundations

### Connection to Free Energy Principle

The relationship between VFE and the Free Energy Principle can be expressed through:

```math
\begin{aligned}
& \text{1. Existence:} \\
& F_{\text{existence}} = \mathbb{E}_Q[\ln Q(s) - \ln P(o,s)] \geq -\ln P(o) \\
& \text{2. Boundary:} \\
& F_{\text{markov}} = \mathbb{E}_Q[\ln Q(μ,b) - \ln P(μ,b,η)] \\
& \text{3. Dynamics:} \\
& \dot{F} = -\frac{\partial F}{\partial s}^T \Gamma \frac{\partial F}{\partial s} \leq 0
\end{aligned}
```

### Information Geometric Structure

The geometry of VFE manifolds:

```math
\begin{aligned}
& \text{Fisher Metric:} \\
& g_{ij} = \mathbb{E}_Q\left[\frac{\partial \ln Q}{\partial θ_i}\frac{\partial \ln Q}{\partial θ_j}\right] \\
& \text{Natural Gradient:} \\
& \dot{θ} = -g^{-1}\frac{\partial F}{\partial θ} \\
& \text{Geodesic Flow:} \\
& \ddot{θ}^i + \Gamma^i_{jk}\dot{θ}^j\dot{θ}^k = 0
\end{aligned}
```

### Statistical Physics Connection

The thermodynamic interpretation:

```math
\begin{aligned}
& F = U - TS \\
& U = \mathbb{E}_Q[E(s)] \\
& S = -\mathbb{E}_Q[\ln Q(s)] \\
& \beta = \frac{1}{T} = \text{precision}
\end{aligned}
```

## Advanced Implementation Frameworks

### 1. Hierarchical VFE Computer

```python
class HierarchicalVFE:
    """Computes hierarchical VFE across multiple scales"""
    def __init__(self, n_levels: int):
        self.n_levels = n_levels
        self.level_computers = [
            VFEComputer() for _ in range(n_levels)
        ]
        self.level_couplings = [
            LevelCoupling() for _ in range(n_levels-1)
        ]
        
    def compute_hierarchical_vfe(
        self,
        beliefs: List[Distribution],
        observations: List[np.ndarray]
    ) -> Tuple[float, dict]:
        """Compute VFE across hierarchy"""
        # Level-wise computation
        level_vfes = []
        for l in range(self.n_levels):
            vfe = self.level_computers[l].compute(
                beliefs[l], observations[l])
            level_vfes.append(vfe)
            
        # Coupling computation
        coupling_terms = []
        for l in range(self.n_levels-1):
            coupling = self.level_couplings[l].compute(
                beliefs[l], beliefs[l+1])
            coupling_terms.append(coupling)
            
        # Total VFE
        total_vfe = sum(level_vfes) + sum(coupling_terms)
        
        metrics = {
            'level_vfes': level_vfes,
            'coupling_terms': coupling_terms
        }
        
        return total_vfe, metrics
```

### 2. Information Geometric Optimizer

```python
class InfoGeometricVFE:
    """Optimizes VFE using information geometry"""
    def __init__(self):
        self.metric_computer = FisherMetric()
        self.connection_computer = LeviCivitaConnection()
        self.geodesic_solver = GeodesicFlow()
        
    def optimize_vfe(
        self,
        initial_beliefs: Distribution,
        observations: np.ndarray,
        n_steps: int
    ) -> Distribution:
        """Optimize VFE using natural gradient"""
        current_beliefs = initial_beliefs
        
        for _ in range(n_steps):
            # Compute Fisher metric
            metric = self.metric_computer.compute(
                current_beliefs)
                
            # Compute connection coefficients
            connection = self.connection_computer.compute(
                current_beliefs, metric)
                
            # Compute VFE gradient
            grad_vfe = self.compute_vfe_gradient(
                current_beliefs, observations)
                
            # Natural gradient step
            natural_grad = solve(metric, grad_vfe)
            
            # Geodesic update
            current_beliefs = self.geodesic_solver.step(
                current_beliefs,
                natural_grad,
                connection)
                
        return current_beliefs
```

### 3. Stochastic VFE Dynamics

```python
class StochasticVFE:
    """Implements stochastic VFE dynamics"""
    def __init__(self):
        self.sde_solver = StochasticDifferential()
        self.noise_generator = NoiseProcess()
        self.drift_computer = DriftField()
        
    def simulate_vfe_dynamics(
        self,
        initial_state: np.ndarray,
        time_span: float,
        dt: float,
        temperature: float
    ) -> np.ndarray:
        """Simulate stochastic VFE dynamics"""
        def drift(x, t):
            return -self.drift_computer.compute_field(x)
            
        def diffusion(x, t):
            return np.sqrt(2 * temperature)
            
        trajectory = self.sde_solver.solve(
            drift,
            diffusion,
            initial_state,
            time_span,
            dt)
            
        return trajectory
```

## Advanced Mathematical Bridges

### 1. Path Integral to VFE Bridge

The connection between path integral formulation and VFE:

```math
\begin{aligned}
& \text{Path Integral VFE:} \\
& F_{\text{PI}}[q] = \int \mathcal{D}[s(\tau)] q[s(\tau)] \left(\ln q[s(\tau)] - \ln p[s(\tau),o(\tau)]\right) \\
& \text{Discrete-Continuous Bridge:} \\
& F_{\text{bridge}} = \lim_{\Delta t \to 0} \sum_t F_t\Delta t = \int_0^T F(\tau)d\tau \\
& \text{Action-Value Relationship:} \\
& S[s(\tau)] = \beta \int_0^T \mathcal{L}(s,\dot{s},t)d\tau = -\ln p[s(\tau)]
\end{aligned}
```

### 2. Information Geometric Bridge

```math
\begin{aligned}
& \text{Fisher-Rao Metric:} \\
& g_{\mu\nu}(θ) = \mathbb{E}_{q_θ}\left[\frac{\partial \ln q_θ}{\partial θ^\mu}\frac{\partial \ln q_θ}{\partial θ^\nu}\right] \\
& \text{Natural Gradient Flow:} \\
& \dot{θ}^\mu = -g^{\mu\nu}(θ)\frac{\partial F}{\partial θ^\nu} \\
& \text{Wasserstein Gradient:} \\
& \nabla_W F = -\text{div}(\rho\nabla\frac{\delta F}{\delta \rho})
\end{aligned}
```

### 3. Quantum-Classical Bridge

```math
\begin{aligned}
& \text{Quantum VFE:} \\
& F_Q = \text{Tr}[\rho(\ln\rho - \ln\sigma)] + \beta\text{Tr}[\rho H] \\
& \text{Classical Limit:} \\
& \lim_{\hbar \to 0} F_Q = F_{\text{classical}} \\
& \text{Quantum Policy:} \\
& |\psi_\pi\rangle = \sum_a \sqrt{P(a|\pi)}|a\rangle
\end{aligned}
```

## Advanced Implementation Frameworks

### 1. Multi-Scale Integration Engine

```python
class MultiScaleIntegrationEngine:
    """Integrates VFE computation across scales"""
    def __init__(self):
        self.quantum_computer = QuantumVFEComputer()
        self.classical_computer = ClassicalVFEComputer()
        self.path_integral_computer = PathIntegralComputer()
        self.scale_bridge = ScaleBridgeComputer()
        
    def compute_multi_scale_vfe(self,
                               quantum_state: QuantumState,
                               classical_state: ClassicalState,
                               path_config: PathConfiguration,
                               scale_params: ScaleParameters) -> Dict[str, float]:
        """Compute VFE across multiple scales"""
        # Quantum scale computation
        quantum_vfe = self.quantum_computer.compute(
            quantum_state)
            
        # Classical scale computation
        classical_vfe = self.classical_computer.compute(
            classical_state)
            
        # Path integral computation
        path_vfe = self.path_integral_computer.compute(
            path_config)
            
        # Bridge computations
        quantum_classical_bridge = self.scale_bridge.bridge_quantum_classical(
            quantum_vfe, classical_vfe)
            
        classical_path_bridge = self.scale_bridge.bridge_classical_path(
            classical_vfe, path_vfe)
            
        return {
            'quantum': quantum_vfe,
            'classical': classical_vfe,
            'path': path_vfe,
            'q_c_bridge': quantum_classical_bridge,
            'c_p_bridge': classical_path_bridge
        }
```

### 2. Advanced Geometric Optimizer

```python
class GeometricOptimizer:
    """Geometric optimization for VFE"""
    def __init__(self):
        self.metric_computer = FisherRaoMetric()
        self.connection_computer = LeviCivitaConnection()
        self.parallel_transport = ParallelTransport()
        
    def optimize_geometric(self,
                         initial_state: Distribution,
                         target_state: Distribution,
                         n_steps: int = 100) -> Distribution:
        """Optimize using geometric methods"""
        current_state = initial_state
        
        for _ in range(n_steps):
            # Compute metric
            metric = self.metric_computer.compute(
                current_state)
                
            # Compute connection
            connection = self.connection_computer.compute(
                current_state, metric)
                
            # Compute geodesic
            geodesic = self.compute_geodesic(
                current_state, target_state, metric)
                
            # Parallel transport update
            current_state = self.parallel_transport.transport(
                current_state, geodesic, connection)
                
        return current_state
```

### 3. Stochastic Path Integral Computer

```python
class StochasticPathIntegralComputer:
    """Computes path integrals with stochastic dynamics"""
    def __init__(self):
        self.sde_solver = StochasticDifferential()
        self.path_sampler = PathSampler()
        self.action_computer = ActionComputer()
        
    def compute_stochastic_path_integral(self,
                                       initial_state: np.ndarray,
                                       final_state: np.ndarray,
                                       beta: float,
                                       n_samples: int = 1000) -> float:
        """Compute path integral using stochastic sampling"""
        paths = []
        actions = []
        
        for _ in range(n_samples):
            # Sample path
            path = self.path_sampler.sample(
                initial_state, final_state)
                
            # Compute stochastic action
            action = self.action_computer.compute_stochastic(
                path, beta)
                
            paths.append(path)
            actions.append(action)
            
        # Compute path integral
        Z = np.mean(np.exp(-np.array(actions)))
        F = -np.log(Z) / beta
        
        return F
```

## Advanced Theoretical Extensions

### 1. Relativistic VFE Framework

```math
\begin{aligned}
& \text{Covariant VFE:} \\
& F_{\text{cov}} = \int d^4x \sqrt{-g} \left(T^{\mu\nu}\nabla_\mu\nabla_\nu\ln\rho + V[\rho]\right) \\
& \text{Spacetime Action:} \\
& S_{\text{spacetime}} = \int d^4x \sqrt{-g}\mathcal{L}(\phi, \partial_\mu\phi) \\
& \text{Causal Structure:} \\
& \delta F_{\text{cov}}/\delta\rho = 0 \text{ on } J^+(x)
\end{aligned}
```

### 2. Quantum Field Theory Extension

```math
\begin{aligned}
& \text{Field VFE:} \\
& F_{\text{field}} = \int \mathcal{D}[\phi] \rho[\phi] \left(\ln\rho[\phi] - \ln P[\phi]\right) \\
& \text{Effective Action:} \\
& \Gamma[\phi] = -\ln \int \mathcal{D}[\chi] \exp(-S[\chi]) \\
& \text{Ward Identity:} \\
& \frac{\delta \Gamma}{\delta \phi} = \langle\frac{\delta S}{\delta \phi}\rangle
\end{aligned}
```

### 3. Topological Extensions

```math
\begin{aligned}
& \text{Topological VFE:} \\
& F_{\text{top}} = \oint_{\partial M} \omega + \int_M d\omega \\
& \text{Characteristic Classes:} \\
& c_1(F) = \frac{i}{2\pi}\text{tr}(F) \\
& \text{Index Theorem:} \\
& \text{index}(D) = \int_M \hat{A}(M)\text{ch}(E)
\end{aligned}
```

## Implementation Considerations

### 1. Advanced Numerical Methods
- Symplectic integration for Hamiltonian dynamics
- Adaptive Runge-Kutta methods
- Stochastic variational integrators

### 2. Parallel Computing Strategies
- GPU-accelerated path integral computation
- Distributed belief propagation
- Multi-scale parallel processing

### 3. Optimization Techniques
- Natural gradient methods
- Hamiltonian Monte Carlo
- Riemannian optimization

## References
- [[friston_2022]] - "The Free Energy Principle: A Unified Brain Theory?"
- [[parr_2022]] - "Active Inference: The Free Energy Principle in Mind, Brain, and Behavior"
- [[da_costa_2022]] - "The Mathematics of Active Inference"
- [[ramstead_2022]] - "Neural and cognitive architectures for active inference"

## See Also
- [[quantum_field_theory]]
- [[differential_geometry]]
- [[stochastic_processes]]
- [[information_geometry]]
- [[topological_quantum_field_theory]] 