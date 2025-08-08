---

title: Lagrangian Mechanics

type: concept

status: stable

created: 2024-02-12

tags:

  - mathematics

  - mechanics

  - physics

  - dynamics

semantic_relations:

  - type: foundation

    links:

      - [[calculus_of_variations]]

      - [[geometric_mechanics]]

  - type: relates

    links:

      - [[hamiltonian_mechanics]]

      - [[classical_field_theory]]

      - [[control_theory]]

---

# Lagrangian Mechanics

## Core Concepts

### Configuration Space

1. **Generalized Coordinates**

   ```math

   q = (q^1,...,q^n) \in Q

   ```

   where:

   - Q is configuration manifold

   - n is degrees of freedom

1. **Tangent Bundle**

   ```math

   TQ = \{(q,\dot{q}) : q \in Q, \dot{q} \in T_qQ\}

   ```

   where:

   - T_qQ is tangent space

   - q̇ is velocity

### Variational Principles

1. **Action Functional**

   ```math

   S[q] = \int_{t_1}^{t_2} L(q,\dot{q},t)dt

   ```

   where:

   - L is Lagrangian

   - q(t) is path

1. **Hamilton's Principle**

   ```math

   δS = 0 \iff \frac{d}{dt}\frac{∂L}{∂\dot{q}} - \frac{∂L}{∂q} = 0

   ```

   where:

   - δ is variation

   - L is Lagrangian

## Advanced Concepts

### Symmetries and Conservation

1. **Noether's Theorem**

   ```math

   \frac{d}{dt}\left(\frac{∂L}{∂\dot{q}^i}ξ^i\right) = 0

   ```

   where:

   - ξ is symmetry generator

   - L is Lagrangian

1. **Conserved Quantities**

   ```math

   p_i = \frac{∂L}{∂\dot{q}^i}

   ```

   where:

   - p_i are momenta

   - L is Lagrangian

### Constrained Systems

1. **Holonomic Constraints**

   ```math

   f(q^1,...,q^n) = 0

   ```

   where:

   - f is constraint function

   - q^i are coordinates

1. **D'Alembert's Principle**

   ```math

   \sum_{i=1}^n (F_i - m_i\ddot{q}_i)·δq_i = 0

   ```

   where:

   - F_i are forces

   - δq_i are virtual displacements

## Implementation

### Lagrangian Dynamics

```python

class LagrangianSystem:

    def __init__(self,

                 lagrangian: Callable,

                 dimension: int):

        """Initialize Lagrangian system.

        Args:

            lagrangian: System Lagrangian

            dimension: Configuration dimension

        """

        self.L = lagrangian

        self.dim = dimension

    def equations_of_motion(self,

                          state: np.ndarray,

                          t: float) -> np.ndarray:

        """Compute equations of motion.

        Args:

            state: State (q,q̇)

            t: Time

        Returns:

            derivatives: Time derivatives

        """

        # Split state

        q, v = np.split(state, 2)

        # Get accelerations

        a = self._solve_euler_lagrange(q, v, t)

        return np.concatenate([v, a])

    def energy(self,

              state: np.ndarray,

              t: float) -> float:

        """Compute total energy.

        Args:

            state: System state

            t: Time

        Returns:

            E: Total energy

        """

        # Split state

        q, v = np.split(state, 2)

        # Compute energy

        T = self._kinetic_energy(q, v)

        V = self._potential_energy(q)

        return T + V

```

### Constrained Dynamics

```python

class ConstrainedSystem:

    def __init__(self,

                 lagrangian: Callable,

                 constraints: List[Callable]):

        """Initialize constrained system.

        Args:

            lagrangian: System Lagrangian

            constraints: Constraint functions

        """

        self.L = lagrangian

        self.constraints = constraints

    def constraint_forces(self,

                        state: np.ndarray,

                        t: float) -> np.ndarray:

        """Compute constraint forces.

        Args:

            state: System state

            t: Time

        Returns:

            forces: Constraint forces

        """

        # Get constraint Jacobian

        J = self._constraint_jacobian(state)

        # Solve for multipliers

        lambda_ = self._solve_multipliers(state, J)

        return J.T @ lambda_

    def project_to_constraints(self,

                             state: np.ndarray) -> np.ndarray:

        """Project onto constraint manifold.

        Args:

            state: System state

        Returns:

            projected: Projected state

        """

        return self._project_state(state)

```

## Advanced Applications

### Variational Integrators

```python

class VariationalIntegrator:

    def __init__(self,

                 lagrangian: Callable,

                 step_size: float):

        """Initialize variational integrator.

        Args:

            lagrangian: System Lagrangian

            step_size: Time step

        """

        self.L = lagrangian

        self.h = step_size

    def discrete_euler_lagrange(self,

                              q_k: np.ndarray,

                              q_kp1: np.ndarray) -> np.ndarray:

        """Compute discrete Euler-Lagrange.

        Args:

            q_k: Current position

            q_kp1: Next position

        Returns:

            DEL: Discrete E-L equations

        """

        # Get discrete derivatives

        D1Ld = self._D1_discrete_lagrangian(q_k, q_kp1)

        D2Ld = self._D2_discrete_lagrangian(q_k, q_kp1)

        return D1Ld + D2Ld

    def step(self,

            q_km1: np.ndarray,

            q_k: np.ndarray) -> np.ndarray:

        """Take integration step.

        Args:

            q_km1: Previous position

            q_k: Current position

        Returns:

            q_kp1: Next position

        """

        # Solve DEL equations

        return self._solve_step(q_km1, q_k)

```

### Optimal Control

```python

class LagrangianControl:

    def __init__(self,

                 lagrangian: Callable,

                 control_forces: Callable):

        """Initialize controlled system.

        Args:

            lagrangian: System Lagrangian

            control_forces: Control inputs

        """

        self.L = lagrangian

        self.F = control_forces

    def optimal_trajectory(self,

                         initial: np.ndarray,

                         target: np.ndarray,

                         cost: Callable) -> Tuple[np.ndarray, np.ndarray]:

        """Compute optimal trajectory.

        Args:

            initial: Initial state

            target: Target state

            cost: Cost functional

        Returns:

            trajectory,controls: Optimal solution

        """

        # Set up optimization

        problem = self._setup_optimal_control(

            initial, target, cost

        )

        # Solve problem

        return self._solve_optimization(problem)

    def verify_optimality(self,

                         trajectory: np.ndarray,

                         controls: np.ndarray) -> bool:

        """Verify optimality conditions.

        Args:

            trajectory: System trajectory

            controls: Control inputs

        Returns:

            optimal: Whether solution is optimal

        """

        return self._check_pontryagin(trajectory, controls)

```

## Advanced Topics

### Field Theories

1. **Field Lagrangian**

   ```math

   L = \int_Ω \mathcal{L}(φ,∂_μφ)d^nx

   ```

   where:

   - φ is field

   - L is Lagrangian density

1. **Field Equations**

   ```math

   \frac{∂\mathcal{L}}{∂φ} - ∂_μ\frac{∂\mathcal{L}}{∂(∂_μφ)} = 0

   ```

   where:

   - L is Lagrangian density

   - ∂_μ is partial derivative

### Discrete Mechanics

1. **Discrete Lagrangian**

   ```math

   L_d(q_k,q_{k+1}) \approx \int_{t_k}^{t_{k+1}} L(q(t),\dot{q}(t))dt

   ```

   where:

   - L_d is discrete Lagrangian

   - L is continuous Lagrangian

1. **Discrete Noether**

   ```math

   \frac{∂L_d}{∂q_k}ξ_k + \frac{∂L_d}{∂q_{k+1}}ξ_{k+1} = 0

   ```

   where:

   - ξ is symmetry generator

   - L_d is discrete Lagrangian

## Future Directions

### Emerging Areas

1. **Geometric Integration**

   - Structure-Preserving Methods

   - Multisymplectic Integration

   - Discrete Field Theories

1. **Applications**

   - Robotics and Control

   - Quantum Field Theory

   - Numerical Relativity

### Open Problems

1. **Theoretical Challenges**

   - Singular Lagrangians

   - Discrete Reduction

   - Field Discretization

1. **Computational Challenges**

   - Constraint Handling

   - Energy Conservation

   - Symplectic Integration

## Related Topics

1. [[hamiltonian_mechanics|Hamiltonian Mechanics]]

1. [[geometric_mechanics|Geometric Mechanics]]

1. [[classical_field_theory|Classical Field Theory]]

1. [[control_theory|Control Theory]]

