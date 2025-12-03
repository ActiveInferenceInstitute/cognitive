---

title: Hamiltonian Mechanics

type: mathematical_concept

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

      - [[symplectic_geometry]]

      - [[geometric_mechanics]]

  - type: relates

    links:

      - [[lagrangian_mechanics]]

      - [[quantum_mechanics]]

      - [[integrable_systems]]

---

# Hamiltonian Mechanics

## Core Concepts

### Phase Space

1. **State Space**

   ```math

   (q,p) \in T^*Q \cong \mathbb{R}^{2n}

   ```

   where:

   - q are positions

   - p are momenta

   - T^*Q is cotangent bundle

1. **Hamiltonian Function**

   ```math

   H(q,p) = \sum_{i=1}^n \frac{p_i^2}{2m_i} + V(q)

   ```

   where:

   - m_i are masses

   - V is potential energy

### Hamilton's Equations

1. **Canonical Equations**

   ```math

   \dot{q}_i = \frac{∂H}{∂p_i}, \quad \dot{p}_i = -\frac{∂H}{∂q_i}

   ```

   where:

   - H is Hamiltonian

   - (q_i,p_i) are conjugate pairs

1. **Poisson Bracket**

   ```math

   \{F,G\} = \sum_{i=1}^n \frac{∂F}{∂q_i}\frac{∂G}{∂p_i} - \frac{∂F}{∂p_i}\frac{∂G}{∂q_i}

   ```

   where:

   - F,G are observables

   - {,} is Poisson bracket

## Advanced Concepts

### Canonical Transformations

1. **Generating Functions**

   ```math

   F_1(q,Q,t): \quad p = \frac{∂F_1}{∂q}, \quad P = -\frac{∂F_1}{∂Q}

   ```

   where:

   - (q,p) are old coordinates

   - (Q,P) are new coordinates

1. **Symplectic Condition**

   ```math

   J^T\Omega J = \Omega, \quad \Omega = \begin{pmatrix} 0 & I \\ -I & 0 \end{pmatrix}

   ```

   where:

   - J is Jacobian matrix

   - Ω is symplectic form

### Action-Angle Variables

1. **Action Variables**

   ```math

   I_i = \frac{1}{2π}\oint p_i dq_i

   ```

   where:

   - p_i dq_i is action form

   - ∮ is loop integral

1. **Angle Variables**

   ```math

   θ_i = \frac{∂S}{∂I_i}

   ```

   where:

   - S is generating function

   - I_i are action variables

## Implementation

### Hamiltonian Dynamics

```python

class HamiltonianSystem:

    def __init__(self,

                 hamiltonian: Callable,

                 dimension: int):

        """Initialize Hamiltonian system.

        Args:

            hamiltonian: System Hamiltonian

            dimension: Phase space dimension

        """

        self.H = hamiltonian

        self.dim = dimension

    def equations_of_motion(self,

                          state: np.ndarray,

                          t: float) -> np.ndarray:

        """Compute equations of motion.

        Args:

            state: Phase space point (q,p)

            t: Time

        Returns:

            derivatives: Time derivatives

        """

        # Split state

        q, p = np.split(state, 2)

        # Compute derivatives

        dq = self._dH_dp(q, p)

        dp = -self._dH_dq(q, p)

        return np.concatenate([dq, dp])

    def flow(self,

            initial_state: np.ndarray,

            time_span: Tuple[float, float],

            n_steps: int) -> np.ndarray:

        """Compute Hamiltonian flow.

        Args:

            initial_state: Initial conditions

            time_span: Time interval

            n_steps: Number of steps

        Returns:

            trajectory: System trajectory

        """

        # Set up integrator

        integrator = self._setup_integrator()

        # Integrate system

        return integrator.solve(

            self.equations_of_motion,

            time_span,

            initial_state,

            n_steps

        )

```

### Canonical Transformations

```python

class CanonicalTransformation:

    def __init__(self,

                 generating_function: Callable,

                 type: int = 1):

        """Initialize canonical transformation.

        Args:

            generating_function: Generator

            type: Type (1-4)

        """

        self.F = generating_function

        self.type = type

    def transform(self,

                 old_coords: np.ndarray) -> np.ndarray:

        """Apply canonical transformation.

        Args:

            old_coords: Old coordinates (q,p)

        Returns:

            new_coords: New coordinates (Q,P)

        """

        if self.type == 1:

            return self._transform_type1(old_coords)

        elif self.type == 2:

            return self._transform_type2(old_coords)

        else:

            raise ValueError(f"Type {self.type} not implemented")

    def verify_canonical(self,

                       jacobian: np.ndarray) -> bool:

        """Verify canonical condition.

        Args:

            jacobian: Transformation Jacobian

        Returns:

            canonical: Whether transformation is canonical

        """

        # Get symplectic form

        omega = self._get_symplectic_form()

        # Check condition

        return self._check_symplectic_condition(jacobian, omega)

```

## Advanced Applications

### Hamilton-Jacobi Theory

```python

class HamiltonJacobi:

    def __init__(self,

                 hamiltonian: Callable):

        """Initialize Hamilton-Jacobi solver.

        Args:

            hamiltonian: System Hamiltonian

        """

        self.H = hamiltonian

    def solve_equation(self,

                      initial_condition: Callable) -> Callable:

        """Solve Hamilton-Jacobi equation.

        Args:

            initial_condition: Initial S

        Returns:

            solution: Principal function

        """

        # Set up PDE

        pde = self._setup_hj_equation()

        # Solve equation

        return self._solve_pde(pde, initial_condition)

    def action_angle_variables(self,

                             solution: Callable) -> Tuple[np.ndarray, np.ndarray]:

        """Compute action-angle variables.

        Args:

            solution: HJ solution

        Returns:

            actions,angles: Action-angle variables

        """

        # Compute actions

        I = self._compute_actions(solution)

        # Compute angles

        theta = self._compute_angles(solution, I)

        return I, theta

```

### Perturbation Theory

```python

class HamiltonianPerturbation:

    def __init__(self,

                 unperturbed: Callable,

                 perturbation: Callable,

                 epsilon: float):

        """Initialize perturbation theory.

        Args:

            unperturbed: H_0

            perturbation: H_1

            epsilon: Small parameter

        """

        self.H0 = unperturbed

        self.H1 = perturbation

        self.eps = epsilon

    def normal_form(self,

                   order: int) -> Callable:

        """Compute normal form.

        Args:

            order: Perturbation order

        Returns:

            H_norm: Normal form

        """

        # Get generating function

        W = self._compute_generator(order)

        # Transform Hamiltonian

        return self._transform_hamiltonian(W)

    def resonance_analysis(self,

                          frequencies: np.ndarray) -> List[Tuple[int]]:

        """Find resonances.

        Args:

            frequencies: Unperturbed frequencies

        Returns:

            resonances: Resonant combinations

        """

        return self._find_resonances(frequencies)

```

## Advanced Topics

### KAM Theory

1. **KAM Tori**

   ```math

   ω · k \neq 0 \text{ for all } k \in \mathbb{Z}^n\setminus\{0\}

   ```

   where:

   - ω are frequencies

   - k are integer vectors

1. **Diophantine Condition**

   ```math

   |ω · k| ≥ \frac{γ}{|k|^τ}

   ```

   where:

   - γ > 0 is constant

   - τ > n-1 is exponent

### Integrable Systems

1. **Liouville-Arnold Theorem**

   ```math

   \{F_i,F_j\} = 0, \quad dF_1 ∧ ... ∧ dF_n \neq 0

   ```

   where:

   - F_i are integrals

   - {,} is Poisson bracket

1. **Action Variables**

   ```math

   I_i = \frac{1}{2π}\oint_{γ_i} p \cdot dq

   ```

   where:

   - γ_i are basis cycles

   - p·dq is action form

## Future Directions

### Emerging Areas

1. **Quantum Hamiltonian Systems**

   - Quantum Integrability

   - Quantum KAM Theory

   - Quantum Chaos

1. **Applications**

   - Celestial Mechanics

   - Plasma Physics

   - Chemical Dynamics

### Open Problems

1. **Theoretical Challenges**

   - Arnol'd Diffusion

   - Nekhoroshev Stability

   - Quantum-Classical Correspondence

1. **Computational Challenges**

   - Long-time Integration

   - Resonance Detection

   - Chaos Prediction

## Related Topics

1. [[lagrangian_mechanics|Lagrangian Mechanics]]

1. [[geometric_mechanics|Geometric Mechanics]]

1. [[quantum_mechanics|Quantum Mechanics]]

1. [[integrable_systems|Integrable Systems]]

