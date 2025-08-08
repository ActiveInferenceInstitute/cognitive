---

type: mathematical_concept

id: expected_free_energy_update_001

created: 2024-02-05

modified: 2024-02-05

tags: [mathematics, active-inference, policy-selection, free-energy]

aliases: [EFE-update, policy-prior-update]

---

# Expected Free Energy Update

## Mathematical Definition

The Expected Free Energy (G) for each action a is:

$G(a) = \underbrace{\mathbb{E}_{Q(o,s|a)}[\ln Q(s|a) - \ln P(s|a)]}_{\text{epistemic value}} + \underbrace{\mathbb{E}_{Q(o|a)}[-\ln P(o)]}_{\text{pragmatic value}}$

where:

- Q(s|a) is predicted state distribution under action a

- P(s|a) is prior state distribution

- Q(o|a) is predicted observation distribution

- P(o) is preferred observation distribution (from C matrix)

## Policy Prior Update

The policy prior E is updated using:

$E_{t+1} = (1-\alpha)E_t + \alpha\sigma(-\gamma G)$

where:

- $E_t$ is current policy prior

- $\alpha$ is learning rate (0 for static prior)

- $\gamma$ is precision parameter

- $\sigma$ is softmax function

- G is vector of Expected Free Energies

## Implementation

```python

def update_policy_prior(

    A: np.ndarray,           # Observation model P(o|s)

    B: np.ndarray,           # Transition model P(s'|s,a)

    C: np.ndarray,           # Log preferences ln P(o)

    E: np.ndarray,           # Current policy prior P(a)

    beliefs: np.ndarray,     # Current state beliefs Q(s)

    alpha: float = 0.1,      # Learning rate

    gamma: float = 1.0       # Precision

) -> np.ndarray:

    """Update policy prior using Expected Free Energy.

    Args:

        A: Observation likelihood matrix [n_obs x n_states]

        B: State transition tensor [n_states x n_states x n_actions]

        C: Log preference vector [n_obs]

        E: Current policy prior [n_actions]

        beliefs: Current belief state [n_states]

        alpha: Learning rate (0 for static prior)

        gamma: Precision parameter

    Returns:

        Updated policy prior E [n_actions]

    """

    n_actions = B.shape[2]

    G = np.zeros(n_actions)

    for a in range(n_actions):

        # Predicted next state distribution

        Qs_a = B[:, :, a] @ beliefs

        # Predicted observation distribution

        Qo_a = A @ Qs_a

        # Epistemic value (state uncertainty)

        epistemic = compute_entropy(Qs_a)

        # Pragmatic value (preference satisfaction)

        pragmatic = -np.sum(Qo_a * C)  # Negative because C is log preferences

        # Total Expected Free Energy

        G[a] = epistemic + pragmatic

    # Compute new policy distribution using softmax

    E_new = softmax(-gamma * G)

    # Update with learning rate

    E_updated = (1 - alpha) * E + alpha * E_new

    return E_updated

```

## Usage

The function is used in the Active Inference loop to update action priors:

```python

# Initialize uniform action prior

E = np.ones(n_actions) / n_actions

# In simulation loop:

E = update_policy_prior(

    A=model.A,

    B=model.B, 

    C=model.C,

    E=model.E,

    beliefs=model.state.beliefs,

    alpha=model.config['inference']['learning_rate'],

    gamma=model.config['inference']['temperature']

)

```

## Properties

### Mathematical Properties

- [[probability_conservation]] - Output is valid probability distribution

- [[policy_convergence]] - Converges to optimal policy under right conditions

- [[learning_dynamics]] - Controlled by learning rate and precision

### Computational Properties

- [[numerical_stability]] - Uses log space for preferences

- [[computational_efficiency]] - Vectorized operations

- [[memory_usage]] - O(n_actions) space complexity

## Related Concepts

- [[free_energy_principle]]

- [[active_inference]]

- [[policy_selection]]

- [[belief_updating]]

## Advanced Theoretical Framework

### Convergence Analysis and Stability Theory

**Theorem** (EFE Update Convergence): Under mild regularity conditions, the policy prior update converges to the Gibbs distribution:

$$\lim_{t \to \infty} E_t = \frac{\exp(-\gamma G^*)}{\sum_a \exp(-\gamma G^*_a)}$$

where $G^*$ is the fixed-point EFE vector.

**Proof**: Using Lyapunov function analysis and the contraction mapping principle for the update operator.

```python

class AdvancedEFEUpdateAnalysis:

    """Advanced theoretical analysis of Expected Free Energy updates with rigorous mathematical foundation."""

    def __init__(self,

                 convergence_tolerance: float = 1e-8,

                 max_iterations: int = 10000,

                 stability_threshold: float = 1e-6):

        """Initialize advanced EFE update analysis framework.

        Args:

            convergence_tolerance: Tolerance for convergence detection

            max_iterations: Maximum iterations for convergence analysis

            stability_threshold: Threshold for stability analysis

        """

        self.convergence_tol = convergence_tolerance

        self.max_iterations = max_iterations

        self.stability_threshold = stability_threshold

        # Analysis storage

        self.convergence_history = []

        self.stability_metrics = []

        self.lyapunov_values = []

    def convergence_analysis(self,

                           A: np.ndarray,

                           B: np.ndarray,

                           C: np.ndarray,

                           initial_beliefs: np.ndarray,

                           alpha: float,

                           gamma: float) -> Dict[str, Any]:

        """Comprehensive convergence analysis of EFE updates.

        Analyzes convergence properties including rate, stability,

        and basin of attraction for the policy prior update dynamics.

        Args:

            A: Observation model matrix

            B: Transition model tensor

            C: Log preference vector

            initial_beliefs: Initial belief state

            alpha: Learning rate

            gamma: Precision parameter

        Returns:

            Complete convergence analysis including theoretical bounds

        """

        # Initialize policy prior

        n_actions = B.shape[2]

        E = np.ones(n_actions) / n_actions

        # Track convergence

        E_history = [E.copy()]

        G_history = []

        lyapunov_history = []

        for t in range(self.max_iterations):

            # Compute EFE for current beliefs

            G = self._compute_efe_vector(A, B, C, initial_beliefs)

            G_history.append(G.copy())

            # Update policy prior

            E_new_target = self._softmax(-gamma * G)

            E_new = (1 - alpha) * E + alpha * E_new_target

            # Compute Lyapunov function value

            lyapunov_value = self._compute_lyapunov_function(E, E_new_target, gamma)

            lyapunov_history.append(lyapunov_value)

            # Check convergence

            convergence_error = np.linalg.norm(E_new - E)

            if convergence_error < self.convergence_tol:

                break

            E = E_new

            E_history.append(E.copy())

        # Analyze convergence properties

        convergence_rate = self._estimate_convergence_rate(E_history)

        stability_analysis = self._analyze_stability(E_history, lyapunov_history)

        fixed_point_analysis = self._analyze_fixed_points(E_history[-1], G_history[-1], gamma)

        # Theoretical bounds verification

        theoretical_bounds = self._verify_theoretical_bounds(

            E_history, G_history, alpha, gamma

        )

        return {

            'converged': t < self.max_iterations - 1,

            'convergence_iterations': t + 1,

            'convergence_rate': convergence_rate,

            'final_policy': E_history[-1],

            'final_efe': G_history[-1],

            'stability_analysis': stability_analysis,

            'fixed_point_analysis': fixed_point_analysis,

            'theoretical_bounds': theoretical_bounds,

            'policy_history': np.array(E_history),

            'efe_history': np.array(G_history),

            'lyapunov_history': np.array(lyapunov_history)

        }

    def stability_analysis(self,

                         A: np.ndarray,

                         B: np.ndarray,

                         C: np.ndarray,

                         equilibrium_point: np.ndarray,

                         alpha: float,

                         gamma: float) -> Dict[str, Any]:

        """Analyze stability of equilibrium points in EFE update dynamics.

        Uses linear stability analysis and Lyapunov methods to determine

        stability properties of fixed points in the policy update dynamics.

        Args:

            A, B, C: Generative model matrices

            equilibrium_point: Equilibrium policy distribution

            alpha: Learning rate

            gamma: Precision parameter

        Returns:

            Comprehensive stability analysis

        """

        # Compute Jacobian matrix at equilibrium

        jacobian = self._compute_jacobian_at_equilibrium(

            A, B, C, equilibrium_point, alpha, gamma

        )

        # Eigenvalue analysis

        eigenvalues, eigenvectors = np.linalg.eig(jacobian)

        # Stability classification

        stability_type = self._classify_stability(eigenvalues)

        # Lyapunov exponents

        lyapunov_exponents = self._compute_lyapunov_exponents(eigenvalues)

        # Basin of attraction estimation

        basin_analysis = self._estimate_basin_of_attraction(

            A, B, C, equilibrium_point, alpha, gamma

        )

        # Bifurcation analysis

        bifurcation_analysis = self._analyze_bifurcations(

            A, B, C, alpha, gamma

        )

        return {

            'jacobian_matrix': jacobian,

            'eigenvalues': eigenvalues,

            'eigenvectors': eigenvectors,

            'stability_type': stability_type,

            'lyapunov_exponents': lyapunov_exponents,

            'basin_of_attraction': basin_analysis,

            'bifurcation_analysis': bifurcation_analysis,

            'linear_stability': np.all(np.real(eigenvalues) < 0)

        }

    def learning_rate_optimization(self,

                                 A: np.ndarray,

                                 B: np.ndarray,

                                 C: np.ndarray,

                                 initial_beliefs: np.ndarray,

                                 gamma: float,

                                 alpha_range: Tuple[float, float] = (0.001, 0.5)) -> Dict[str, Any]:

        """Optimize learning rate for fastest stable convergence.

        Analyzes the trade-off between convergence speed and stability

        to find optimal learning rate parameters.

        Args:

            A, B, C: Generative model matrices

            initial_beliefs: Initial belief state

            gamma: Precision parameter

            alpha_range: Range of learning rates to test

        Returns:

            Learning rate optimization analysis

        """

        alpha_values = np.linspace(alpha_range[0], alpha_range[1], 50)

        convergence_rates = []

        stability_margins = []

        final_errors = []

        for alpha in alpha_values:

            # Analyze convergence for this learning rate

            analysis = self.convergence_analysis(

                A, B, C, initial_beliefs, alpha, gamma

            )

            convergence_rates.append(analysis['convergence_rate'])

            stability_margins.append(

                analysis['stability_analysis']['linear_stability']

            )

            final_errors.append(

                np.linalg.norm(analysis['final_policy'] - 

                             analysis['theoretical_bounds']['optimal_policy'])

            )

        # Find optimal learning rate

        # Balance between speed and stability

        combined_score = (

            np.array(convergence_rates) * np.array(stability_margins) / 

            (1 + np.array(final_errors))

        )

        optimal_idx = np.argmax(combined_score)

        optimal_alpha = alpha_values[optimal_idx]

        return {

            'optimal_learning_rate': optimal_alpha,

            'alpha_values': alpha_values,

            'convergence_rates': convergence_rates,

            'stability_margins': stability_margins,

            'final_errors': final_errors,

            'combined_scores': combined_score,

            'optimization_surface': {

                'alpha': alpha_values,

                'performance': combined_score

            }

        }

    def robustness_analysis(self,

                          A: np.ndarray,

                          B: np.ndarray,

                          C: np.ndarray,

                          alpha: float,

                          gamma: float,

                          noise_levels: List[float] = [0.01, 0.05, 0.1, 0.2]) -> Dict[str, Any]:

        """Analyze robustness of EFE updates to model uncertainties and noise.

        Tests how sensitive the convergence and stability properties are

        to perturbations in the generative model and noise in updates.

        Args:

            A, B, C: Generative model matrices

            alpha: Learning rate

            gamma: Precision parameter

            noise_levels: Levels of noise to test

        Returns:

            Robustness analysis results

        """

        robustness_metrics = {}

        for noise_level in noise_levels:

            # Add noise to model matrices

            A_noisy = A + noise_level * np.random.randn(*A.shape)

            B_noisy = B + noise_level * np.random.randn(*B.shape)

            C_noisy = C + noise_level * np.random.randn(*C.shape)

            # Normalize A and B to maintain probability constraints

            A_noisy = np.abs(A_noisy)

            A_noisy /= np.sum(A_noisy, axis=0, keepdims=True)

            B_noisy = np.abs(B_noisy)

            B_noisy /= np.sum(B_noisy, axis=0, keepdims=True)

            # Analyze convergence with noisy model

            initial_beliefs = np.random.rand(A.shape[1])

            initial_beliefs /= np.sum(initial_beliefs)

            try:

                noisy_analysis = self.convergence_analysis(

                    A_noisy, B_noisy, C_noisy, initial_beliefs, alpha, gamma

                )

                robustness_metrics[f'noise_{noise_level}'] = {

                    'converged': noisy_analysis['converged'],

                    'convergence_rate': noisy_analysis['convergence_rate'],

                    'stability': noisy_analysis['stability_analysis']['linear_stability'],

                    'final_policy_deviation': np.linalg.norm(

                        noisy_analysis['final_policy'] - 

                        self._compute_clean_final_policy(A, B, C, initial_beliefs, alpha, gamma)

                    )

                }

            except Exception as e:

                robustness_metrics[f'noise_{noise_level}'] = {

                    'converged': False,

                    'error': str(e)

                }

        # Compute robustness summary statistics

        successful_analyses = [

            metrics for metrics in robustness_metrics.values() 

            if isinstance(metrics, dict) and metrics.get('converged', False)

        ]

        if successful_analyses:

            robustness_summary = {

                'mean_convergence_rate': np.mean([

                    m['convergence_rate'] for m in successful_analyses

                ]),

                'stability_success_rate': np.mean([

                    m['stability'] for m in successful_analyses

                ]),

                'mean_policy_deviation': np.mean([

                    m['final_policy_deviation'] for m in successful_analyses

                ])

            }

        else:

            robustness_summary = {'error': 'No successful analyses'}

        return {

            'noise_analysis': robustness_metrics,

            'robustness_summary': robustness_summary,

            'noise_levels_tested': noise_levels

        }

    def _compute_efe_vector(self,

                          A: np.ndarray,

                          B: np.ndarray,

                          C: np.ndarray,

                          beliefs: np.ndarray) -> np.ndarray:

        """Compute Expected Free Energy vector for all actions."""

        n_actions = B.shape[2]

        G = np.zeros(n_actions)

        for a in range(n_actions):

            # Predicted next state distribution

            Qs_a = B[:, :, a] @ beliefs

            # Predicted observation distribution

            Qo_a = A @ Qs_a

            # Epistemic value (negative entropy)

            epistemic = -self._compute_entropy(Qs_a)

            # Pragmatic value (negative expected log preference)

            pragmatic = -np.sum(Qo_a * C)

            G[a] = epistemic + pragmatic

        return G

    def _compute_entropy(self, distribution: np.ndarray) -> float:

        """Compute entropy of probability distribution."""

        # Avoid log(0) by adding small epsilon

        eps = 1e-16

        return -np.sum(distribution * np.log(distribution + eps))

    def _softmax(self, x: np.ndarray) -> np.ndarray:

        """Numerically stable softmax function."""

        exp_x = np.exp(x - np.max(x))  # Numerical stability

        return exp_x / np.sum(exp_x)

    def _compute_lyapunov_function(self,

                                 E_current: np.ndarray,

                                 E_target: np.ndarray,

                                 gamma: float) -> float:

        """Compute Lyapunov function for convergence analysis."""

        # Use KL divergence as Lyapunov function

        eps = 1e-16

        return np.sum(E_current * np.log((E_current + eps) / (E_target + eps)))

    def _estimate_convergence_rate(self, E_history: List[np.ndarray]) -> float:

        """Estimate exponential convergence rate from policy history."""

        if len(E_history) < 3:

            return 0.0

        # Compute sequence of distances to final point

        final_E = E_history[-1]

        distances = [np.linalg.norm(E - final_E) for E in E_history[:-1]]

        # Fit exponential decay model

        if len(distances) < 2 or distances[0] == 0:

            return 0.0

        # Linear regression on log(distance) vs time

        log_distances = np.log(np.maximum(distances, 1e-16))

        times = np.arange(len(log_distances))

        if len(times) > 1:

            rate = np.polyfit(times, log_distances, 1)[0]

            return -rate  # Negative because we want positive convergence rate

        else:

            return 0.0

    def _analyze_stability(self,

                         E_history: List[np.ndarray],

                         lyapunov_history: List[float]) -> Dict[str, Any]:

        """Analyze stability from simulation history."""

        # Check Lyapunov function monotonicity

        lyapunov_decreasing = np.all(np.diff(lyapunov_history) <= self.stability_threshold)

        # Check policy convergence stability

        if len(E_history) > 10:

            recent_variation = np.std([

                np.linalg.norm(E_history[i] - E_history[i-1])

                for i in range(-10, 0)

            ])

            stable_convergence = recent_variation < self.stability_threshold

        else:

            stable_convergence = False

        return {

            'lyapunov_decreasing': lyapunov_decreasing,

            'stable_convergence': stable_convergence,

            'final_variation': recent_variation if len(E_history) > 10 else float('inf')

        }

    def _analyze_fixed_points(self,

                            final_policy: np.ndarray,

                            final_efe: np.ndarray,

                            gamma: float) -> Dict[str, Any]:

        """Analyze fixed point properties."""

        # Check if policy is consistent with EFE

        expected_policy = self._softmax(-gamma * final_efe)

        consistency_error = np.linalg.norm(final_policy - expected_policy)

        # Entropy of final policy

        policy_entropy = self._compute_entropy(final_policy)

        return {

            'consistency_error': consistency_error,

            'policy_entropy': policy_entropy,

            'is_deterministic': policy_entropy < 0.1,

            'expected_policy': expected_policy

        }

    def _verify_theoretical_bounds(self,

                                 E_history: List[np.ndarray],

                                 G_history: List[np.ndarray],

                                 alpha: float,

                                 gamma: float) -> Dict[str, Any]:

        """Verify theoretical convergence bounds."""

        # Theoretical optimal policy

        if G_history:

            optimal_policy = self._softmax(-gamma * G_history[-1])

        else:

            optimal_policy = np.ones(len(E_history[0])) / len(E_history[0])

        # Check convergence bounds

        final_error = np.linalg.norm(E_history[-1] - optimal_policy)

        # Theoretical bound (simplified)

        theoretical_bound = alpha / (2 - alpha)  # Rough approximation

        return {

            'optimal_policy': optimal_policy,

            'final_error': final_error,

            'theoretical_bound': theoretical_bound,

            'bound_satisfied': final_error <= theoretical_bound

        }

# Example validation and usage

def validate_advanced_efe_update_analysis():

    """Validate advanced EFE update analysis framework."""

    # Create test model

    n_states, n_obs, n_actions = 4, 4, 3

    A = np.random.rand(n_obs, n_states)

    A /= np.sum(A, axis=0, keepdims=True)

    B = np.random.rand(n_states, n_states, n_actions)

    B /= np.sum(B, axis=0, keepdims=True)

    C = np.random.randn(n_obs)

    initial_beliefs = np.random.rand(n_states)

    initial_beliefs /= np.sum(initial_beliefs)

    alpha = 0.1

    gamma = 2.0

    # Initialize analyzer

    analyzer = AdvancedEFEUpdateAnalysis()

    # Perform convergence analysis

    convergence_result = analyzer.convergence_analysis(

        A, B, C, initial_beliefs, alpha, gamma

    )

    print("Advanced EFE Update Analysis Results:")

    print(f"Converged: {convergence_result['converged']}")

    print(f"Convergence iterations: {convergence_result['convergence_iterations']}")

    print(f"Convergence rate: {convergence_result['convergence_rate']:.6f}")

    print(f"Linear stability: {convergence_result['stability_analysis']['linear_stability']}")

    # Learning rate optimization

    lr_optimization = analyzer.learning_rate_optimization(

        A, B, C, initial_beliefs, gamma

    )

    print(f"Optimal learning rate: {lr_optimization['optimal_learning_rate']:.4f}")

    # Robustness analysis

    robustness = analyzer.robustness_analysis(A, B, C, alpha, gamma)

    print(f"Robustness analysis completed for {len(robustness['noise_levels_tested'])} noise levels")

if __name__ == "__main__":

    validate_advanced_efe_update_analysis()

```

### Information-Theoretic Learning Bounds

**Theorem** (PAC-Bayes Bound for EFE Learning): With probability at least $1-\delta$, the generalization error of the learned policy satisfies:

$$\mathbb{E}_{D}[G(\pi)] \leq \hat{\mathbb{E}}[G(\pi)] + \sqrt{\frac{D_{KL}[Q(\pi) \| P(\pi)] + \ln(1/\delta)}{2n}}$$

where $D$ is the data distribution, $Q(\pi)$ is the learned policy distribution, and $P(\pi)$ is the prior.

### Stochastic Approximation Theory

**Theorem** (Robbins-Monro for EFE Updates): The EFE update sequence $\{E_t\}$ with learning rate $\alpha_t$ converges almost surely to the optimal policy if:

1. $\sum_{t=1}^{\infty} \alpha_t = \infty$

1. $\sum_{t=1}^{\infty} \alpha_t^2 < \infty$

1. The EFE function satisfies Lipschitz conditions

This provides theoretical foundation for adaptive learning rate schedules in EFE updates.

## References

- [[friston_2017]] - Mathematical foundations

- [[da_costa_2020]] - Active Inference implementation

- [[parr_2019]] - Policy learning

- [[robbins_monro_1951]] - Stochastic approximation theory

- [[mcallester_1999]] - PAC-Bayes bounds

- [[borkar_2008]] - Stochastic approximation algorithms

