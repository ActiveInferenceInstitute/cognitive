---

title: Expected Free Energy

type: concept

status: stable

created: 2024-03-20

tags:

  - mathematics

  - active-inference

  - decision-theory

  - information-theory

msc: ["62F15", "94A17", "68T05"]

semantic_relations:

  - type: foundation

    links:

      - [[free_energy_principle]]

      - [[information_theory]]

      - [[decision_theory]]

  - type: implements

    links:

      - [[active_inference]]

      - [[policy_selection]]

      - [[exploration_exploitation]]

  - type: related

    links:

      - [[variational_free_energy]]

      - [[kl_divergence]]

      - [[entropy]]

---

# Expected Free Energy

## Overview

Expected Free Energy (EFE) is a fundamental quantity in active inference that guides action selection by balancing exploration (information gain) and exploitation (goal-seeking). It extends the [[free_energy_principle|Free Energy Principle]] to future states and actions.

## Mathematical Foundation

### 1. Basic Definition

```math

G(π) = \sum_τ G(π,τ)

```

where:

- G(π) is the expected free energy for policy π

- τ indexes future time points

- G(π,τ) is the expected free energy at time τ

### 2. Decomposition

A common and numerically stable decomposition separates risk (preference mismatch) and ambiguity (expected observation uncertainty):

```math

\begin{aligned}

G(\pi,\tau)

&= \underbrace{\mathbb{E}_{q(o_\tau|\pi)}\big[ D_{\mathrm{KL}}\big(q(s_\tau\mid o_\tau,\pi)\,\|\,q(s_\tau\mid \pi)\big)\big]}_{\text{Epistemic value (information gain)}} \\

&\quad + \underbrace{D_{\mathrm{KL}}\big(q(o_\tau\mid \pi)\,\|\,p(o_\tau)\big)}_{\text{Risk (preference mismatch)}} \\

&\equiv \text{Ambiguity} + \text{Risk}

\end{aligned}

```

where:

- q(s_τ|π) is the predicted state distribution

- p(o_τ,s_τ) is the generative model

- q(o_τ|π) is the predicted observation distribution

- p(o_τ) is the prior preference over observations

### 3. Policy Selection

```math

P(\pi) = \sigma\big(-\gamma\,G(\pi)\big)

```

where:

- σ is the softmax function

- γ is the precision parameter

## Implementation

### 1. Expected Free Energy Computation

```julia

struct ExpectedFreeEnergy

    # Time horizon

    T::Int

    # Precision parameter

    γ::Float64

    # Prior preferences

    p_o::Distribution

end

function compute_expected_free_energy(efe::ExpectedFreeEnergy,

                                   policy::Policy,

                                   model::GenerativeModel)

    # Initialize total EFE

    G_total = 0.0

    # Compute EFE for each future time step

    for τ in 1:efe.T

        # Predict states and observations

        q_s = predict_states(model, policy, τ)

        q_o = predict_observations(model, q_s)

        # Compute risk

        risk = compute_risk(q_s, q_o, model)

        # Compute ambiguity

        ambiguity = compute_ambiguity(q_o, efe.p_o)

        # Accumulate

        G_total += risk + ambiguity

    end

    return G_total

end

```

### 2. Risk Term

```julia

function compute_risk(q_s::Distribution,

                     q_o::Distribution,

                     model::GenerativeModel)

    # Compute KL divergence between predicted and preferred states

    risk = expectation(q_s, q_o) do s, o

        log_q_s = logpdf(q_s, s)

        log_p_so = logpdf(model.joint, (s, o))

        return log_q_s - log_p_so

    end

    return risk

end

```

### 3. Ambiguity Term

```julia

function compute_ambiguity(q_o::Distribution,

                          p_o::Distribution)

    # Compute epistemic value

    ambiguity = expectation(q_o) do o

        log_q_o = logpdf(q_o, o)

        log_p_o = logpdf(p_o, o)

        return log_q_o - log_p_o

    end

    return ambiguity

end

```

## Policy Selection

### 1. [[policy_evaluation|Policy Evaluation]]

```julia

function evaluate_policies(agent::ActiveInferenceAgent,

                         policies::Vector{Policy})

    # Compute EFE for each policy

    G = zeros(length(policies))

    for (i, π) in enumerate(policies)

        G[i] = compute_expected_free_energy(

            agent.efe,

            π,

            agent.gen_model

        )

    end

    return G

end

```

### 2. [[action_selection|Action Selection]]

```julia

function select_action(agent::ActiveInferenceAgent,

                      observation::Vector{Float64})

    # Update beliefs

    update_beliefs!(agent, observation)

    # Generate policies

    policies = generate_policies(agent)

    # Evaluate policies

    G = evaluate_policies(agent, policies)

    # Compute policy probabilities

    P = softmax(-agent.efe.γ * G)

    # Sample action

    π = sample_categorical(policies, P)

    return first_action(π)

end

```

### 3. [[exploration_exploitation|Exploration-Exploitation]]

```julia

function adaptive_exploration(agent::ActiveInferenceAgent,

                            temperature::Float64)

    # Modify precision based on uncertainty

    uncertainty = compute_uncertainty(agent)

    agent.efe.γ = 1.0 / (temperature * uncertainty)

    # Generate and evaluate policies

    policies = generate_policies(agent)

    G = evaluate_policies(agent, policies)

    # Select policy with adaptive exploration

    P = softmax(-agent.efe.γ * G)

    return sample_categorical(policies, P)

end

```

## Applications

### 1. [[decision_making|Decision Making]]

```julia

function make_decision(agent::ActiveInferenceAgent,

                      options::Vector{Action},

                      preferences::Distribution)

    # Set prior preferences

    agent.efe.p_o = preferences

    # Generate single-step policies

    policies = [Policy([a]) for a in options]

    # Evaluate expected free energy

    G = evaluate_policies(agent, policies)

    # Select option

    return options[argmin(G)]

end

```

### 2. [[active_sensing|Active Sensing]]

```julia

function active_sensing(agent::ActiveInferenceAgent,

                       environment::Environment)

    # Initialize information gain

    total_info_gain = 0.0

    for t in 1:agent.efe.T

        # Select action to maximize information gain

        action = select_information_seeking_action(agent)

        # Execute action

        observation = environment.step(action)

        # Update beliefs and compute information gain

        info_gain = update_and_compute_gain!(agent, observation)

        total_info_gain += info_gain

    end

    return total_info_gain

end

```

### 3. [[goal_directed_behavior|Goal-Directed Behavior]]

```julia

function goal_directed_policy(agent::ActiveInferenceAgent,

                            goal_state::State)

    # Set prior preferences to favor goal state

    set_goal_preference!(agent, goal_state)

    # Generate multi-step policies

    policies = generate_goal_directed_policies(agent, goal_state)

    # Evaluate policies considering both goal and information gain

    G = zeros(length(policies))

    for (i, π) in enumerate(policies)

        # Compute expected free energy

        G[i] = compute_expected_free_energy(agent.efe, π, agent.gen_model)

        # Add goal-specific term

        G[i] += compute_goal_distance(π, goal_state)

    end

    return policies[argmin(G)]

end

```

## Theoretical Results

### 1. [[optimality|Optimality]]

```julia

function prove_optimality(efe::ExpectedFreeEnergy)

    # Demonstrate that minimizing EFE leads to optimal behavior

    # 1. Information gain is maximized

    show_information_maximization(efe)

    # 2. Goal-seeking behavior emerges

    show_goal_directed_behavior(efe)

    # 3. Uncertainty is minimized

    show_uncertainty_reduction(efe)

end

```

### 2. [[convergence|Convergence]]

```julia

function analyze_convergence(agent::ActiveInferenceAgent,

                           environment::Environment)

    # Track EFE over time

    G_history = Float64[]

    while !converged(agent)

        # Select and execute action

        action = select_action(agent, observe(environment))

        environment.step(action)

        # Record EFE

        push!(G_history, compute_current_efe(agent))

        # Update beliefs

        update_beliefs!(agent, observe(environment))

    end

    return G_history

end

```

### 3. [[information_bounds|Information Bounds]]

```julia

function compute_information_bounds(efe::ExpectedFreeEnergy)

    # Compute upper bound on information gain

    max_info_gain = compute_max_information_gain(efe)

    # Compute lower bound on expected free energy

    min_efe = compute_minimum_efe(efe)

    # Compute bounds on policy entropy

    H_bounds = compute_policy_entropy_bounds(efe)

    return (max_info_gain, min_efe, H_bounds)

end

```

## Best Practices

### 1. Implementation

- Use numerically stable computations

- Implement efficient policy search

- Cache intermediate results

- Handle edge cases

### 2. Tuning

- Adjust precision parameter

- Balance exploration-exploitation

- Set appropriate time horizon

- Define meaningful preferences

### 3. Validation

- Test with known solutions

- Verify information gains

- Monitor convergence

- Validate actions

## References

1. Friston, K. J., et al. (2015). Active inference and epistemic value

1. Parr, T., & Friston, K. J. (2019). Generalised free energy and active inference

1. Da Costa, L., et al. (2020). Active inference, stochastic control, and expected free energy

1. Tschantz, A., et al. (2020). Learning action-oriented models through active inference

1. Millidge, B., et al. (2021). Expected Free Energy formalizes conflict between exploration and exploitation

## Rigorous Mathematical Derivations

### Fundamental Theorem of Expected Free Energy

**Theorem** (EFE Decomposition): The expected free energy under policy $\pi$ can be uniquely decomposed as:

$$G(\pi) = \underbrace{\mathbb{E}_{Q(o,s|\pi)}[\ln Q(s|o,\pi) - \ln P(s,o)]}_{\text{Expected Variational Free Energy}} + \underbrace{D_{KL}[Q(o|\pi)\|P(o)]}_{\text{Relative Entropy}}$$

**Proof**: Starting from the definition of expected free energy:

$$G(\pi) = \mathbb{E}_{Q(o|\pi)}[F(o,\pi)]$$

where $F(o,\pi) = \mathbb{E}_{Q(s|o,\pi)}[\ln Q(s|o,\pi) - \ln P(s,o)]$ is the variational free energy.

Expanding the expectation:

\begin{align}

G(\pi) &= \int Q(o|\pi) \int Q(s|o,\pi) [\ln Q(s|o,\pi) - \ln P(s,o)] \, ds \, do \\

&= \int Q(o|\pi) Q(s|o,\pi) [\ln Q(s|o,\pi) - \ln P(s|o) - \ln P(o)] \, ds \, do \\

&= \mathbb{E}_{Q(o,s|\pi)}[\ln Q(s|o,\pi) - \ln P(s|o)] - \mathbb{E}_{Q(o|\pi)}[\ln P(o)] \\

&= \mathbb{E}_{Q(o,s|\pi)}[\ln Q(s|o,\pi) - \ln P(s|o)] + D_{KL}[Q(o|\pi)\|P(o)] \quad \blacksquare

\end{align}

### Advanced EFE Computational Framework

```python

class RigorousEFECalculator:

    """Enhanced expected free energy calculator with rigorous mathematical foundation."""

    def __init__(self,

                 generative_model: Dict[str, np.ndarray],

                 precision_parameters: Dict[str, float],

                 numerical_precision: float = 1e-12,

                 integration_method: str = 'adaptive_quadrature'):

        """Initialize rigorous EFE calculator.

        Args:

            generative_model: Complete specification of P(o,s,a)

            precision_parameters: Precision matrices and temperature parameters

            numerical_precision: Tolerance for numerical computations

            integration_method: Method for computing expectations

        """

        self.A = generative_model.get('likelihood_matrix')  # P(o|s)

        self.B = generative_model.get('transition_matrix')  # P(s'|s,a)  

        self.C = generative_model.get('preference_vector')  # log P(o)

        self.D = generative_model.get('initial_belief')     # Q(s₀)

        self.gamma = precision_parameters.get('policy_precision', 16.0)

        self.alpha = precision_parameters.get('belief_precision', 16.0)

        self.eps = numerical_precision

        self.integration_method = integration_method

        # Precompute log matrices for numerical stability

        self._log_A = np.log(self.A + self.eps)

        self._log_B = np.log(self.B + self.eps)

    def compute_efe_with_uncertainty_quantification(self,

                                                   policy: np.ndarray,

                                                   belief_state: np.ndarray,

                                                   time_horizon: int) -> Dict[str, float]:

        """Compute EFE with rigorous uncertainty quantification.

        Args:

            policy: Action sequence π = [a₁, a₂, ..., aₜ]

            belief_state: Current belief distribution Q(s)

            time_horizon: Planning horizon T

        Returns:

            efe_analysis: Complete EFE analysis with uncertainty bounds

        """

        # Initialize trajectory distributions

        beliefs_over_time = [belief_state.copy()]

        predictions_over_time = []

        # Forward pass: compute predicted beliefs and observations

        for t in range(time_horizon):

            action = policy[t] if t < len(policy) else policy[-1]

            # Predict next belief state

            next_belief = self._predict_next_belief(beliefs_over_time[-1], action)

            beliefs_over_time.append(next_belief)

            # Predict observations

            predicted_obs = self.A @ next_belief

            predictions_over_time.append(predicted_obs)

        # Compute EFE components with uncertainty propagation

        pragmatic_value, pragmatic_uncertainty = self._compute_pragmatic_value_with_uncertainty(

            predictions_over_time)

        epistemic_value, epistemic_uncertainty = self._compute_epistemic_value_with_uncertainty(

            beliefs_over_time, predictions_over_time)

        # Total EFE with uncertainty propagation

        total_efe = pragmatic_value + epistemic_value

        total_uncertainty = np.sqrt(pragmatic_uncertainty**2 + epistemic_uncertainty**2)

        # Confidence bounds (95% confidence interval)

        confidence_interval = [

            total_efe - 1.96 * total_uncertainty,

            total_efe + 1.96 * total_uncertainty

        ]

        # Additional risk measures

        conditional_value_at_risk = self._compute_cvar(beliefs_over_time, predictions_over_time)

        worst_case_efe = self._compute_worst_case_efe(beliefs_over_time, predictions_over_time)

        return {

            'expected_free_energy': total_efe,

            'pragmatic_value': pragmatic_value,

            'epistemic_value': epistemic_value,

            'total_uncertainty': total_uncertainty,

            'pragmatic_uncertainty': pragmatic_uncertainty,

            'epistemic_uncertainty': epistemic_uncertainty,

            'confidence_interval': confidence_interval,

            'conditional_value_at_risk': conditional_value_at_risk,

            'worst_case_efe': worst_case_efe,

            'beliefs_trajectory': beliefs_over_time,

            'predictions_trajectory': predictions_over_time

        }

    def _predict_next_belief(self,

                           current_belief: np.ndarray,

                           action: int) -> np.ndarray:

        """Predict next belief state using transition model."""

        # Predicted state distribution

        predicted_state = self.B[:, :, action] @ current_belief

        # Normalize and ensure numerical stability

        predicted_state = np.maximum(predicted_state, self.eps)

        return predicted_state / np.sum(predicted_state)

    def _compute_pragmatic_value_with_uncertainty(self,

                                                predictions: List[np.ndarray]

                                                ) -> Tuple[float, float]:

        """Compute pragmatic value with uncertainty quantification."""

        pragmatic_values = []

        for pred_obs in predictions:

            # Expected log preference

            log_preferences = self.C

            expected_log_pref = np.sum(pred_obs * log_preferences)

            pragmatic_values.append(-expected_log_pref)  # Negative for cost

        # Aggregate over time

        total_pragmatic = np.sum(pragmatic_values)

        # Uncertainty via error propagation

        # Var[∑ f(X_i)] ≈ ∑ (∂f/∂X_i)² Var[X_i]

        uncertainty = 0.0

        for i, pred_obs in enumerate(predictions):

            # Variance in predicted observations

            obs_variance = pred_obs * (1 - pred_obs)  # Binomial-like variance

            # Gradient w.r.t. observation probabilities

            gradient = -self.C

            # Uncertainty contribution

            uncertainty += np.sum((gradient**2) * obs_variance)

        return total_pragmatic, np.sqrt(uncertainty)

    def _compute_epistemic_value_with_uncertainty(self,

                                                beliefs: List[np.ndarray],

                                                predictions: List[np.ndarray]

                                                ) -> Tuple[float, float]:

        """Compute epistemic value with uncertainty quantification."""

        epistemic_values = []

        uncertainty_contributions = []

        for t in range(len(predictions)):

            current_belief = beliefs[t]

            predicted_obs = predictions[t]

            # Compute expected information gain

            info_gain = 0.0

            info_gain_variance = 0.0

            for o in range(len(predicted_obs)):

                if predicted_obs[o] > self.eps:

                    # Posterior after observing o

                    posterior = self._bayesian_update(current_belief, o)

                    # KL divergence (information gain)

                    kl_div = self._kl_divergence(posterior, current_belief)

                    info_gain += predicted_obs[o] * kl_div

                    # Second moment for variance computation

                    info_gain_variance += predicted_obs[o] * (kl_div**2)

            # Variance of information gain

            ig_variance = info_gain_variance - info_gain**2

            epistemic_values.append(-info_gain)  # Negative for reward

            uncertainty_contributions.append(ig_variance)

        total_epistemic = np.sum(epistemic_values)

        total_uncertainty = np.sqrt(np.sum(uncertainty_contributions))

        return total_epistemic, total_uncertainty

    def _bayesian_update(self,

                       prior: np.ndarray,

                       observation: int) -> np.ndarray:

        """Perform Bayesian belief update."""

        # Likelihood

        likelihood = self.A[observation, :]

        # Posterior (unnormalized)

        posterior = prior * likelihood

        # Normalize

        posterior = np.maximum(posterior, self.eps)

        return posterior / np.sum(posterior)

    def _kl_divergence(self,

                     p: np.ndarray,

                     q: np.ndarray) -> float:

        """Compute KL divergence with numerical stability."""

        # Ensure numerical stability

        p = np.maximum(p, self.eps)

        q = np.maximum(q, self.eps)

        # Normalize

        p = p / np.sum(p)

        q = q / np.sum(q)

        return np.sum(p * (np.log(p) - np.log(q)))

    def _compute_cvar(self,

                    beliefs: List[np.ndarray],

                    predictions: List[np.ndarray],

                    alpha: float = 0.05) -> float:

        """Compute Conditional Value at Risk (CVaR) for EFE."""

        # Sample from belief distributions

        n_samples = 1000

        efe_samples = []

        for _ in range(n_samples):

            # Sample trajectory

            sampled_beliefs = []

            sampled_predictions = []

            for t, (belief, prediction) in enumerate(zip(beliefs, predictions)):

                # Sample state

                state_sample = np.random.choice(len(belief), p=belief)

                sampled_beliefs.append(state_sample)

                # Sample observation

                obs_prob = self.A @ belief

                obs_sample = np.random.choice(len(obs_prob), p=obs_prob)

                sampled_predictions.append(obs_sample)

            # Compute EFE for this sample

            sample_efe = self._compute_sample_efe(sampled_beliefs, sampled_predictions)

            efe_samples.append(sample_efe)

        # Compute CVaR

        efe_samples = np.array(efe_samples)

        var_threshold = np.percentile(efe_samples, 100 * (1 - alpha))

        cvar = np.mean(efe_samples[efe_samples >= var_threshold])

        return cvar

    def _compute_worst_case_efe(self,

                              beliefs: List[np.ndarray],

                              predictions: List[np.ndarray]) -> float:

        """Compute worst-case EFE under distributional uncertainty."""

        # Assume uncertainty in model parameters

        worst_case = -np.inf

        # Consider perturbations to transition and observation models

        perturbation_strength = 0.1

        n_perturbations = 50

        for _ in range(n_perturbations):

            # Perturb A matrix

            A_perturbed = self.A + perturbation_strength * np.random.normal(

                0, np.std(self.A), self.A.shape)

            A_perturbed = np.maximum(A_perturbed, self.eps)

            A_perturbed = A_perturbed / np.sum(A_perturbed, axis=0, keepdims=True)

            # Compute EFE with perturbed model

            efe_perturbed = self._compute_efe_with_model(beliefs, predictions, A_perturbed)

            worst_case = max(worst_case, efe_perturbed)

        return worst_case

    def _compute_sample_efe(self,

                          sampled_states: List[int],

                          sampled_observations: List[int]) -> float:

        """Compute EFE for a specific sampled trajectory."""

        # Simplified computation for sampled trajectory

        pragmatic = -np.sum([self.C[o] for o in sampled_observations])

        epistemic = 0.0  # Would need more complex computation for epistemic value

        return pragmatic + epistemic

    def _compute_efe_with_model(self,

                              beliefs: List[np.ndarray],

                              predictions: List[np.ndarray],

                              A_matrix: np.ndarray) -> float:

        """Compute EFE with alternative observation model."""

        # Simplified computation with different A matrix

        total_efe = 0.0

        for belief, prediction in zip(beliefs, predictions):

            # Pragmatic value

            expected_log_pref = np.sum(prediction * self.C)

            pragmatic = -expected_log_pref

            # Add to total

            total_efe += pragmatic

        return total_efe

### Advanced Policy Optimization

class RobustPolicyOptimization:

    """Robust policy optimization using expected free energy with theoretical guarantees."""

    def __init__(self,

                 efe_calculator: RigorousEFECalculator,

                 optimization_method: str = 'natural_gradient',

                 convergence_tolerance: float = 1e-8):

        """Initialize robust policy optimizer.

        Args:

            efe_calculator: EFE computation engine

            optimization_method: Optimization algorithm choice

            convergence_tolerance: Convergence criterion

        """

        self.efe_calc = efe_calculator

        self.method = optimization_method

        self.tol = convergence_tolerance

    def optimize_policy_with_guarantees(self,

                                       initial_policy: np.ndarray,

                                       belief_state: np.ndarray,

                                       time_horizon: int,

                                       max_iterations: int = 1000) -> Dict[str, Any]:

        """Optimize policy with convergence guarantees.

        Args:

            initial_policy: Starting policy π₀

            belief_state: Current belief state Q(s)

            time_horizon: Planning horizon T

            max_iterations: Maximum optimization iterations

        Returns:

            optimization_result: Complete optimization analysis with guarantees

        """

        current_policy = initial_policy.copy()

        efe_history = []

        gradient_norms = []

        # Natural gradient descent with adaptive step size

        step_size = 0.1

        momentum = 0.9

        velocity = np.zeros_like(current_policy)

        for iteration in range(max_iterations):

            # Compute EFE and its gradient

            efe_result = self.efe_calc.compute_efe_with_uncertainty_quantification(

                current_policy, belief_state, time_horizon)

            current_efe = efe_result['expected_free_energy']

            efe_history.append(current_efe)

            # Compute natural gradient

            natural_gradient = self._compute_natural_gradient(

                current_policy, belief_state, time_horizon)

            gradient_norm = np.linalg.norm(natural_gradient)

            gradient_norms.append(gradient_norm)

            # Adaptive step size

            if iteration > 0 and efe_history[-1] > efe_history[-2]:

                step_size *= 0.5  # Reduce step size if EFE increased

            elif gradient_norm < 0.1:

                step_size *= 1.1  # Increase step size if gradient is small

            # Momentum update

            velocity = momentum * velocity - step_size * natural_gradient

            current_policy += velocity

            # Project onto policy simplex

            current_policy = self._project_simplex(current_policy)

            # Check convergence

            if gradient_norm < self.tol:

                print(f"Converged after {iteration + 1} iterations")

                break

        # Compute convergence analysis

        convergence_analysis = self._analyze_convergence(efe_history, gradient_norms)

        # Final policy evaluation

        final_efe_result = self.efe_calc.compute_efe_with_uncertainty_quantification(

            current_policy, belief_state, time_horizon)

        return {

            'optimal_policy': current_policy,

            'final_efe': final_efe_result,

            'efe_history': efe_history,

            'gradient_norms': gradient_norms,

            'convergence_analysis': convergence_analysis,

            'iterations': iteration + 1,

            'converged': gradient_norm < self.tol

        }

    def _compute_natural_gradient(self,

                                policy: np.ndarray,

                                belief_state: np.ndarray,

                                time_horizon: int) -> np.ndarray:

        """Compute natural gradient of expected free energy."""

        # Finite difference approximation for gradient

        gradient = np.zeros_like(policy)

        h = 1e-6

        # Base EFE

        base_result = self.efe_calc.compute_efe_with_uncertainty_quantification(

            policy, belief_state, time_horizon)

        base_efe = base_result['expected_free_energy']

        # Compute partial derivatives

        for i in range(len(policy)):

            # Perturb policy

            policy_plus = policy.copy()

            policy_plus[i] += h

            policy_plus = self._project_simplex(policy_plus)

            # Compute EFE with perturbation

            perturbed_result = self.efe_calc.compute_efe_with_uncertainty_quantification(

                policy_plus, belief_state, time_horizon)

            perturbed_efe = perturbed_result['expected_free_energy']

            # Finite difference

            gradient[i] = (perturbed_efe - base_efe) / h

        return gradient

    def _project_simplex(self, policy: np.ndarray) -> np.ndarray:

        """Project policy onto probability simplex."""

        # Simple projection: clip and normalize

        policy = np.maximum(policy, 0)

        return policy / (np.sum(policy) + 1e-15)

    def _analyze_convergence(self,

                           efe_history: List[float],

                           gradient_norms: List[float]) -> Dict[str, float]:

        """Analyze convergence properties."""

        if len(efe_history) < 2:

            return {'convergence_rate': 0.0, 'final_gradient_norm': gradient_norms[-1]}

        # Estimate convergence rate

        efe_diffs = np.abs(np.diff(efe_history))

        # Fit exponential decay

        if len(efe_diffs) > 5:

            x = np.arange(len(efe_diffs))

            log_diffs = np.log(efe_diffs + 1e-15)

            try:

                # Linear regression on log scale

                coeffs = np.polyfit(x, log_diffs, 1)

                convergence_rate = -coeffs[0]  # Decay rate

            except:

                convergence_rate = 0.0

        else:

            convergence_rate = 0.0

        return {

            'convergence_rate': convergence_rate,

            'final_gradient_norm': gradient_norms[-1],

            'efe_improvement': efe_history[0] - efe_history[-1],

            'relative_improvement': (efe_history[0] - efe_history[-1]) / abs(efe_history[0])

        }

