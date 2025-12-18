---
title: Variational Free Energy
type: concept
status: stable
created: 2025-01-01
tags:
  - active_inference
  - variational_inference
  - free_energy
  - bayesian_inference
semantic_relations:
  - type: relates
    links:
      - free_energy_principle
      - variational_inference
      - active_inference
      - bayesian_brain_hypothesis
  - type: foundation
    links:
      - [[../mathematics/variational_free_energy]]
      - [[../mathematics/variational_inference]]
---

# Variational Free Energy

Variational free energy is a key quantity in active inference and the free energy principle, serving as both a measure of model evidence and an objective function for perception and action. It provides a unified framework for understanding Bayesian inference in biological systems.

## Mathematical Foundations

### Definition

The variational free energy F is defined as:

F(θ,μ) = E_{q(θ)}[ln q(θ) - ln p(y,θ)] + H[q(θ)]

Where:
- θ represents latent variables (hidden states)
- μ represents variational parameters
- y represents observed data
- q(θ) is the variational approximation to the true posterior
- H[q(θ)] is the entropy of the variational distribution

```python
class VariationalFreeEnergy:
    """Implementation of variational free energy calculations."""

    def __init__(self, generative_model):
        self.model = generative_model

    def calculate_vfe(self, observations, variational_params):
        """Calculate variational free energy for given observations."""

        # Energy term: E_q[ln q - ln p(y,θ)]
        energy = self.calculate_energy_term(observations, variational_params)

        # Entropy term: -H[q(θ)]
        entropy = self.calculate_entropy_term(variational_params)

        # Variational free energy
        vfe = energy - entropy

        return vfe

    def calculate_energy_term(self, observations, variational_params):
        """Calculate the energy term of VFE."""

        energy = 0.0

        # Sample from variational distribution
        n_samples = 100
        for _ in range(n_samples):
            # Sample latent variables
            latent_sample = self.sample_from_variational(variational_params)

            # Evaluate joint log-likelihood
            log_joint = self.model.log_joint_probability(observations, latent_sample)

            # Evaluate variational log-probability
            log_variational = self.variational_log_probability(latent_sample, variational_params)

            energy += log_variational - log_joint

        return energy / n_samples

    def calculate_entropy_term(self, variational_params):
        """Calculate the entropy term of VFE."""

        # For common distributions, entropy has analytical form
        if variational_params['distribution'] == 'gaussian':
            entropy = self.gaussian_entropy(variational_params)
        elif variational_params['distribution'] == 'categorical':
            entropy = self.categorical_entropy(variational_params)

        return entropy
```

### Relationship to Evidence Lower Bound

Variational free energy provides a lower bound on model evidence:

ln p(y) ≥ E_{q(θ)}[ln p(y,θ)] - E_{q(θ)}[ln q(θ)] = -F

This relationship enables variational inference as maximum likelihood learning.

## Perception as Inference

### Predictive Coding

Variational free energy minimization through predictive coding:

```python
class PredictiveCodingNetwork:
    """Neural network implementing predictive coding."""

    def __init__(self, layers, learning_rate=0.01):
        self.layers = layers
        self.learning_rate = learning_rate
        self.predictions = {}
        self.errors = {}

    def predict(self, input_data):
        """Generate predictions through hierarchical network."""

        # Forward pass - generate predictions
        current_input = input_data
        for layer in self.layers:
            prediction = layer.predict(current_input)
            self.predictions[layer.level] = prediction

            # Prediction error
            if layer.level > 0:
                error = current_input - prediction
                self.errors[layer.level] = error

            current_input = prediction

        return self.predictions[self.layers[-1].level]

    def update_beliefs(self, observation):
        """Update beliefs by minimizing variational free energy."""

        # Initialize with observation
        self.errors[0] = observation

        # Backward pass - update predictions
        for layer_idx in reversed(range(len(self.layers))):
            layer = self.layers[layer_idx]

            # Calculate precision-weighted prediction error
            precision = layer.get_precision()
            error = self.errors[layer.level]

            # Update layer parameters
            layer.update_parameters(error, precision, self.learning_rate)

            # Propagate error upward
            if layer_idx > 0:
                prediction_error = layer.backward_error(error)
                self.errors[layer.level - 1] += prediction_error

    def minimize_vfe(self, observation, max_iterations=100):
        """Minimize variational free energy through iteration."""

        for iteration in range(max_iterations):
            # Generate predictions
            self.predict(observation)

            # Update beliefs to reduce prediction errors
            self.update_beliefs(observation)

            # Check convergence
            total_error = sum(np.sum(error**2) for error in self.errors.values())
            if total_error < 1e-6:
                break

        return self.predictions
```

## Action as Inference

### Expected Free Energy

Action selection through expected free energy minimization:

```python
class ActiveInferenceAgent:
    """Agent that selects actions by minimizing expected free energy."""

    def __init__(self, generative_model, action_space):
        self.model = generative_model
        self.actions = action_space
        self.policy_prior = self.initialize_policy_prior()

    def select_action(self, current_beliefs):
        """Select action that minimizes expected free energy."""

        action_values = {}

        for action in self.actions:
            # Calculate expected free energy for this action
            efe = self.calculate_expected_free_energy(action, current_beliefs)
            action_values[action] = efe

        # Select action with minimum EFE
        optimal_action = min(action_values, key=action_values.get)

        return optimal_action, action_values

    def calculate_expected_free_energy(self, action, beliefs):
        """Calculate expected free energy for an action."""

        efe = 0.0

        # Epistemic affordance (information gain)
        epistemic_efe = self.calculate_epistemic_affordance(action, beliefs)

        # Extrinsic value (goal achievement)
        extrinsic_efe = self.calculate_extrinsic_value(action, beliefs)

        # Intrinsic value (homeostatic regulation)
        intrinsic_efe = self.calculate_intrinsic_value(action, beliefs)

        efe = epistemic_efe + extrinsic_efe + intrinsic_efe

        return efe

    def calculate_epistemic_affordance(self, action, beliefs):
        """Calculate information gain from action."""

        # Predict posterior beliefs after action
        predicted_beliefs = self.model.predict_posterior(action, beliefs)

        # KL divergence between prior and posterior
        epistemic_value = kl_divergence(beliefs, predicted_beliefs)

        return epistemic_value

    def calculate_extrinsic_value(self, action, beliefs):
        """Calculate goal-directed value."""

        # Predict sensory consequences
        predicted_observations = self.model.predict_observations(action, beliefs)

        # Compare to goal prior
        extrinsic_value = kl_divergence(predicted_observations, self.goal_prior)

        return extrinsic_value
```

## Learning and Adaptation

### Variational Learning

Model parameter learning through variational free energy minimization:

```python
class VariationalLearner:
    """Learn generative model parameters using variational methods."""

    def __init__(self, model, variational_family):
        self.model = model
        self.variational_family = variational_family
        self.learning_rate = 0.01

    def learn_from_data(self, dataset, max_epochs=100):
        """Learn model parameters from data."""

        for epoch in range(max_epochs):
            epoch_loss = 0.0

            for batch in dataset:
                # Calculate variational free energy
                vfe = self.calculate_batch_vfe(batch)

                # Compute gradients
                gradients = self.compute_vfe_gradients(batch)

                # Update model parameters
                self.update_parameters(gradients)

                epoch_loss += vfe

            # Check convergence
            if epoch_loss < 1e-6:
                break

        return self.model

    def calculate_batch_vfe(self, batch):
        """Calculate variational free energy for a batch."""

        total_vfe = 0.0

        for observation in batch:
            # Optimize variational parameters for this observation
            variational_params = self.optimize_variational_params(observation)

            # Calculate VFE
            vfe = self.vfe_calculator.calculate_vfe(observation, variational_params)
            total_vfe += vfe

        return total_vfe / len(batch)

    def optimize_variational_params(self, observation):
        """Optimize variational parameters for single observation."""

        # Initialize variational parameters
        params = self.variational_family.initialize_params()

        # Gradient-based optimization
        for _ in range(50):  # Limited iterations for efficiency
            vfe = self.vfe_calculator.calculate_vfe(observation, params)
            gradients = self.compute_variational_gradients(observation, params)

            # Update parameters
            params = self.update_variational_params(params, gradients)

        return params
```

## Information Geometry

### Fisher Information and Natural Gradients

Using the Fisher information matrix for efficient optimization:

```python
class NaturalGradientOptimizer:
    """Optimize using natural gradients based on Fisher information."""

    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.learning_rate = learning_rate

    def natural_gradient_update(self, observation, variational_params):
        """Update parameters using natural gradient."""

        # Calculate standard gradient
        gradient = self.calculate_vfe_gradient(observation, variational_params)

        # Calculate Fisher information matrix
        fisher_matrix = self.calculate_fisher_information(variational_params)

        # Compute natural gradient
        natural_gradient = np.linalg.solve(fisher_matrix, gradient)

        # Update parameters
        self.model.update_parameters(
            -self.learning_rate * natural_gradient
        )

        return natural_gradient

    def calculate_fisher_information(self, variational_params):
        """Calculate Fisher information matrix."""

        # Monte Carlo estimation of Fisher information
        n_samples = 100
        fisher_matrix = np.zeros((len(variational_params), len(variational_params)))

        for _ in range(n_samples):
            # Sample from variational distribution
            sample = self.sample_from_variational(variational_params)

            # Score function
            score = self.score_function(sample, variational_params)

            # Outer product
            fisher_matrix += np.outer(score, score)

        fisher_matrix /= n_samples

        return fisher_matrix

    def score_function(self, sample, params):
        """Calculate score function for variational distribution."""

        if params['distribution'] == 'gaussian':
            # For Gaussian: (x - μ) / σ²
            mean = params['mean']
            variance = params['variance']
            score = (sample - mean) / variance
        elif params['distribution'] == 'categorical':
            # For categorical: score for selected category
            category_idx = np.argmax(sample)
            score = np.zeros(len(params['probabilities']))
            score[category_idx] = 1.0 / params['probabilities'][category_idx]

        return score
```

## Neural Implementation

### Synaptic Plasticity

Synaptic changes that implement variational free energy minimization:

```python
class SynapticPlasticity:
    """Synaptic plasticity implementing variational learning."""

    def __init__(self, neurons, connections):
        self.neurons = neurons
        self.connections = connections

    def hebbian_learning(self, pre_activity, post_activity, dopamine_signal):
        """Hebbian learning modulated by dopamine."""

        for connection in self.connections:
            # Pre- and post-synaptic activity
            pre = pre_activity[connection.pre_neuron]
            post = post_activity[connection.post_neuron]

            # Prediction error signal (dopamine)
            pe = dopamine_signal

            # Weight update
            delta_w = self.learning_rate * pre * post * pe
            connection.weight += delta_w

            # Maintain weight bounds
            connection.weight = np.clip(connection.weight, -1.0, 1.0)

    def precision_weighted_learning(self, prediction_error, precision):
        """Learning weighted by precision (confidence)."""

        # Precision-modulated learning rate
        effective_lr = self.learning_rate * precision

        # Update weights based on precision-weighted error
        for connection in self.connections:
            pre = self.neurons[connection.pre_neuron].activity
            error_contribution = prediction_error * connection.weight

            delta_w = effective_lr * pre * error_contribution
            connection.weight += delta_w
```

## Applications

### Perception

Variational free energy in sensory processing:

- **Visual perception**: Contour integration and figure-ground segregation
- **Auditory perception**: Stream segregation and auditory scene analysis
- **Multisensory integration**: Combining information from multiple modalities

### Motor Control

Active inference in movement generation:

- **Optimal feedback control**: State estimation and control
- **Motor learning**: Adaptation to novel dynamics
- **Motor coordination**: Synchronization and coordination

### Decision Making

Value-based choices under uncertainty:

- **Reinforcement learning**: Learning state-action values
- **Social decision making**: Theory of mind and cooperation
- **Economic choices**: Risk and ambiguity attitudes

## Computational Considerations

### Scalability

Making variational inference tractable for large models:

```python
class ScalableVariationalInference:
    """Scalable variational inference methods."""

    def __init__(self, model):
        self.model = model

    def stochastic_variational_inference(self, data, batch_size=100):
        """Stochastic variational inference for large datasets."""

        # Initialize variational parameters
        variational_params = self.initialize_variational_params()

        for epoch in range(self.max_epochs):
            # Shuffle data
            np.random.shuffle(data)

            for batch_start in range(0, len(data), batch_size):
                batch_end = min(batch_start + batch_size, len(data))
                batch = data[batch_start:batch_end]

                # Calculate stochastic gradients
                gradients = self.calculate_stochastic_gradients(batch, variational_params)

                # Update variational parameters
                variational_params = self.update_variational_params(
                    variational_params, gradients
                )

        return variational_params

    def amortized_variational_inference(self, data):
        """Amortized inference using neural networks."""

        # Train inference network
        inference_network = self.train_inference_network(data)

        # Use network for fast inference
        variational_params = inference_network.encode(data)

        return variational_params

    def train_inference_network(self, data):
        """Train neural network for amortized inference."""

        # Encoder network architecture
        encoder = self.build_encoder_network()

        # Training loop
        for batch in self.create_batches(data):
            # Encode data to variational parameters
            params = encoder(batch)

            # Calculate VFE
            vfe = self.calculate_vfe_batch(batch, params)

            # Backpropagation
            self.optimizer.zero_grad()
            vfe.backward()
            self.optimizer.step()

        return encoder
```

---

## Related Concepts

- [[free_energy_principle]] - Foundational principle
- [[active_inference]] - Action-oriented inference
- [[predictive_coding]] - Neural implementation
- [[variational_inference]] - General inference framework
- [[bayesian_inference]] - Probabilistic reasoning
