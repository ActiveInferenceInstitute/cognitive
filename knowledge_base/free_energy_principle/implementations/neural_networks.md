---
title: "Neural Network Implementations of Active Inference"
type: implementation
status: stable
created: 2025-02-06
updated: 2025-02-06
tags:
  - free_energy_principle
  - implementation
  - neural_networks
  - deep_learning
  - variational_autoencoders
  - active_inference
  - amortized_inference
  - world_models
semantic_relations:
  - type: relates
    links:
      - [[python_framework|Python Framework]]
      - [[robotics|Robotics Implementations]]
      - [[simulation|Simulation Environments]]
      - [[benchmarking|Benchmarking]]
      - [[knowledge_base/free_energy_principle/mathematics/variational_free_energy|Variational Free Energy]]
      - [[knowledge_base/free_energy_principle/mathematics/core_principle|Core Principle]]
      - [[knowledge_base/free_energy_principle/cognitive/perception|Perception]]
      - [[knowledge_base/free_energy_principle/cognitive/learning|Learning]]
---

# Neural Network Implementations of Active Inference

## Overview

The Free Energy Principle (FEP) and active inference provide a normative framework for understanding perception, action, and learning. Neural networks offer a scalable computational substrate for implementing these principles in high-dimensional, real-world domains. This document covers how deep learning architectures -- variational autoencoders, transformers, and world models -- can be understood as, or used to implement, free energy minimizing agents.

The central insight is that the variational free energy objective used in the FEP is mathematically equivalent to the Evidence Lower Bound (ELBO) optimized by variational autoencoders. This equivalence provides a direct bridge between modern deep learning and active inference.

## Architecture and Design

### The FEP-Deep Learning Correspondence

| FEP Concept | Deep Learning Equivalent |
|---|---|
| Generative model $p(o, s)$ | Decoder network $p_\theta(x \mid z)p(z)$ |
| Recognition model $q(s \mid o)$ | Encoder network $q_\phi(z \mid x)$ |
| Variational free energy | Negative ELBO |
| Prediction error minimization | Reconstruction loss minimization |
| Precision weighting | Learned variance / attention |
| Active inference | Policy optimization under EFE |
| Hierarchical generative model | Deep generative model with multiple latent layers |

### Variational Autoencoders as Free Energy Minimization

A VAE optimizes the ELBO, which is the negative of variational free energy:

$$
\mathcal{F}[q_\phi, \theta] = \underbrace{D_{KL}[q_\phi(z|x) \| p(z)]}_{\text{complexity}} - \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{accuracy}}
$$

This decomposes into:
- **Complexity**: The KL divergence penalizes the approximate posterior for deviating from the prior -- this is the "Occam's razor" term in FEP.
- **Accuracy**: The expected log-likelihood ensures the model explains observations -- this is the "prediction error" term.

Minimizing $\mathcal{F}$ with respect to $\phi$ (encoder parameters) performs approximate Bayesian inference. Minimizing with respect to $\theta$ (decoder parameters) performs model learning. Both correspond to the two key operations in the FEP.

### The Reparameterization Trick

To backpropagate through the stochastic sampling operation $z \sim q_\phi(z|x)$, we use the reparameterization trick:

$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

This separates the deterministic computation (which gradients flow through) from the stochastic component. In FEP terms, this corresponds to parameterizing the sufficient statistics of the approximate posterior (mean and precision) as functions of observations.

### Amortized Inference

Traditional active inference performs iterative inference for each new observation. Neural networks enable **amortized inference**: the encoder $q_\phi(z|x)$ learns a direct mapping from observations to approximate posteriors, eliminating the need for iterative optimization at test time.

$$
\text{Iterative:} \quad q^*(s) = \arg\min_{q} \mathcal{F}[q, o_t] \quad \text{(slow, per observation)}
$$
$$
\text{Amortized:} \quad q_\phi(s|o_t) \approx q^*(s) \quad \text{(fast, single forward pass)}
$$

The amortization gap -- the difference between the amortized and optimal posteriors -- represents the cost of speed. This can be reduced through iterative amortization (running a few gradient steps on $\phi$ at test time).

## Implementation Details

### Deep Active Inference Agent Architecture

A deep active inference agent consists of three core components:

1. **Generative model** (world model): $p_\theta(o_{t+1}, s_{t+1} | s_t, a_t)$
   - Predicts future observations and states given current state and action
   - Implemented as a recurrent or transformer-based neural network

2. **Recognition model** (encoder): $q_\phi(s_t | o_{\leq t}, a_{< t})$
   - Infers hidden states from observations and action history
   - Implemented as a recurrent encoder or attention-based encoder

3. **Policy network** (action model): $\pi_\psi(a_t | s_t)$
   - Selects actions that minimize expected free energy
   - Can be trained via policy gradient or direct EFE optimization

### Contrastive Learning Connections

Contrastive learning methods can be understood through the FEP lens. The InfoNCE loss used in contrastive learning provides a lower bound on mutual information:

$$
\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}\left[\log \frac{f(x, z^+)}{f(x, z^+) + \sum_{j} f(x, z^-_j)}\right]
$$

This is related to free energy minimization because maximizing mutual information between observations and latent states is equivalent to minimizing uncertainty (surprise) about observations given the model. Contrastive methods learn representations that are maximally informative about observations -- exactly what the FEP prescribes.

### Transformer-Based Active Inference

Transformers offer several advantages for active inference:

- **Attention as precision weighting**: Self-attention naturally implements precision-weighted message passing. The attention weights can be interpreted as the precision (inverse variance) assigned to each element.
- **Sequence modeling**: Transformers excel at modeling temporal dependencies in observation sequences, essential for world models.
- **Scalability**: Transformer architectures scale to very high-dimensional observation and action spaces.

The attention mechanism computes:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Under the FEP interpretation, $Q$ represents predictions, $K$ represents observations, and the softmax-scaled dot product computes precision-weighted prediction errors. The output $V$ is the precision-weighted update to beliefs.

## Code Examples

### PyTorch VAE Implementing Free Energy Minimization

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence


class FEPVariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder framed as Free Energy Minimization.

    The encoder is the recognition model q(z|x), performing approximate
    Bayesian inference (perception in FEP terms).
    The decoder is the generative model p(x|z), defining how hidden
    causes generate observations.
    Training minimizes variational free energy F = complexity - accuracy.
    """

    def __init__(self, obs_dim: int, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim

        # Recognition model q_phi(z | x) -- "perception"
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        # Generative model p_theta(x | z) -- "generation"
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

        # Prior p(z) = N(0, I)
        self.prior_mu = torch.zeros(latent_dim)
        self.prior_logvar = torch.zeros(latent_dim)

    def encode(self, x: torch.Tensor):
        """Recognition model: infer approximate posterior q(z|x)."""
        h = self.encoder(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """Reparameterization trick: z = mu + sigma * epsilon."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor):
        """Generative model: predict observations from latent states."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z

    def free_energy(self, x: torch.Tensor):
        """
        Compute variational free energy F = complexity - accuracy.

        F = D_KL[q(z|x) || p(z)] - E_q[log p(x|z)]

        Returns total free energy and its components for analysis.
        """
        x_recon, mu, logvar, z = self.forward(x)

        # Complexity: KL divergence from prior
        # D_KL[N(mu, sigma^2) || N(0, I)]
        complexity = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(), dim=-1
        )

        # Accuracy: expected log-likelihood (reconstruction)
        accuracy = -F.mse_loss(x_recon, x, reduction='none').sum(dim=-1)

        # Free energy = complexity - accuracy
        F_total = complexity - accuracy  # We minimize this

        return F_total.mean(), complexity.mean(), (-accuracy).mean()
```

### Deep Active Inference Agent

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class WorldModel(nn.Module):
    """
    Recurrent world model for deep active inference.
    Learns transition dynamics: p(s_{t+1}, o_{t+1} | s_t, a_t)
    """

    def __init__(self, obs_dim: int, action_dim: int, state_dim: int,
                 hidden_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim

        # Encoder: o_t -> features
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Recurrent core: (h_t, features, a_t) -> h_{t+1}
        self.rnn = nn.GRUCell(hidden_dim + action_dim, hidden_dim)

        # State posterior: q(s_t | h_t, o_t)
        self.posterior_mu = nn.Linear(hidden_dim + hidden_dim, state_dim)
        self.posterior_logvar = nn.Linear(hidden_dim + hidden_dim, state_dim)

        # State prior (transition): p(s_{t+1} | h_t)
        self.prior_mu = nn.Linear(hidden_dim, state_dim)
        self.prior_logvar = nn.Linear(hidden_dim, state_dim)

        # Observation decoder: p(o_t | s_t, h_t)
        self.obs_decoder = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

        # Reward predictor: p(r_t | s_t, h_t)
        self.reward_head = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs_seq: torch.Tensor, action_seq: torch.Tensor):
        """
        Process a sequence of observations and actions.
        Returns posteriors, priors, reconstructions, rewards.
        """
        batch_size, seq_len, _ = obs_seq.shape
        h = torch.zeros(batch_size, 256, device=obs_seq.device)

        posteriors_mu, posteriors_logvar = [], []
        priors_mu, priors_logvar = [], []
        obs_recons, reward_preds = [], []

        for t in range(seq_len):
            obs_feat = self.obs_encoder(obs_seq[:, t])

            # Prior: p(s_t | h_t) -- prediction before seeing observation
            p_mu = self.prior_mu(h)
            p_logvar = self.prior_logvar(h)

            # Posterior: q(s_t | h_t, o_t) -- belief after seeing observation
            combined = torch.cat([h, obs_feat], dim=-1)
            q_mu = self.posterior_mu(combined)
            q_logvar = self.posterior_logvar(combined)

            # Sample state from posterior
            std = torch.exp(0.5 * q_logvar)
            eps = torch.randn_like(std)
            s_t = q_mu + std * eps

            # Decode observation and reward
            decode_input = torch.cat([s_t, h], dim=-1)
            obs_recon = self.obs_decoder(decode_input)
            reward_pred = self.reward_head(decode_input)

            # Update recurrent state
            if t < seq_len - 1:
                rnn_input = torch.cat([obs_feat, action_seq[:, t]], dim=-1)
                h = self.rnn(rnn_input, h)

            posteriors_mu.append(q_mu)
            posteriors_logvar.append(q_logvar)
            priors_mu.append(p_mu)
            priors_logvar.append(p_logvar)
            obs_recons.append(obs_recon)
            reward_preds.append(reward_pred)

        return {
            'posterior_mu': torch.stack(posteriors_mu, dim=1),
            'posterior_logvar': torch.stack(posteriors_logvar, dim=1),
            'prior_mu': torch.stack(priors_mu, dim=1),
            'prior_logvar': torch.stack(priors_logvar, dim=1),
            'obs_recon': torch.stack(obs_recons, dim=1),
            'reward_pred': torch.stack(reward_preds, dim=1),
        }


class DeepActiveInferenceAgent:
    """
    Deep active inference agent using a learned world model.

    The agent:
    1. Maintains a world model (generative model)
    2. Performs amortized inference via the encoder (perception)
    3. Selects actions by minimizing expected free energy (planning)
    4. Updates its model from experience (learning)
    """

    def __init__(self, obs_dim: int, action_dim: int, state_dim: int = 32,
                 planning_horizon: int = 15, n_candidates: int = 500,
                 top_k: int = 50, learning_rate: float = 1e-3):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.planning_horizon = planning_horizon
        self.n_candidates = n_candidates
        self.top_k = top_k

        self.world_model = WorldModel(obs_dim, action_dim, state_dim)
        self.optimizer = torch.optim.Adam(
            self.world_model.parameters(), lr=learning_rate
        )

        # Preferred observations (prior preferences in FEP)
        self.preferred_obs = None

    def set_preferences(self, preferred_obs: np.ndarray):
        """Set prior preferences C -- the observations the agent 'expects'."""
        self.preferred_obs = torch.tensor(preferred_obs, dtype=torch.float32)

    def compute_world_model_loss(self, obs_seq, action_seq):
        """
        Compute variational free energy for the world model.

        F = sum_t [ D_KL[q(s_t|h_t,o_t) || p(s_t|h_t)] - E_q[log p(o_t|s_t,h_t)] ]

        This is the sequence-level ELBO, decomposed into per-timestep terms.
        """
        outputs = self.world_model(obs_seq, action_seq)

        # Complexity: KL divergence between posterior and prior at each timestep
        complexity = -0.5 * torch.sum(
            1 + outputs['posterior_logvar'] - outputs['prior_logvar']
            - (outputs['posterior_logvar'].exp()
               + (outputs['posterior_mu'] - outputs['prior_mu']).pow(2))
            / outputs['prior_logvar'].exp(),
            dim=-1
        ).mean()

        # Accuracy: reconstruction of observations
        recon_loss = F.mse_loss(outputs['obs_recon'], obs_seq)

        # Total free energy
        free_energy = complexity + recon_loss
        return free_energy, complexity, recon_loss

    def plan_actions(self, current_obs: np.ndarray) -> np.ndarray:
        """
        Plan actions by minimizing expected free energy via CEM.

        G(pi) = E_q(o_tau|pi) [ -log p(o_tau) - H[q(s_tau|pi)] ]
              = pragmatic_value + epistemic_value

        Uses Cross-Entropy Method (CEM) for action optimization.
        """
        obs_tensor = torch.tensor(current_obs, dtype=torch.float32).unsqueeze(0)

        # Initialize action distribution
        action_mu = torch.zeros(self.planning_horizon, self.action_dim)
        action_std = torch.ones(self.planning_horizon, self.action_dim)

        for iteration in range(5):  # CEM iterations
            # Sample candidate action sequences
            actions = action_mu.unsqueeze(0) + action_std.unsqueeze(0) * \
                torch.randn(self.n_candidates, self.planning_horizon,
                            self.action_dim)
            actions = torch.clamp(actions, -1.0, 1.0)

            # Evaluate expected free energy for each candidate
            efe_scores = self._evaluate_efe(obs_tensor, actions)

            # Select top-k candidates
            _, top_indices = torch.topk(-efe_scores, self.top_k)
            top_actions = actions[top_indices]

            # Update action distribution
            action_mu = top_actions.mean(dim=0)
            action_std = top_actions.std(dim=0) + 1e-6

        return action_mu[0].detach().numpy()

    def _evaluate_efe(self, obs: torch.Tensor,
                      action_seqs: torch.Tensor) -> torch.Tensor:
        """
        Evaluate expected free energy for action sequences.

        G = -pragmatic_value - epistemic_value
          = E_q[log q(s) - log p(o) - log p(s|o)]
        """
        n_candidates = action_seqs.shape[0]
        obs_expanded = obs.expand(n_candidates, -1).unsqueeze(1)

        efe = torch.zeros(n_candidates)

        with torch.no_grad():
            h = torch.zeros(n_candidates, 256)

            for t in range(self.planning_horizon):
                # Prior prediction
                p_mu = self.world_model.prior_mu(h)
                p_logvar = self.world_model.prior_logvar(h)

                # Pragmatic value: how close predicted obs are to preferences
                s_sample = p_mu + torch.exp(0.5 * p_logvar) * torch.randn_like(p_mu)
                decode_input = torch.cat([s_sample, h], dim=-1)
                predicted_obs = self.world_model.obs_decoder(decode_input)

                if self.preferred_obs is not None:
                    pragmatic = -F.mse_loss(
                        predicted_obs, self.preferred_obs.unsqueeze(0).expand_as(predicted_obs),
                        reduction='none'
                    ).sum(dim=-1)
                else:
                    pragmatic = torch.zeros(n_candidates)

                # Epistemic value: entropy of predictions (uncertainty)
                epistemic = 0.5 * p_logvar.sum(dim=-1)

                efe += -pragmatic - epistemic

                # Advance recurrent state
                obs_feat = self.world_model.obs_encoder(predicted_obs)
                rnn_input = torch.cat([obs_feat, action_seqs[:, t]], dim=-1)
                h = self.world_model.rnn(rnn_input, h)

        return efe

    def learn(self, obs_batch: np.ndarray, action_batch: np.ndarray):
        """Update the world model from a batch of experience."""
        obs_tensor = torch.tensor(obs_batch, dtype=torch.float32)
        action_tensor = torch.tensor(action_batch, dtype=torch.float32)

        self.optimizer.zero_grad()
        free_energy, complexity, accuracy = self.compute_world_model_loss(
            obs_tensor, action_tensor
        )
        free_energy.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 100.0)
        self.optimizer.step()

        return {
            'free_energy': free_energy.item(),
            'complexity': complexity.item(),
            'accuracy': accuracy.item(),
        }
```

### Usage Example

```python
# Create agent
agent = DeepActiveInferenceAgent(
    obs_dim=64,
    action_dim=4,
    state_dim=32,
    planning_horizon=15,
)

# Set prior preferences (desired observations)
target_obs = np.zeros(64)
target_obs[:3] = [1.0, 0.0, 0.5]  # Target position
agent.set_preferences(target_obs)

# Interaction loop
obs = env.reset()
for step in range(1000):
    # Plan actions (minimize expected free energy)
    action = agent.plan_actions(obs)

    # Execute action
    next_obs, reward, done, info = env.step(action)

    # Store experience and learn
    # (In practice, use a replay buffer)
    metrics = agent.learn(
        obs_batch=np.array(obs, next_obs),
        action_batch=np.array(action, np.zeros_like(action))
    )

    print(f"Step {step}: F={metrics['free_energy']:.4f}, "
          f"Complexity={metrics['complexity']:.4f}, "
          f"Accuracy={metrics['accuracy']:.4f}")

    obs = next_obs
    if done:
        obs = env.reset()
```

## Best Practices

### Model Design

1. **Start with simple generative models** before scaling to deep architectures. Validate on toy problems first.
2. **Use beta-VAE weighting** ($\beta \cdot D_{KL}$) to control the complexity-accuracy trade-off. Higher $\beta$ encourages disentangled representations.
3. **Hierarchical latent spaces** capture multi-scale structure (fast sensory dynamics at lower levels, slow abstract dynamics at higher levels).
4. **Precision learning** (learned variance on decoder outputs) is critical for balancing different sensory modalities.

### Training Stability

1. **KL annealing**: Gradually increase the weight on the complexity term to avoid posterior collapse (where $q(z|x) \approx p(z)$ and the latent code is ignored).
2. **Gradient clipping**: Essential for world models with recurrent components. Clip at 100.0 as a reasonable default.
3. **Warm-up the world model** before using it for planning. A poorly trained model leads to degenerate action selection.
4. **Separate learning rates** for encoder and decoder can help when one converges faster.

### Planning and Action Selection

1. **Cross-Entropy Method (CEM)** is simple and effective for action optimization in continuous spaces.
2. **Monte Carlo Tree Search (MCTS)** is preferable for discrete action spaces.
3. **Shorter planning horizons** (5-15 steps) are often sufficient and more computationally tractable.
4. **Ensemble world models** provide better epistemic uncertainty estimates for exploration.

### Scaling Considerations

1. **Amortized inference** is essential for real-time operation. Per-observation optimization is too slow.
2. **Convolutional encoders/decoders** for image observations; recurrent or transformer cores for temporal structure.
3. **Mixed-precision training** (fp16) significantly reduces memory and computation with minimal accuracy loss.
4. **Distributed experience collection** with centralized model training scales to complex environments.

## References

1. Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes. *ICLR*.
2. Ueltzhoeffer, K. (2018). Deep active inference. *Biological Cybernetics*, 112(6), 547-573.
3. Catal, O., Nauta, J., Verbelen, T., Simoens, P., & Dhoedt, B. (2020). Learning perception and planning with deep active inference. *ICASSP*.
4. Millidge, B. (2020). Deep active inference as variational policy gradients. *Journal of Mathematical Psychology*, 96, 102348.
5. Fountas, Z., Sajid, N., Mediano, P. A. M., & Friston, K. (2020). Deep active inference agents using Monte-Carlo methods. *NeurIPS*.
6. Tschantz, A., Millidge, B., Seth, A. K., & Buckley, C. L. (2020). Reinforcement learning through active inference. *arXiv preprint* arXiv:2002.12636.
7. Hafner, D., Lillicrap, T., Fischer, I., Villegas, R., Ha, D., Lee, H., & Davidson, J. (2019). Learning latent dynamics for planning from pixels. *ICML*.
8. Mazzaglia, P., Verbelen, T., Catal, O., & Dhoedt, B. (2022). The free energy principle for perception and action: a deep learning perspective. *Entropy*, 24(2), 301.
9. Lanillos, P., et al. (2021). Active inference in robotics and artificial agents: Survey and challenges. *arXiv preprint* arXiv:2112.01871.
10. Van de Maele, T., Verbelen, T., Catal, O., De Boom, C., & Dhoedt, B. (2022). Causal reasoning in the brain and beyond: Active inference, causality, and the free energy principle. *Neuroscience & Biobehavioral Reviews*.

## See Also

- [[python_framework|Python Framework]] -- foundational discrete and continuous implementations
- [[robotics|Robotics Implementations]] -- deploying neural network models on physical robots
- [[simulation|Simulation Environments]] -- environments for training and testing deep active inference
- [[benchmarking|Benchmarking]] -- evaluating neural network active inference against baselines
- [[knowledge_base/free_energy_principle/mathematics/variational_free_energy|Variational Free Energy]] -- mathematical foundations of the ELBO
- [[knowledge_base/free_energy_principle/mathematics/core_principle|Core Principle]] -- the overarching free energy principle
- [[knowledge_base/free_energy_principle/cognitive/perception|Perception]] -- perceptual inference and the recognition model
- [[knowledge_base/free_energy_principle/cognitive/learning|Learning]] -- learning as free energy minimization over parameters
