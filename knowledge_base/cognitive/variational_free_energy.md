---
title: Variational Free Energy
type: concept
status: stable
created: 2024-01-01
tags:
  - active-inference
  - variational-inference
  - free-energy
  - bayesian-inference
semantic_relations:
  - type: relates
    links:
      - "[[free_energy_principle]]"
      - "[[variational_inference]]"
      - "[[active_inference]]"
      - "[[bayesian_brain_hypothesis]]"
  - type: foundation
    links:
      - "[[../mathematics/variational_free_energy]]"
      - "[[../mathematics/variational_inference]]"
---

# Variational Free Energy

> **Code on this page.** The Python snippets illustrate the ideas. Unless a snippet imports the
> `cognitive` package, it is not part of the package and is not run by its tests, so treat it as a
> sketch. The tested implementations are listed in [[docs/api/api_reference|the API reference]].

Variational free energy is a key quantity in active inference and the free energy principle, serving as both a measure of model evidence and an objective function for perception and action. It provides a unified framework for understanding Bayesian inference in biological systems.

## Mathematical Foundations

### Definition

The variational free energy F is defined as:

F(θ,μ) = E_{q(θ)}[ln q(θ) - ln p(y,θ)] = E_{q(θ)}[-ln p(y,θ)] - H[q(θ)]

Where:

- θ represents latent variables (hidden states)
- μ represents variational parameters
- y represents observed data
- q(θ) is the variational approximation to the true posterior
- H[q(θ)] is the entropy of the variational distribution

```python
import numpy as np

# Illustrative model: theta ~ N(0, 1), y | theta ~ N(theta, 1); variational q(theta) = N(m, s2)
def vfe(y, m, s2):
    """F = E_q[-ln p(y, theta)] - H[q], in closed form for this Gaussian model."""
    energy = np.log(2 * np.pi) + 0.5 * ((y - m) ** 2 + s2) + 0.5 * (m ** 2 + s2)
    entropy = 0.5 * np.log(2 * np.pi * np.e * s2)
    return energy - entropy

y = 1.0
log_evidence = -0.5 * np.log(2 * np.pi * 2.0) - y ** 2 / (2 * 2.0)   # p(y) = N(0, 2)
print("F at the exact posterior (y/2, 1/2):", vfe(y, y / 2, 0.5), " -ln p(y):", -log_evidence)
print("F at a poorer q:", vfe(y, 0.0, 1.0))                          # F >= -ln p(y) for every q
```

### Relationship to Evidence Lower Bound

Variational free energy provides a lower bound on model evidence:

ln p(y) ≥ E_{q(θ)}[ln p(y,θ)] - E_{q(θ)}[ln q(θ)] = -F

Because the bound is tractable, minimizing F gives an approximation to maximum-likelihood (model evidence) inference.

## Perception as Inference

### Predictive Coding

Variational free energy minimization through predictive coding:

```python
# Predictive coding: gradient descent on F for y = W mu + noise with a Gaussian prior on mu
W = np.array([[1.0, 0.5], [0.0, 1.0], [0.5, 0.5]])                   # illustrative generative weights
y_obs = np.array([1.0, 0.5, 0.8])
pi_y, pi_mu = 4.0, 1.0                                              # sensory and prior precisions
mu = np.zeros(2)
for step in range(200):
    eps_y = y_obs - W @ mu                                          # sensory prediction error
    eps_mu = mu                                                     # prior prediction error (prior mean 0)
    F = 0.5 * (pi_y * eps_y @ eps_y + pi_mu * eps_mu @ eps_mu)
    mu = mu + 0.05 * (pi_y * W.T @ eps_y - pi_mu * eps_mu)          # mu <- mu - lr * dF/dmu
    if step % 50 == 0:
        print(step, round(F, 4))
print("mu:", mu, "exact MAP:", np.linalg.solve(pi_y * W.T @ W + pi_mu * np.eye(2), pi_y * W.T @ y_obs))
```

## Action as Inference

### Expected Free Energy

Action selection through expected free energy minimization:

```python
def policy_posterior(G, gamma=2.0):
    """pi ~ softmax(-gamma * G) over policies with expected free energies G."""
    p = np.exp(-gamma * (G - G.min()))
    return p / p.sum()

# G = risk + ambiguity for each policy in a small model (A[o, s], B[a][s', s], log-preferences)
A = np.array([[0.9, 0.1], [0.1, 0.9]])
B = [np.eye(2), np.array([[0.1, 0.1], [0.9, 0.9]])]
log_C = np.log(np.array([0.2, 0.8]))
q = np.array([0.7, 0.3])
G = []
for Ba in B:
    qo = A @ (Ba @ q)
    ambiguity = (Ba @ q) @ (-np.sum(A * np.log(A), axis=0))
    G.append(np.sum(qo * (np.log(qo) - log_C)) + ambiguity)
print(np.round(G, 3), np.round(policy_posterior(np.array(G)), 3))
```

## Learning and Adaptation

### Variational Learning

Model parameter learning through variational free energy minimization:

```python
# Variational EM for y = w * theta + noise, theta ~ N(0, 1), noise variance sigma2 (known)
rng = np.random.default_rng(0)
w_true, sigma2 = 2.0, 0.25
y = w_true * rng.standard_normal(2000) + np.sqrt(sigma2) * rng.standard_normal(2000)
w = 0.5
for _ in range(100):
    s2 = 1.0 / (1.0 + w ** 2 / sigma2)                  # E-step: q(theta_i) = N(m_i, s2)
    m = s2 * w * y / sigma2
    w = np.sum(m * y) / np.sum(m ** 2 + s2)             # M-step: minimise F with respect to w
print("estimated |w|:", abs(w), "moment estimate:", np.sqrt(y.var() - sigma2))
```

## Information Geometry

### Fisher Information and Natural Gradients

Using the Fisher information matrix for efficient optimization:

```python
# Natural gradient on the Gaussian family q = N(m, s2): the Fisher information for m is 1 / s2
def grad_m(y, m, s2, eps=1e-6):
    return (vfe(y, m + eps, s2) - vfe(y, m - eps, s2)) / (2 * eps)

y, s2 = 1.0, 0.5
m_plain, m_nat = -3.0, -3.0
for _ in range(5):
    m_plain -= 0.1 * grad_m(y, m_plain, s2)                       # plain gradient step
    m_nat -= 1.0 * s2 * grad_m(y, m_nat, s2)                      # natural step: Fisher^{-1} * gradient
print("after 5 steps (optimum m = 0.5):", m_plain, m_nat)         # here the natural step lands on the optimum
```

## Neural Implementation

### Synaptic Plasticity

Synaptic changes that implement variational free energy minimization:

```python
# Synaptic plasticity as a local gradient: dF/dW = -pi * outer(error, pre)
rng = np.random.default_rng(1)
W_true = rng.standard_normal((2, 3))
W = np.zeros((2, 3))
lr, pi_y = 0.05, 1.0
for _ in range(2000):
    pre = rng.standard_normal(3)                                  # presynaptic activity
    post = W_true @ pre + 0.1 * rng.standard_normal(2)                                          # noisy postsynaptic target
    error = post - W @ pre                                        # prediction error
    W += lr * pi_y * np.outer(error, pre)                         # delta = lr * pi * error * pre
print("max |W - W_true|:", round(np.abs(W - W_true).max(), 3))
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
# Scalable inference: stochastic natural-gradient VI with minibatches (Gaussian-mean model)
rng = np.random.default_rng(2)
data = 3.0 + rng.standard_normal(100_000)                         # y_i ~ N(mu, 1), prior mu ~ N(0, 10)
N, B = len(data), 256
eta0 = np.array([0.0, 1 / 10])                                    # prior natural parameters (prec * mean, prec)
lam = eta0.copy()                                                 # q(mu) natural parameters
for t in range(1, 2001):
    y = rng.choice(data, B)
    lam_hat = eta0 + (N / B) * np.array([y.sum(), B])             # estimate as if the batch were the full data
    rho = (t + 1) ** -0.7                                         # decaying step size
    lam = (1 - rho) * lam + rho * lam_hat
print("posterior mean:", lam[0] / lam[1], "full-data mean:", data.mean())
```

---

## Related Concepts

- [[free_energy_principle]] - Foundational principle
- [[active_inference]] - Action-oriented inference
- [[predictive_coding]] - Neural implementation
- [[variational_inference]] - General inference framework
- [[bayesian_inference]] - Probabilistic reasoning

## See also

- [[knowledge_base/free_energy_principle/mathematics/variational_free_energy|Variational Free Energy]]
- [[knowledge_base/mathematics/variational_free_energy|Variational Free Energy]]

## References

- Friston, K. (2010). The free-energy principle: A unified brain theory? Nature Reviews Neuroscience, 11(2), 127-138.
- Buckley, C. L., Kim, C. S., McGregor, S., & Seth, A. K. (2017). The free energy principle for action and perception: A mathematical review. Journal of Mathematical Psychology, 81, 55-79.
- Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational inference: A review for statisticians. Journal of the American Statistical Association, 112(518), 859-877.
- Amari, S. (1998). Natural gradient works efficiently in learning. Neural Computation, 10(2), 251-276.
- Parr, T., Pezzulo, G., & Friston, K. J. (2022). Active Inference: The Free Energy Principle in Mind, Brain, and Behavior. MIT Press.
