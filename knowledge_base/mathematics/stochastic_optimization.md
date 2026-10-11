---
title: Stochastic Optimization
type: mathematical_concept
status: stable
created: 2025-03-03
updated: 2026-10-09
tags:
- optimization
- stochastic
semantic_relations:
  - type: relates_to
    links:
      - "[[langevin_dynamics]]"
      - "[[temperature_parameter]]"
      - "[[policy_search_methods]]"
---


# Stochastic Optimization

Stochastic optimisation methods inject randomness into the search so that it can leave poor local optima, handle noisy objectives, or sample from a distribution instead of finding one point. Three methods recur around Active Inference: stochastic gradient descent, Langevin dynamics and simulated annealing. They are the same idea at three temperatures.

## Stochastic gradient descent

SGD replaces the exact gradient with an unbiased estimate $\hat g$, $\mathbb{E}[\hat g] = \nabla f$, and updates $\theta \leftarrow \theta - \eta_t \hat g$. With step sizes satisfying $\sum_t \eta_t = \infty$ and $\sum_t \eta_t^2 < \infty$ it converges on convex problems; a constant step size converges only to a noise ball whose radius scales with $\eta$. In Active Inference the stochasticity comes from sampled observations, sampled particles, or minibatches of trajectories used to learn the parameters of $A$ and $B$.

## Langevin dynamics

Adding Gaussian noise of matched scale to gradient descent turns an optimiser into a sampler. The Langevin update

$$\theta_{k+1} = \theta_k - \epsilon\,\nabla U(\theta_k) + \sqrt{2\epsilon T}\;\xi_k, \qquad \xi_k \sim \mathcal{N}(0, I)$$

has stationary distribution $\propto e^{-U/T}$ as $\epsilon \to 0$ (see [[langevin_dynamics]] and [[fokker_planck]]). This is the Boltzmann form that appears in policy selection: $P(\pi) \propto e^{-G(\pi)/T}$ is the stationary law of a Langevin process on the free energy landscape at temperature $T$. For $U = \theta^2/2$ the target is $\mathcal{N}(0, T)$, and the discretised chain with step $\epsilon$ has variance $T/(1-\epsilon/2)$, a bias of order $\epsilon/2$. Simulating 5000 chains for 400 steps with $\epsilon = 0.05$ gives sample variances $0.518,\ 1.025,\ 2.042$ at $T = 0.5,\ 1,\ 2$, against stationary values $0.513,\ 1.026,\ 2.051$.

## Simulated annealing

Simulated annealing proposes a move $x \to y$ and accepts it with probability $\min\{1, e^{-(U(y) - U(x))/T}\}$, then lowers $T$ along a schedule ([[annealing_schedules]]). At high $T$ the chain crosses barriers; at low $T$ it settles into a minimum. A geometric schedule $T_{k+1} = 0.998\,T_k$ is common; convergence guarantees to the global minimum need logarithmically slow cooling, so practical schedules are heuristic. On the asymmetric double well $U(x) = (x^2-1)^2 + 0.3x$, started in the shallower well at $x = 1$ with proposal scale $0.15$ and 5000 steps, a chain held cold at $T = 0.02$ reaches the deeper well in 0 of 10 seeds, while a chain cooled from $T = 1$ at rate $0.998$ does so in 9 of 10.

## In this repository

- `ActiveInferenceDispatcher._sampling_policy_inference` draws `num_samples` first actions from $\mathrm{softmax}(-G/T)$ and returns normalised counts plus $10^{-6}$, a Monte Carlo estimate of the same Boltzmann distribution.
- `_sampling_belief_update` averages Dirichlet particles centred on the exact posterior.
- The `temperature` field of `InferenceConfig` is the $T$ above; annealing it over time is a policy-level analogue of the schedule here, but `code/tools/src` contains no scheduler for it.

## Reproducing the numbers

```python
import numpy as np

rng = np.random.default_rng(0)

# Unadjusted Langevin on U(theta) = theta^2 / 2: target N(0, T).
eps = 0.05
for T in (0.5, 1.0, 2.0):
    theta = np.zeros(5000)
    for _ in range(400):
        theta = theta - eps * theta + np.sqrt(2 * eps * T) * rng.standard_normal(theta.size)
    print("T", T, "sample var", round(float(theta.var()), 3), "stationary var", round(T / (1 - eps / 2), 3))


# Simulated annealing on an asymmetric double well (global minimum near x = -1).
def U(x):
    return (x**2 - 1) ** 2 + 0.3 * x


def anneal(T0, decay, seed, steps=5000, step=0.15):
    r = np.random.default_rng(seed)
    x, T = 1.0, T0                      # start in the shallower right-hand well
    for _ in range(steps):
        y = x + step * r.standard_normal()
        d = U(y) - U(x)
        if d < 0 or r.random() < np.exp(-d / T):
            x = y
        T *= decay
    return x


for label, T0, decay in (("cold", 0.02, 1.0), ("annealed", 1.0, 0.998)):
    ends = [anneal(T0, decay, seed) for seed in range(10)]
    print(label, "reached left well:", sum(e < 0 for e in ends), "of 10")
```

## Links

- [[annealing_schedules]]
- [[optimization_strategies]]
- [[langevin_dynamics]] · [[temperature_parameter]] · [[policy_search_methods]]

## References

- Robbins, H., & Monro, S. (1951). A stochastic approximation method. The Annals of Mathematical Statistics, 22(3), 400-407.
- Bottou, L., Curtis, F. E., & Nocedal, J. (2018). Optimization methods for large-scale machine learning. SIAM Review, 60(2), 223-311.
- Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing. Science, 220(4598), 671-680.
- Hajek, B. (1988). Cooling schedules for optimal annealing. Mathematics of Operations Research, 13(2), 311-329.
- Risken, H. (1996). The Fokker-Planck equation: Methods of solution and applications (2nd ed.). Springer.
