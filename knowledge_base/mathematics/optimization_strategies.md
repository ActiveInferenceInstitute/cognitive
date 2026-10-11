---
title: Optimization Strategies
type: guide
status: stable
created: 2025-03-03
updated: 2026-10-09
tags:
- optimization
semantic_relations:
  - type: relates_to
    links:
      - "[[natural_gradients]]"
      - "[[variational_free_energy]]"
      - "[[expected_free_energy]]"
      - "[[worked_example_vfe]]"
---


# Optimization Strategies

Active Inference poses two optimisation problems, and they call for different tools. Perception minimises variational free energy $F[q]$ over beliefs $q(s)$. Planning minimises expected free energy $G(\pi)$ over policies $\pi$. The first is a smooth problem on a probability simplex; the second, in the discrete models of this repository, is a search over a finite set.

## Perception: first-order methods

Parameterise a categorical belief as $q = \mathrm{softmax}(\theta)$ and descend $\theta$. The gradient with respect to $q$ is $\ln q - \ln p(o, s) + 1$, and the chain rule through the softmax gives $q \odot (g - \langle q, g\rangle)$. On the two-state model of [[worked_example_vfe]], starting from the prior with step size 1, five steps reach beliefs $(0.850, 0.150)$ and $F = 0.4800$, against the optimum $F = 0.4780$ at $(0.871, 0.129)$. Convergence is linear and slows near the optimum because the softmax Jacobian shrinks as $q$ saturates.

## Natural gradients and mirror descent

Plain gradients depend on the parameterisation. The natural gradient rescales by the inverse Fisher information, which for a categorical distribution makes the step steepest descent in KL divergence rather than in Euclidean distance (see [[natural_gradients]] and [[fisher_information]]). For $F[q]$ over a categorical $q$, the natural-gradient step on the logits reduces to

$$\theta \leftarrow \theta - \eta\,\big(\ln q - \ln p(o, s)\big), \qquad q_{\text{new}} \propto q^{1-\eta}\; p(o,s)^{\eta}$$

up to an additive constant that the softmax ignores. This is mirror descent with the entropy mirror map. Because $p(o, s) \propto p(s \mid o)$, step size $\eta = 1$ lands on the exact posterior in one move; with $\eta = 0.5$ the same model gives $F = 0.5227,\ 0.4875,\ 0.4802,\ 0.4786,\ 0.4782$ over five steps, closer to the optimum than the plain gradient at every iteration.

The repository's `mean_field` belief update implements this step. In `ActiveInferenceDispatcher._mean_field_belief_update` the logits are $\ln q + \lambda\,\gamma\,(\ln p(s \mid o) - \ln q)$, with `learning_rate` $\lambda$ and precision $\gamma$, which is $q^{1-\eta}\,p(s\mid o)^{\eta}$ with $\eta = \lambda\gamma$. From the prior $(0.6, 0.4)$ with $\lambda = 0.5$, $\gamma = 1$ it returns $(0.7609, 0.2391)$, matching the value above. The `variational` update is instead a convex blend $(1-\lambda)\,q + \lambda\,p(s\mid o)$ and gives $(0.7355, 0.2645)$ for the same inputs. Both treat the current belief as the prior on each call, so applying them repeatedly to a single observation keeps adding that evidence.

## Planning: enumerate, aggregate, softmax

Policy values are not differentiable in the discrete setting, so the dispatcher enumerates action sequences (`DiscreteGenerativeModel.enumerate_policies`), evaluates each with `evaluate_policy`, aggregates by first action (`PolicyAggregation.MIN` or `MARGINAL`) and applies a softmax. See [[search_strategies]] and [[policy_search_methods]] for what to do when the $N_a^{H}$ action sequences (actions $N_a$, horizon $H$) are too many to enumerate.

## Trust regions and annealing

A trust-region view constrains each update to $D_{\mathrm{KL}}(q_{\text{new}}\,\|\,q) \le \delta$. Mirror descent with a fixed $\eta$ is the Lagrangian relaxation of that constraint. For non-convex objectives, such as free energy over several coupled factors, annealing lowers a temperature so the search starts broad and ends sharp; see [[annealing_schedules]] and [[stochastic_optimization]]. Stopping rules and step-size control are covered in [[convergence_control]].

## Choosing a method

| problem | variable | recommended | reason |
|---------|----------|-------------|--------|
| perception, one categorical factor | $q(s)$ | closed form $q \propto p(o, s)$, or mirror descent | exact or geometry-aware |
| perception, coupled factors | factorised $q$ | coordinate updates, [[message_passing]] | each factor update is a mirror step |
| planning, short horizon | policy index | enumeration | exact, cheap |
| planning, long horizon | action sequence | sampling or evolution strategies | avoids evaluating all $N_a^H$ sequences |

## Reproducing the numbers

```python
import numpy as np

A = np.array([[0.9, 0.2], [0.1, 0.8]])
D = np.array([0.6, 0.4])
ln_joint = np.log(A[0]) + np.log(D)


def free_energy(q):
    return float(np.sum(q * (np.log(q) - ln_joint)))


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


logits, q = np.log(D), D.copy()
for _ in range(5):                                   # plain gradient, step 1
    g = np.log(q) - ln_joint + 1.0
    logits = logits - q * (g - np.sum(q * g))
    q = softmax(logits)
print("plain", q.round(3), round(free_energy(q), 4))

for eta in (0.5, 1.0):                               # natural gradient / mirror
    logits, q = np.log(D), D.copy()
    values = []
    for _ in range(5):
        logits = logits - eta * (np.log(q) - ln_joint)
        q = softmax(logits)
        values.append(round(free_energy(q), 4))
    print("eta", eta, values)
```

## Links

- [[stochastic_optimization]]
- [[convergence_control]]
- [[natural_gradients]] · [[search_strategies]] · [[policy_search_methods]]

## References

- Amari, S. (1998). Natural gradient works efficiently in learning. Neural Computation, 10(2), 251-276.
- Beck, A., & Teboulle, M. (2003). Mirror descent and nonlinear projected subgradient methods for convex optimization. Operations Research Letters, 31(3), 167-175.
- Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational inference: A review for statisticians. Journal of the American Statistical Association, 112(518), 859-877.
- Parr, T., Pezzulo, G., & Friston, K. J. (2022). Active inference: The free energy principle in mind, brain, and behavior. MIT Press.
