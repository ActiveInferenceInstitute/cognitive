---
title: Social Learning
type: concept
status: stable
created: 2024-01-01
tags:
  - social_cognition
  - learning
  - imitation
  - observation
aliases:
  - cultural_learning
semantic_relations:
  - type: relates
    links:
      - "[[social_cognition]]"
      - "[[learning_mechanisms]]"
      - "[[imitation]]"
      - "[[observational_learning]]"
  - type: implements
    links:
      - "[[social_cognition_detailed]]"
      - "[[collective_behavior]]"
---

# Social Learning

> **Code on this page.** The Python snippets illustrate the ideas. Unless a snippet imports the
> `cognitive` package, it is not part of the package and is not run by its tests, so treat it as a
> sketch. The tested implementations are listed in [[docs/api/api_reference|the API reference]].

Social learning encompasses the cognitive processes by which individuals acquire knowledge, skills, and behaviors through observation, imitation, and interaction with others. It enables efficient cultural transmission and adaptation beyond individual trial-and-error learning.

## Core Mechanisms

### Observational Learning

Learning through observation of others' behaviors and outcomes:

```python
import numpy as np

# Observer learns which of two actions pays off by watching a demonstrator's outcomes
rng = np.random.default_rng(0)
true_p = np.array([0.3, 0.7])                  # illustrative success probabilities (unknown to the observer)
alpha = np.ones(2)                             # Beta(1, 1) priors: successes + 1
beta = np.ones(2)                              # failures + 1
for _ in range(50):
    a = rng.integers(2)                        # action the demonstrator takes
    success = rng.random() < true_p[a]         # outcome the observer sees
    alpha[a] += success
    beta[a] += 1 - success
print("posterior mean success:", alpha / (alpha + beta))
```

### Imitation Learning

Direct reproduction of observed behaviors:

```python
import numpy as np

# Behavioural cloning: estimate P(action | state) from demonstrations (state, action) pairs
rng = np.random.default_rng(0)
n_states, n_actions = 3, 2
expert = np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]])      # illustrative demonstrator policy
demos = [(s, rng.choice(n_actions, p=expert[s])) for s in rng.integers(n_states, size=2000)]

counts = np.ones((n_states, n_actions))        # add-one smoothing
for s, a in demos:
    counts[s, a] += 1
learner = counts / counts.sum(axis=1, keepdims=True)
print(np.abs(learner - expert).max())          # small: the learner reproduces the demonstrator
```

## Neural Mechanisms

### Social Brain Network

Distributed neural systems for social cognition:

- **Superior Temporal Sulcus (STS)**: Biological motion perception
- **Fusiform Face Area (FFA)**: Face processing
- **Amygdala**: Emotional processing
- **Orbitofrontal Cortex**: Reward and social value
- **Temporal Parietal Junction (TPJ)**: Theory of mind

## Applications

### Cultural Evolution Modeling

Understanding cultural transmission dynamics:

```python
import numpy as np

# Payoff-biased copying of two cultural variants (replicator dynamics, illustrative payoffs)
payoff = np.array([1.0, 1.2])
p = 0.1                                        # initial frequency of variant 1
for _ in range(100):
    mean_payoff = (1 - p) * payoff[0] + p * payoff[1]
    p = p * payoff[1] / mean_payoff            # variants are copied in proportion to payoff
print(p)
```

---

## Related Concepts

- [[social_cognition]] - Social information processing
- [[learning_mechanisms]] - Fundamental learning processes
- [[imitation]] - Direct behavioral reproduction
- [[collective_behavior]] - Group-level social dynamics
- [[cultural_evolution]] - Cultural transmission processes

## References

- Bandura, A. (1977). Social Learning Theory. Prentice Hall.
- Rizzolatti, G., & Craighero, L. (2004). The mirror-neuron system. Annual Review of Neuroscience, 27, 169-192.
- Heyes, C. (2012). What's social about social learning? Journal of Comparative Psychology, 126(2), 193-202.
- Laland, K. N. (2004). Social learning strategies. Learning & Behavior, 32(1), 4-14.
- Boyd, R., & Richerson, P. J. (1985). Culture and the Evolutionary Process. University of Chicago Press.
