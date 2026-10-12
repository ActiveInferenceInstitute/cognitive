---
title: Hyperparameter Optimization
type: guide
status: stable
created: 2025-03-03
updated: 2026-10-09
tags:
- optimization
- bayesian_optimization
semantic_relations:
  - type: relates_to
    links:
      - "[[parameter_tuning]]"
      - "[[sensitivity_analysis]]"
      - "[[stochastic_optimization]]"
aliases:
  - hyperparameters
  - hyperparameter_tuning
---


# Hyperparameter Optimization

> **Code on this page.** The Python snippets illustrate the ideas. Unless a snippet imports the
> `cognitive` package, it is not part of the package and is not run by its tests, so treat it as a
> sketch. The tested implementations are listed in [[docs/api/api_reference|the API reference]].

Model-based tuning (Bayesian optimization, bandits) for parameters like $\gamma$, learning rates, and precisions. The task is to choose settings such as the policy temperature, belief learning rate, discount factor and horizon so that an agent performs well, when each evaluation means running whole episodes and returns a noisy score.

## What makes the problem hard

The objective is expensive (an episode is many inference steps), noisy (sampled actions and observations), and has no gradient with respect to the settings. Methods therefore differ in how they spend evaluations. The score must be an outcome measure such as reward collected, not $G$ or $F$, which depend on the parameters being tuned ([[parameter_tuning]]).

## Methods

- **Gaussian-process Bayesian optimisation.** Fit a Gaussian process to the evaluations so far and choose the next point by an acquisition function such as expected improvement, which trades predicted value against uncertainty. It is sample-efficient in a handful of continuous dimensions and degrades in high dimensions or with many categorical settings.
- **Tree-structured Parzen estimators.** Model $p(x \mid \text{good})$ and $p(x \mid \text{bad})$ with kernel densities split at a quantile of the observed scores, and propose points that maximise their ratio. It handles mixed, conditional search spaces more naturally than a Gaussian process.
- **Multi-armed bandits.** Treat each candidate setting as an arm. UCB1 pulls the arm maximising mean score plus $\sqrt{2\ln n / n_i}$; Thompson sampling pulls the arm that is best under a posterior sample. Suited to a small discrete set of candidates.
- **Successive halving and Hyperband.** Give many candidates a small budget, discard the worse half, and double the budget for survivors. It assumes early scores are informative about final ones.
- **Random search.** The baseline to beat; with only a few influential parameters it is hard to beat by much.

Practical rules: search scale-like parameters ($T$, learning rate) on a log scale; fix random seeds per candidate or use common random numbers so comparisons are paired; and set the range wide enough that the optimum is not on its edge.

## An example with the repository's dispatcher

The task is the two-state model of [[worked_example_efe]]. The agent starts in the ambiguous state with belief $(0.5, 0.5)$, chooses actions from `ActiveInferenceDispatcher` (horizon 2), receives sampled outcomes, updates its belief exactly, and scores the fraction of 20 steps on which it sees the preferred outcome. Tuning the policy `temperature` over $\{0.03, 0.1, 0.3, 1, 3, 10, 30\}$ with 200 seeds per value gives mean scores $0.882,\ 0.869,\ 0.859,\ 0.853,\ 0.850,\ 0.849,\ 0.848$. The score falls as the policy gets noisier, and the spread across values is small, about 0.03.

Successive halving with 8 seeds in round one and 16 in round two selects $T = 0.03$ using 104 episodes against 1400 for the full grid. The winner sits on the lower edge of the range, so the correct next step is to extend the range, not to trust the edge. Here the agent's model matches the environment exactly, which favours greedy choice. In a task with an uncertain model a very low temperature commits early and can lock in a wrong belief; check on the target task, not this one.

Evaluation noise decides how far this can be trusted. The per-episode standard deviation is about $0.08$, so 8 seeds give a standard error near $0.03$, about as large as the whole $0.034$ spread between the best and worst temperature. Halving agreed with the 200-seed grid here, but with so few seeds it could easily have picked a neighbour; when the effect is this small, spend the budget on more seeds for fewer candidates.

## Reproducing the numbers

```python
import numpy as np

from cognitive.models.active_inference import (
    ActiveInferenceDispatcher, DiscreteGenerativeModel, InferenceConfig, ModelState)

A = np.array([[0.9, 0.5], [0.1, 0.5]])
B = np.zeros((2, 2, 2))
B[:, :, 0] = np.eye(2)
B[:, :, 1] = [[0.9, 0.9], [0.1, 0.1]]
model = DiscreteGenerativeModel(A, B, np.array([2.0, 0.0]), np.full(2, 0.5), np.full(2, 0.5))


def episode_reward(temperature, seed, steps=20):
    """Fraction of steps on which the preferred outcome 0 is observed."""
    rng = np.random.default_rng(seed)
    config = InferenceConfig(method="variational", policy_type="discrete", temporal_horizon=2,
                             learning_rate=1.0, precision_init=1.0, temperature=temperature,
                             discount_factor=1.0, seed=seed)
    dispatcher = ActiveInferenceDispatcher(config, model)
    true_state = 1                                   # start in the ambiguous state
    beliefs = np.array([0.5, 0.5])
    hits = 0
    for _ in range(steps):
        state = ModelState(beliefs=beliefs, policies=np.full(2, 0.5), precision=1.0,
                           free_energy=0.0, prediction_error=0.0)
        probs = dispatcher.dispatch_policy_inference(state)
        action = int(rng.choice(2, p=probs))
        true_state = int(rng.choice(2, p=B[:, true_state, action]))
        outcome = int(rng.choice(2, p=A[:, true_state]))
        hits += outcome == 0
        beliefs = model.posterior(outcome, model.predict_states(beliefs, action))
    return hits / steps


def score(temperature, seeds):
    return float(np.mean([episode_reward(temperature, s) for s in seeds]))


grid = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
for T in grid:
    print(T, round(score(T, range(200)), 3))   # full grid, 200 seeds each


def successive_halving(candidates, seeds_per_round=8, rounds=2):
    """Keep the better half each round, doubling the number of seeds per survivor."""
    survivors, episodes = list(candidates), 0
    for r in range(rounds):
        seeds = range(1000 + 100 * r, 1000 + 100 * r + seeds_per_round * 2**r)
        scores = {t: score(t, seeds) for t in survivors}
        episodes += len(survivors) * len(seeds)
        survivors = sorted(survivors, key=scores.get, reverse=True)[: max(1, len(survivors) // 2)]
        print("round", r, "seeds", len(seeds), "survivors", survivors)
    return survivors[0], episodes


best, episodes = successive_halving(grid)
print("best", best, "episodes", episodes, "versus", len(grid) * 200, "for the full grid")
```

## Links

- [[parameter_tuning]]
- [[sensitivity_analysis]]
- [[stochastic_optimization]] · [[temperature_parameter]]

## References

- Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian optimization of machine learning algorithms. Advances in Neural Information Processing Systems, 25.
- Bergstra, J., Bardenet, R., Bengio, Y., & Kegl, B. (2011). Algorithms for hyper-parameter optimization. Advances in Neural Information Processing Systems, 24.
- Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. Journal of Machine Learning Research, 13, 281-305.
- Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. Machine Learning, 47(2-3), 235-256.
- Jamieson, K., & Talwalkar, A. (2016). Non-stochastic best arm identification and hyperparameter optimization. Proceedings of the 19th International Conference on Artificial Intelligence and Statistics (AISTATS), PMLR 51, 240-248.
- Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2018). Hyperband: A novel bandit-based approach to hyperparameter optimization. Journal of Machine Learning Research, 18(185), 1-52.
