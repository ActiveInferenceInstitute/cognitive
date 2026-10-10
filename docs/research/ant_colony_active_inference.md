---
title: Active inference for ant colony foraging (design proposal)
type: research
status: proposal
created: 2024-02-07
updated: 2026-10-10
tags:
  - active_inference
  - ant_colony
  - collective_behavior
  - proposal
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/cognitive/collective_behavior_ants]]"
      - "[[knowledge_base/cognitive/stigmergic_coordination]]"
      - "[[knowledge_base/cognitive/swarm_intelligence]]"
      - "[[knowledge_base/cognitive/active_inference]]"
      - "[[docs/api/ant_colony_api]]"
---

# Active inference for ant colony foraging (design proposal)

**Status: proposal.** Nothing on this page has been built or run at colony scale.
`code/Things/Ant_Colony/` contains documentation only (`README.md`, `AGENTS.md`,
`Ant_Colony_README.md`), the package has no environment, pheromone field, or
population class, and no foraging result exists in the repository. See
[[docs/api/ant_colony_api|the ant colony API page]]. The only executable part of
this page is the single-ant sketch below.

## Research question

Can foragers that each minimize expected free energy over a local, discrete model
reproduce trail formation through stigmergy, and does the colony-level pattern
depend on the precision or the preference parameters of the individual?

Background: [[knowledge_base/cognitive/collective_behavior_ants|ant colony behaviour]],
[[knowledge_base/cognitive/stigmergic_coordination|stigmergic coordination]],
[[knowledge_base/cognitive/swarm_intelligence|swarm intelligence]],
[[knowledge_base/cognitive/emergence_self_organization|emergence and self-organization]].

## Single-ant sketch (runs)

One ant at a fork senses whether the pheromone is `weak` or `strong` and chooses
`left` or `right`. The hidden state is which branch leads to food. This is only
the individual decision, built from the existing discrete model.

```python
import numpy as np

from cognitive import ActiveInferenceDispatcher, DiscreteGenerativeModel, InferenceConfig, ModelState

# States: food on left / food on right. Observations: pheromone weak / strong.
A = np.array([[0.4, 0.6], [0.6, 0.4]])             # A[o, s] = P(o | s)
B = np.stack([np.eye(2), np.eye(2)], axis=2)       # the food does not move
model = DiscreteGenerativeModel(
    A=A, B=B, C=np.array([0.0, 1.0]), D=np.array([0.5, 0.5]), E=np.array([0.5, 0.5])
)
config = InferenceConfig("variational", "discrete", 1, 0.5, 1.0, seed=0)
dispatcher = ActiveInferenceDispatcher(config, model)

state = ModelState(model.D.copy(), model.E.copy(), 1.0, 0.0, 0.0)
beliefs = dispatcher.dispatch_belief_update(1, state)   # observe strong pheromone
assert np.isclose(beliefs.sum(), 1.0)
```

## Proposed colony study (not built)

1. Implement a grid environment with food sources and an evaporating pheromone
   field. This requires new code outside the current package.
2. Give each ant a local model like the sketch above, with observations taken
   from the pheromone concentration at the ant's position.
3. Deposit pheromone as a side effect of action, so that the environment carries
   the memory.
4. Measure, with fixed seeds, the time to first food return, the fraction of
   ants on the shortest path, and trail persistence; vary the number of ants,
   the evaporation rate, and the agent's precision.
5. Compare against a non-inferential baseline (a classical ant colony
   optimization rule) so that any difference is attributable to the agent model.

No expected outcome is asserted. Results would be registered through the
manuscript build, which injects numbers into text rather than typing them
([[docs/manuscript/06_reproducibility|reproducibility chapter]]).

## Known limitations of the design

- Each ant's model is a small discrete POMDP, so spatial navigation must be
  encoded in the state space; the package has no spatial or hierarchical agent.
- Policy enumeration grows as `num_actions ** horizon`; keep the horizon short
  ([[docs/research/active_inference/scaling|scaling]]).

## References

- Deneubourg, J. L., Aron, S., Goss, S., and Pasteels, J. M. (1990). The
  self-organizing exploratory pattern of the Argentine ant. *Journal of Insect
  Behavior*, 3, 159-168.
- Friston, K. (2010). The free-energy principle: a unified brain theory?
  *Nature Reviews Neuroscience*, 11, 127-138.
- Ramstead, M. J. D., Kirchhoff, M. D., and Friston, K. J. (2020). A tale of two
  densities: active inference is enactive inference. *Adaptive Behavior*, 28,
  225-239.

## Related documentation

- [[code/Things/Ant_Colony/README|Ant colony folder]] and
  [[code/Things/Ant_Colony/AGENTS|its agent notes]]
- [[docs/guides/ant_colony_guide|Ant colony guide]]
- [[docs/examples/ant_colony_example|Ant colony example]]
