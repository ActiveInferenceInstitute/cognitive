---
title: Active Sensing
description: The selection of sensing actions, such as eye movements, that are chosen to reduce uncertainty about the world, and how this is formalised in information-theoretic and Active Inference accounts.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - perception
  - information_gain
  - sensing
semantic_relations:
  - type: relates
    links:
      - "[[epistemic_foraging]]"
      - "[[information_gain]]"
      - "[[eye_movements]]"
      - "[[expected_free_energy]]"
      - "[[epistemic_value]]"
---

# Active Sensing

## Overview
Active sensing is the control of sensors so that the data collected are as informative as possible. An agent does not receive observations passively; it chooses where to look, touch or listen, and thereby shapes the evidence available for inference. Bajcsy (1988) framed this as active perception in robotics: sensing is a goal-directed process of choosing viewpoints and measurements, not a fixed front end. Biological examples include saccades, whisking in rodents and electrolocation in weakly electric fish.

## Formulation
Let $q(s)$ be the current belief over a hidden state $s$, and let an action $a$ determine which observation channel $o$ is sampled. A common criterion is the expected information gain
$$
I(s;o\mid a)=\mathbb{E}_{p(o\mid a)}\!\left[D_{\mathrm{KL}}\big[q(s\mid o,a)\,\|\,q(s)\big]\right]=H[q(s)]-\mathbb{E}\,H[q(s\mid o,a)],
$$
and the sensing action is the one that maximises it, possibly subject to costs such as travel time or energy. The criterion is greedy when applied one step at a time; planning several sensing actions ahead is a sequential decision problem in belief space. In visual search, models of this kind based on an ideal Bayesian searcher have been compared with human fixation choices (Najemnik & Geisler, 2005).

## Relation to Active Inference
In Active Inference the connection is a derivation within the framework. The expected free energy of a policy contains an epistemic term equal to the expected information gain about hidden states, so policies that sample salient, uncertainty-resolving outcomes are favoured. Friston and colleagues (2012) presented saccades as experiments that test hypotheses about the visual scene, with the epistemic term driving fixation toward ambiguous locations. This account is a specific modelling proposal; its quantitative fit to eye-movement data is still an active research question, and it sits alongside other models of fixation selection.

## Related concepts
- [[epistemic_foraging]] — information-seeking behaviour as foraging
- [[information_gain]] — the quantity maximised by sensing actions
- [[eye_movements]] — the main biological example
- [[visual_search]] — task used to compare models of fixation choice
- [[belief_space_planning]] — planning several sensing steps ahead
- [[expected_free_energy]] — the objective that contains the epistemic term

## References
- Bajcsy, R. (1988). Active perception. Proceedings of the IEEE, 76, 966-1005.
- Najemnik, J., & Geisler, W. S. (2005). Optimal eye movement strategies in visual search. Nature, 434, 387-391.
- Friston, K., Adams, R. A., Perrinet, L., & Breakspear, M. (2012). Perceptions as hypotheses: Saccades as experiments. Frontiers in Psychology, 3, 151.
