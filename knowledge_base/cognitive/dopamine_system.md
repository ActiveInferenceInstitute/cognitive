---
title: Dopamine System
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - neuromodulation
  - dopamine
  - neuroscience
aliases:
  - dopamine
semantic_relations:
  - type: relates
    links:
      - "[[active_inference]]"
      - "[[precision_weighting]]"
      - "[[reinforcement_learning]]"
---

# Dopamine System

## Overview
The dopamine system comprises the neurons that synthesize dopamine, their projections, and dopamine receptors across the brain. Although dopaminergic neurons are few in number, their widespread axonal arbors let them modulate motor control, motivation, learning, and cognition. Dysfunction is central to Parkinson's disease, and dopaminergic drugs act in schizophrenia, addiction, and attention-deficit disorders.

## Mechanism
Principal pathways:

- **Nigrostriatal**: substantia nigra pars compacta to dorsal striatum; degeneration of these neurons causes the motor symptoms of Parkinson's disease.
- **Mesolimbic**: ventral tegmental area (VTA) to nucleus accumbens and limbic structures; linked to motivation and reward.
- **Mesocortical**: VTA to prefrontal cortex; linked to working memory and executive function.
- **Tuberoinfundibular**: hypothalamus to pituitary, where dopamine inhibits prolactin release.

Dopamine acts on G-protein-coupled receptors in two families: D1-like (D1, D5), which are generally coupled to Gs and raise cAMP, and D2-like (D2, D3, D4), coupled to Gi and lowering cAMP. In the striatum, D1- and D2-expressing projection neurons are differentially modulated, forming the basis of influential models of go and no-go action selection. Firing occurs in tonic and phasic modes: slow baseline activity sets a background dopamine level, while bursts produce transient release.

The best-known functional account is that phasic firing in midbrain neurons carries a reward prediction error (Schultz, Dayan and Montague, 1997). Other proposals emphasize incentive salience ("wanting"), movement vigor, salience and novelty, and uncertainty. The field does not agree that a single function accounts for all dopamine signaling; there is also evidence for heterogeneous dopamine neuron populations with different response profiles.

## Role in Active Inference
In active inference, dopamine is most often associated with precision, not reward. Friston and colleagues proposed that phasic dopamine encodes the confidence (precision) in beliefs about policies, as a quantity that scales how strongly policy evidence is weighed in action selection (Friston et al., 2014; FitzGerald et al., 2015). Under this account, the reward prediction error behavior arises because outcomes that are better than expected increase confidence in the policy that was being followed. It also explains why dopaminergic drugs and Parkinson's disease alter action selection and the balance of exploration and exploitation. The proposal reproduces classic prediction-error responses in simulation, and has been used to model psychiatric and neurological conditions. It remains a hypothesis; the empirical case relative to the reward-prediction-error account is unresolved, and the two accounts make different predictions only in carefully designed experiments (see [[reward_processing]]).

## Related concepts
- [[precision_weighting]] — the quantity dopamine is proposed to encode
- [[reward_processing]] — functions with which dopamine is traditionally associated
- [[reinforcement_learning]] — temporal-difference error interpretation
- [[action_selection]] — stage at which policy precision acts
- [[precision_mechanisms]] — implementation of precision in neural circuits
- [[motor_learning]] — dopamine-dependent reinforcement of movement

## References
- Schultz, W., Dayan, P., & Montague, P. R. (1997). A neural substrate of prediction and reward. *Science*.
- Friston, K., Schwartenbeck, P., FitzGerald, T., Moutoussis, M., Behrens, T., & Dolan, R. J. (2014). The anatomy of choice: dopamine and decision-making. *Philosophical Transactions of the Royal Society B*.
- FitzGerald, T. H. B., Dolan, R. J., & Friston, K. (2015). Dopamine, reward learning, and active inference. *Frontiers in Computational Neuroscience*.
