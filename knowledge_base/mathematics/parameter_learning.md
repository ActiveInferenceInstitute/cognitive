---
title: Parameter Learning
description: "How an Active Inference agent learns the parameters of its generative model, mainly through Dirichlet concentration updates on likelihood and transition tensors, and why this adds novelty-seeking."
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - learning
  - dirichlet
  - generative_model
semantic_relations:
  - type: relates
    links:
      - "[[generative_model]]"
      - "[[active_inference_pomdp]]"
      - "[[structure_learning]]"
      - "[[hebbian_learning]]"
---

# Parameter Learning

## Overview
Parameter learning is inference about the fixed parameters of a [[generative_model]], as opposed to inference about hidden states (perception) or policies (planning). In discrete-state Active Inference the parameters are the entries of the likelihood tensor $A$, transition tensors $B$, and initial-state prior $D$; sometimes also the preference vector $C$ and habit prior $E$.

## Mechanism
**Dirichlet priors.** Each column of a categorical tensor gets a Dirichlet prior with concentration parameters $a$, so $A_{\cdot s}\sim\mathrm{Dir}(a_{\cdot s})$. The Dirichlet is conjugate to the categorical, so the posterior is again Dirichlet.

**Update rule.** After inferring states $s_\tau$ for observations $o_\tau$, the concentrations accumulate expected co-occurrence counts:
$$a \leftarrow a+\eta\sum_\tau o_\tau\otimes \hat s_\tau,$$
with learning rate $\eta$. The point estimate is the normalised count, and states enter inference through $\mathbb E[\ln A]=\psi(a)-\psi(\textstyle\sum_o a)$, with $\psi$ the digamma function. A simulation with three outcomes and two states recovered the true likelihood to within 0.02 after 5000 samples.

**Timescale.** Parameters change slowly relative to states, giving a hierarchy of timescales: perception within a trial, learning across trials. Because the update is a product of pre- and post-synaptic quantities, it has been compared to [[hebbian_learning]] (an analogy with similar form, not an identity).

**Pruning.** Bayesian model reduction can evaluate whether reduced priors would have given higher evidence, which links parameter learning to [[structure_learning]].

## Role in Active Inference
Learning is part of free-energy minimisation: parameters are treated as slow hidden variables, so the same bound is minimised with respect to them. Because uncertainty over parameters reduces after informative observations, [[expected_free_energy]] acquires a parameter-information-gain term (often called novelty). Schwartenbeck and colleagues showed that this promotes exploration of poorly known outcomes and states; it vanishes as the concentrations grow. Parameter learning is also where the repo's approach is closest to ordinary [[maximum_likelihood]] or MAP fitting, with the Dirichlet counts as sufficient statistics.

## Related concepts
- [[generative_model]] — the object whose parameters are learned
- [[active_inference_pomdp]] — the discrete setting where the Dirichlet scheme is standard
- [[maximum_likelihood]] — the point-estimate limit
- [[epistemic_value]] — the state-level counterpart of parameter novelty
- [[bayes_theorem]] — conjugate updating

## References
- Schwartenbeck, P., Passecker, J., Hauser, T. U., FitzGerald, T. H. B., Kronbichler, M., Friston, K. J. (2019). Computational mechanisms of curiosity and goal-directed exploration. eLife.
