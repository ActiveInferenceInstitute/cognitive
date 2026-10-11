---
title: Variational Density
description: The approximate posterior distribution optimised in variational inference, its definition through free energy minimisation, mean-field and Laplace forms, and its interpretation as an agent's belief.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - variational_inference
  - free_energy
  - beliefs
semantic_relations:
  - type: relates
    links:
      - "[[variational_free_energy]]"
      - "[[variational_inference]]"
      - "[[kl_divergence]]"
      - "[[laplace_approximation]]"
      - "[[generative_model]]"
---

# Variational Density

## Overview
The variational density $q(x)$ is a probability distribution over hidden variables that is chosen to approximate the true posterior $p(x\mid o)$ when that posterior is intractable. It is restricted to a tractable family, for example a factorised or Gaussian family, and its parameters are optimised. The term is also called the approximate posterior or recognition density.

## Definition
Given a generative model $p(o,x)$ and an observation $o$, the variational free energy of $q$ is
$$F[q]=\mathbb{E}_q[\ln q(x)-\ln p(o,x)]=D_{\mathrm{KL}}\big[q(x)\,\|\,p(x\mid o)\big]-\ln p(o).$$
Since the divergence is non-negative, $F$ upper-bounds the surprisal $-\ln p(o)$, and minimising $F$ over $q$ makes $q$ as close as the chosen family allows to the true posterior. At the optimum within an unrestricted family, $q(x)=p(x\mid o)$ and $F=-\ln p(o)$.

## Common forms
- **Mean-field**: $q(x)=\prod_i q_i(x_i)$, with each factor updated given expectations of the others.
- **Laplace**: $q$ is a Gaussian centred on a mode with covariance from the curvature of the log joint (see [[laplace_approximation]]). This is standard in continuous-state Active Inference.
- **Path-space**: for dynamical systems, $q$ can be defined over a whole trajectory, written $q[x(\tau)]$, so that free energy becomes a functional of paths (see [[path_integral_theory]]).

## Interpretation in Active Inference
In the free energy principle, the variational density is parametrised by the internal states of the agent, so that internal states encode, or are said to parametrise, beliefs about external states. Perception is then the optimisation of $q$ to reduce free energy given sensory input, and its sufficient statistics are what a process theory assigns to neuronal activity. This is an interpretation built on the variational machinery: the mathematics of $q$ is the same as in other applications of variational inference, while the claim that a physical system's internal states parametrise such a density is a substantive hypothesis whose scope is still discussed. The density is a belief about causes, not necessarily a conscious or propositional one.

## Related concepts
- [[variational_free_energy]] — the functional minimised to fit $q$
- [[variational_inference]] — the general method
- [[kl_divergence]] — the divergence from $q$ to the true posterior
- [[mean_field_theory]] — the factorised approximation
- [[bayesian_inference]] — the exact computation being approximated
- [[generative_model]] — the joint model that defines the target

## References
- Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational inference: a review for statisticians. Journal of the American Statistical Association, 112, 859-877.
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
- Friston, K. (2010). The free-energy principle: a unified brain theory? Nature Reviews Neuroscience, 11, 127-138.
