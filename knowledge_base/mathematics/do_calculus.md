---
title: Do-Calculus
description: Pearl's set of three inference rules that translate interventional probabilities into observational ones using a causal diagram, with the backdoor and front-door adjustments as worked consequences.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - causal_inference
  - graphical_models
  - intervention
semantic_relations:
  - type: relates
    links:
      - "[[causal_inference]]"
      - "[[directed_acyclic_graphs]]"
      - "[[d_separation]]"
      - "[[bayesian_networks]]"
      - "[[conditional_independence]]"
---

# Do-Calculus

## Overview
The do-calculus is a set of rules, introduced by Pearl (1995), for deciding whether a causal effect can be computed from observational data given a causal diagram. The central object is the interventional distribution $P(y\mid\mathrm{do}(x))$, the distribution of $Y$ when $X$ is set to $x$ by external action. This differs from the conditional $P(y\mid x)$ because conditioning also reflects the causes that led $X$ to take that value, whereas intervening severs them. In graph terms, $\mathrm{do}(x)$ deletes the arrows into $X$.

## The rules
Using the diagram $G$ and its modifications, the three rules permit (1) insertion or deletion of an observation, (2) exchange of an action with an observation, and (3) insertion or deletion of an action, each when a specified d-separation holds in an appropriately mutilated graph. The rules are sound, and later work showed them to be complete: if a causal effect is identifiable from observational data and the diagram, some sequence of the rules derives an expression for it.

Two common consequences are:
- **Backdoor adjustment**: if a set $Z$ blocks all backdoor paths from $X$ to $Y$ and contains no descendant of $X$, then $P(y\mid\mathrm{do}(x))=\sum_z P(y\mid x,z)\,P(z)$.
- **Front-door adjustment**: if a mediator set $M$ intercepts all directed paths from $X$ to $Y$, with no unblocked backdoor path from $X$ to $M$ and with $X$ blocking all backdoor paths from $M$ to $Y$, then $P(y\mid\mathrm{do}(x))=\sum_m P(m\mid x)\sum_{x'}P(y\mid m,x')\,P(x')$, even when a confounder of $X$ and $Y$ is unobserved.

## Relation to Active Inference
The link is an analogy with some formal overlap. An agent's action is an intervention on its environment, since it overrides the process that would otherwise have determined the affected variables, and policy evaluation in Active Inference predicts outcomes under hypothetical actions, which is an interventional query. The standard Active Inference formalism does not use the do-operator or identifiability analysis, and its generative models are normally assumed to be correctly specified; work connecting the two covers interpreting policies as interventions and learning causal structure, but remains a developing area rather than a settled equivalence.

## Related concepts
- [[causal_inference]] — the field in which the calculus was developed
- [[directed_acyclic_graphs]] — the causal diagrams on which the rules operate
- [[d_separation]] — the criterion invoked in each rule
- [[bayesian_networks]] — the probabilistic models underlying the diagrams
- [[granger_causality]] — a predictive, non-interventional notion of causation

## References
- Pearl, J. (1995). Causal diagrams for empirical research. Biometrika, 82, 669-688.
- Pearl, J. (2009). Causality: Models, Reasoning, and Inference (2nd ed.). Cambridge University Press.
- Pearl, J., Glymour, M., & Jewell, N. P. (2016). Causal Inference in Statistics: A Primer. Wiley.
