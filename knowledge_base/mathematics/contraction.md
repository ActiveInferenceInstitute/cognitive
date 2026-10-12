---
title: Contraction
description: The contraction axiom of conditional independence, which combines two independence statements into one, its probabilistic proof, its role among the semi-graphoid axioms, and the unrelated contraction mapping sense.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - conditional_independence
  - graphoid_axioms
  - probability
semantic_relations:
  - type: relates
    links:
      - "[[conditional_independence]]"
      - "[[d_separation]]"
      - "[[markov_properties]]"
      - "[[graphical_models]]"
      - "[[probability_theory]]"
---

# Contraction

## Overview
In the theory of conditional independence, contraction is one of the axioms that any reasonable independence relation should satisfy. Writing $X\perp Y\mid Z$ for "$X$ is conditionally independent of $Y$ given $Z$", contraction states
$$(X\perp Y\mid Z)\ \text{and}\ (X\perp W\mid Y,Z)\ \Longrightarrow\ X\perp (Y,W)\mid Z.$$
It lets two separate independence facts be merged into a single, stronger one. The word also names the contraction mapping of metric spaces (a map with Lipschitz constant below one, as in the Banach fixed-point theorem) and index contraction of tensors; those senses are unrelated to this page.

## Proof for probability distributions
For any joint distribution, contraction can be checked directly. The first premise gives $p(x\mid y,z)=p(x\mid z)$, and the second gives $p(w\mid x,y,z)=p(w\mid y,z)$. Then
$$p(x,y,w\mid z)=p(w\mid x,y,z)\,p(x\mid y,z)\,p(y\mid z)=p(w\mid y,z)\,p(x\mid z)\,p(y\mid z)=p(x\mid z)\,p(y,w\mid z),$$
which is the conclusion. Together with decomposition and weak union it gives an equivalence: $X\perp(Y,W)\mid Z$ holds exactly when both $X\perp Y\mid Z$ and $X\perp W\mid (Y,Z)$ hold.

## Place among the axioms
Probabilistic conditional independence satisfies symmetry, decomposition, weak union and contraction, which together define a semi-graphoid. Adding intersection, which holds for strictly positive distributions, gives a graphoid. Pearl and collaborators studied these axioms as a candidate for a purely qualitative notion of relevance, and showed that separation in undirected graphs and d-separation in directed acyclic graphs obey them. Contraction is what allows a chain of local independencies to be assembled into a global factorisation, which is one of the steps behind the equivalence of local, pairwise and global Markov properties (the pairwise-to-local step also needs intersection).

## Relation to Active Inference
The connection is structural. Generative models in Active Inference are usually written as factor graphs or Bayesian networks, and their conditional independence structure, including Markov blankets, is reasoned about with the axioms above. Contraction is used implicitly when blanket conditions on separate variable sets are combined. It is a general fact about probability, not a claim specific to the free energy principle.

## Related concepts
- [[conditional_independence]] — the relation whose axioms include contraction
- [[d_separation]] — graphical criterion that satisfies the semi-graphoid axioms
- [[markov_properties]] — local and global properties assembled using these axioms
- [[markov_blanket]] — a set rendering a variable independent of the rest
- [[graphical_models]] — models whose structure encodes independencies

## References
- Dawid, A. P. (1979). Conditional independence in statistical theory. Journal of the Royal Statistical Society, Series B, 41, 1-31.
- Pearl, J. (1988). Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference. Morgan Kaufmann.
- Lauritzen, S. L. (1996). Graphical Models. Oxford University Press.
