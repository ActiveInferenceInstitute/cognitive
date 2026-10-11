---
title: Chain Rule of Probability
description: The identity factorising a joint distribution into a product of conditionals, with its information-theoretic counterparts for entropy and mutual information and its use in graphical models and sequential data.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - probability_theory
  - information_theory
  - factorisation
semantic_relations:
  - type: relates
    links:
      - "[[conditional_probability]]"
      - "[[bayes_theorem]]"
      - "[[probability_theory]]"
      - "[[joint_entropy]]"
      - "[[mutual_information]]"
      - "[[information_gain]]"
---

# Chain Rule of Probability

## Overview
The chain rule of probability, also called the product rule or general multiplication rule, expresses a joint distribution as a product of conditional distributions. It follows directly from the definition of conditional probability and holds for any joint distribution, with no independence assumptions. The name is unrelated to the chain rule of calculus.

## Statement
For two events, $P(A\cap B)=P(A\mid B)\,P(B)$. For $n$ random variables, applying this repeatedly gives
$$P(X_1,\dots,X_n)=\prod_{i=1}^{n}P\big(X_i\mid X_1,\dots,X_{i-1}\big).$$
The ordering of variables is arbitrary, so the same joint distribution can be factorised in $n!$ ways. Bayes' theorem follows by writing $P(A,B)$ in both orders and dividing by $P(B)$.

## Information-theoretic versions
Taking logarithms turns the product into a sum, and expectations give chain rules for information measures:
- **Entropy**: $H(X,Y)=H(X)+H(Y\mid X)$.
- **Mutual information**: $I(X;Y,Z)=I(X;Z)+I(X;Y\mid Z)$.
- **KL divergence**: $D_{\mathrm{KL}}\big[p(x,y)\,\|\,q(x,y)\big]=D_{\mathrm{KL}}\big[p(x)\,\|\,q(x)\big]+\mathbb{E}_{p(x)}D_{\mathrm{KL}}\big[p(y\mid x)\,\|\,q(y\mid x)\big]$.

These identities are used to decompose the information gained from a sequence of observations into increments, one per observation, which is the basis of sequential experimental design (see [[information_gain]]).

## Use in models
Conditional independence assumptions simplify the factors. In a Bayesian network each variable is conditioned only on its parents, which reduces the general product to the local factorisation. For a hidden Markov model or state space model with state $s_t$ and observation $o_t$, the chain rule together with the Markov and conditional independence assumptions gives
$$P(o_{1:T},s_{1:T})=P(s_1)\,P(o_1\mid s_1)\prod_{t=2}^{T}P(s_t\mid s_{t-1})\,P(o_t\mid s_t).$$
Autoregressive language and sequence models use the unrestricted form with learned conditionals.

## Relation to Active Inference
Generative models in Active Inference are specified through factorisations of this kind, and the free energy of a model is computed by summing logarithms of the factors. Derivations of expected free energy and its decomposition into epistemic and pragmatic terms also rely on the chain rule to rewrite joint distributions over states, observations and policies. It is a basic identity of probability, not a feature unique to Active Inference.

## Related concepts
- [[conditional_probability]] — the definition from which the rule follows
- [[bayes_theorem]] — obtained by applying the rule in both orders
- [[bayesian_networks]] — factorisations with conditional independence
- [[conditional_entropy]] — the term appearing in the entropy chain rule
- [[conditional_mutual_information]] — the term in the mutual information chain rule
- [[information_gain]] — decomposed sequentially by the rule

## References
- Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory (2nd ed.). Wiley.
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
- Koller, D., & Friedman, N. (2009). Probabilistic Graphical Models: Principles and Techniques. MIT Press.
