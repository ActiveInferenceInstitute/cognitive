---
title: Limits
description: The epsilon-delta definitions of limits of sequences and functions, continuity, their basic rules, and how limits underlie derivatives, integrals and the continuous-time limits used in modelling.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - analysis
  - calculus
  - continuity
semantic_relations:
  - type: relates
    links:
      - "[[calculus]]"
      - "[[real_analysis]]"
      - "[[taylor_series]]"
      - "[[convergence_analysis]]"
      - "[[differential_equations]]"
---

# Limits

## Overview
A limit describes the value that a quantity approaches as its argument or index approaches some target. Limits are the foundation of calculus: derivatives, integrals, infinite series and continuity are all defined through them. The modern definition, due to Cauchy and Weierstrass, avoids appeals to infinitesimals by stating the idea in terms of inequalities.

## Definitions
A sequence $(a_n)$ converges to $L$ if for every $\varepsilon>0$ there is an $N$ such that $|a_n-L|<\varepsilon$ for all $n\ge N$. For a function $f$, the statement $\lim_{x\to c}f(x)=L$ means that for every $\varepsilon>0$ there is a $\delta>0$ such that
$$0<|x-c|<\delta\ \Longrightarrow\ |f(x)-L|<\varepsilon.$$
The value of $f$ at $c$ itself is irrelevant. A function is continuous at $c$ when $\lim_{x\to c}f(x)=f(c)$. One-sided limits and limits at infinity are defined in the same way, and a limit, if it exists, is unique.

## Rules and examples
If $\lim f=L$ and $\lim g=M$, then limits of sums, products and, when $M\neq0$, quotients are the sums, products and quotients of the limits. The squeeze theorem says that if $g\le f\le h$ near $c$ and $g,h$ both tend to $L$, so does $f$. Standard examples are $\lim_{x\to0}\sin x/x=1$ and $\lim_{n\to\infty}(1+1/n)^n=e$.

Two limits define central objects. The derivative is
$$f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h},$$
and the Riemann integral is the limit of sums over finer partitions. Interchanging two limits is not automatically valid, which is why theorems about uniform convergence and dominated convergence are needed.

## Relation to Active Inference
Limits appear in the framework in the same ways they appear throughout applied mathematics. Continuous-time models arise as limits of discrete ones when the time step tends to zero, long-run time averages are limits, and the claim that a free energy bound becomes tight as an approximate posterior approaches the true one is a statement about a limit. No special role beyond this is claimed.

## Related concepts
- [[calculus]] — derivatives and integrals defined by limits
- [[real_analysis]] — the rigorous development of limits and continuity
- [[taylor_series]] — limits of polynomial approximations
- [[convergence_analysis]] — rates and modes of convergence of sequences and algorithms
- [[topology]] — the generalisation of limits to abstract spaces

## References
- Rudin, W. (1976). Principles of Mathematical Analysis (3rd ed.). McGraw-Hill.
- Abbott, S. (2015). Understanding Analysis (2nd ed.). Springer.
- Spivak, M. (2008). Calculus (4th ed.). Publish or Perish.
