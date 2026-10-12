---
title: Real Analysis
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - analysis
  - convergence
  - continuity
semantic_relations:
  - type: foundation
    links:
      - "[[calculus]]"
  - type: relates
    links:
      - "[[measure_theory]]"
      - "[[functional_analysis]]"
      - "[[topology]]"
wikidata: Q854531
---

# Real Analysis

## Overview
Real analysis is the rigorous theory of real numbers, sequences, limits, continuity, differentiation and integration. It supplies the precise statements behind everyday calculus and is the base on which measure theory, functional analysis and probability are built.

## Definition / Formalism
The real line is a complete ordered field: every nonempty bounded-above set has a least upper bound. Equivalently, every Cauchy sequence converges. A sequence $x_n\to x$ if for every $\varepsilon>0$ there is $N$ with $|x_n-x|<\varepsilon$ for $n\ge N$. A function $f$ is continuous at $a$ if for every $\varepsilon>0$ there is $\delta>0$ with $|x-a|<\delta\Rightarrow|f(x)-f(a)|<\varepsilon$. Key theorems:

- Bolzano-Weierstrass: every bounded real sequence has a convergent subsequence.
- Heine-Borel: in $\mathbb R^n$ a set is compact iff it is closed and bounded.
- Extreme value theorem: a continuous function on a compact set attains its minimum and maximum.
- Mean value theorem: if $f$ is continuous on $[a,b]$ and differentiable on $(a,b)$, then $f(b)-f(a)=f'(c)(b-a)$ for some $c$.
- Banach fixed-point theorem: if $T$ is a contraction ($d(Tx,Ty)\le q\,d(x,y)$, $q<1$) on a complete metric space, then $T$ has a unique fixed point and $d(x_n,x^*)\le \frac{q^n}{1-q}d(x_1,x_0)$ for the iteration $x_{n+1}=Tx_n$.

Modes of convergence for functions (pointwise, uniform, in $L^p$) differ, and interchanging limits with integrals or derivatives needs hypotheses such as uniform convergence or dominated convergence.

## Role in Active Inference
Real analysis is background rather than a component of the theory. It underwrites claims that a minimiser of free energy exists (extreme value theorem on a compact parameter set), that an iterative belief update converges (contraction arguments, when they apply), and that gradient flows are well posed (differentiability, Lipschitz continuity). In practice many repo update rules are not shown to be contractions, so convergence is checked empirically (for example by tolerance on free-energy change) rather than proved. Passing from discrete sums to continuous densities and expectations is justified by Lebesgue integration, treated in the measure-theory page.

## Related concepts
- [[calculus]] — the computational toolkit that real analysis makes rigorous
- [[measure_theory]] — extends integration to general spaces and probability
- [[functional_analysis]] — analysis on infinite-dimensional spaces such as densities
- [[convergence_analysis]] — applied convergence criteria for iterative algorithms
- [[numerical_stability]] — floating-point counterpart of limit arguments
- [[topology]] — generalises continuity and compactness
