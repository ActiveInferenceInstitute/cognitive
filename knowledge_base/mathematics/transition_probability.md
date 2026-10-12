---
title: Transition Probability
description: The probability of moving from one state to another in a stochastic process, with transition matrices, kernels, the Chapman-Kolmogorov equation and the role of the B array in Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - markov_chains
  - stochastic_processes
  - dynamics
semantic_relations:
  - type: relates
    links:
      - "[[markov_decision_process]]"
      - "[[markov_properties]]"
      - "[[stochastic_processes]]"
      - "[[kolmogorov_equations]]"
      - "[[state_space]]"
---

# Transition Probability

## Overview
A transition probability is the probability that a stochastic process, currently in state $i$, will be in state $j$ at the next step. For a time-homogeneous Markov chain on a finite state space it is written $p_{ij}=\Pr(X_{t+1}=j\mid X_t=i)$, independent of $t$ and of the earlier history. The Markov property is what makes this single number sufficient to describe the dynamics ([[markov_properties]]).

## Matrices and kernels
- **Transition matrix.** The values $p_{ij}$ form a stochastic matrix $P$ with non-negative entries and each row summing to one. The distribution after $n$ steps from an initial row vector $\mu_0$ is $\mu_n=\mu_0P^n$.
- **Chapman-Kolmogorov equation.** Multi-step probabilities satisfy $p^{(n+m)}_{ij}=\sum_kp^{(n)}_{ik}p^{(m)}_{kj}$, i.e. $P^{(n+m)}=P^{(n)}P^{(m)}$. In continuous time, the family $P(t)$ is a semigroup, $P(t+s)=P(t)P(s)$, and for a finite chain $P(t)=e^{tQ}$ where the generator $Q$ has non-negative off-diagonal entries and zero row sums.
- **Stationary distribution.** A distribution $\pi$ with $\pi P=\pi$ is stationary. An irreducible finite chain has a unique one, and if it is also aperiodic the distribution of $X_n$ converges to it (Norris, 1997).
- **General state spaces.** For continuous states, the transition probability is a kernel $K(x,A)$ giving the probability of landing in the set $A$ from $x$, and it may have a density with respect to a reference measure ([[sigma_algebras]]).
- **Master equation.** The evolution of the state distribution is governed by $\frac{d}{dt}\pi=\pi Q$, whose diffusion limit is the Fokker-Planck equation ([[kolmogorov_equations]]).

Transition probabilities can be conditioned on additional inputs. In a Markov decision process the transition depends on the chosen action, $p(s'\mid s,a)$, so there is one stochastic matrix per action ([[markov_decision_process]]). In a hidden Markov model the states are not observed directly, and observations are generated from them by a separate likelihood.

## Relation to Active Inference
In discrete-state Active Inference models, transition probabilities form the $B$ array of the generative model, one matrix $B(a)$ per action or control state. In the usual convention the entry $B_{ij}$ is the probability of state $i$ at the next time given state $j$ now, so each column sums to one, which is the transpose of the row convention above ([[B_matrix]], [[pomdp_framework]]). Learning the dynamics amounts to inferring Dirichlet parameters over the columns of $B$. Planning evaluates the consequences of action sequences by propagating beliefs forward through these matrices, so they are core ingredients of the generative model and not an approximation.

## Related concepts
- [[markov_decision_process]] — transitions that depend on actions
- [[markov_properties]] — the memoryless assumption
- [[stochastic_processes]] — processes governed by transition laws
- [[kolmogorov_equations]] — continuous-time evolution of transitions
- [[B_matrix]] — the transition array in the POMDP model
- [[state_space]] — the set of states between which transitions occur

## References
- Norris, J. R. (1997). Markov Chains. Cambridge University Press.
- Grimmett, G., & Stirzaker, D. (2001). Probability and Random Processes (3rd ed.). Oxford University Press.
