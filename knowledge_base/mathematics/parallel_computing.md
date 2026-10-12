---
title: Parallel Computing
description: Computation performed by many processing units at once, covering shared and distributed memory, Flynn's taxonomy, Amdahl's and Gustafson's laws, and where parallelism helps Monte Carlo and inference algorithms.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - computation
  - scalability
  - algorithms
semantic_relations:
  - type: relates
    links:
      - "[[computational_complexity]]"
      - "[[time_complexity]]"
      - "[[markov_chain_monte_carlo]]"
      - "[[particle_filtering]]"
      - "[[scientific_computing]]"
wikidata: Q232661
---

# Parallel Computing

## Overview
Parallel computing divides a computation among several processing units that operate at the same time, in order to finish sooner or to handle problems too large for one unit. It differs from concurrency, which concerns the structure of tasks that may interleave, whether or not they run simultaneously.

## Models and limits
Flynn's taxonomy classifies machines by instruction and data streams: single instruction, multiple data (SIMD) covers vector units and graphics processors, and multiple instruction, multiple data (MIMD) covers multicore processors and clusters. In **shared-memory** systems, threads read and write a common address space and must synchronise to avoid conflicting updates. In **distributed-memory** systems, processes hold separate memory and exchange messages, which makes communication cost a central design factor.

Two simple laws bound the benefit. If a fraction $f$ of the work can be parallelised and the rest is serial, Amdahl's law (1967) gives the speedup on $p$ processors,
$$S(p)=\frac{1}{(1-f)+f/p}\le\frac{1}{1-f},$$
so the serial fraction limits the gain however many processors are used. Gustafson (1988) observed that problem size often grows with machine size, and for a scaled workload the speedup is $S(p)=p-(1-f')(p-1)$, where $f'$ is the parallel fraction measured on the parallel machine. The two laws answer different questions: fixed-size and scaled-size speedup. In practice, load balance, communication latency and memory bandwidth reduce speedup further.

Problems whose parts are independent, called embarrassingly parallel, scale well. Examples are running independent Markov chains or simulating many trajectories of a stochastic process. Algorithms with sequential dependence, such as a single Markov chain, parallelise less easily. Particle filters parallelise their propagation and weighting steps, whereas resampling requires coordination.

## Relation to Active Inference
Parallel computing has no theoretical link to Active Inference. It is relevant to implementation, because sampling-based estimates of path integrals and large factor-graph computations decompose into many independent evaluations. Message passing on a factor graph also exposes parallelism, since messages from different nodes can be updated simultaneously. Comparisons with parallel processing in the brain are loose analogies and do not follow from the algorithms.

## Related concepts
- [[computational_complexity]] — work and depth of algorithms
- [[time_complexity]] — sequential cost measure that parallelism reduces
- [[markov_chain_monte_carlo]] — sampling method with independent chains
- [[particle_filtering]] — ensemble method with a parallel structure
- [[scientific_computing]] — setting where large simulations run on parallel machines
- [[message_passing]] — inference scheme with parallel updates

## References
- Amdahl, G. M. (1967). Validity of the single processor approach to achieving large scale computing capabilities. AFIPS Conference Proceedings, 30, 483-485.
- Gustafson, J. L. (1988). Reevaluating Amdahl's law. Communications of the ACM, 31, 532-533.
- Hennessy, J. L., & Patterson, D. A. (2017). Computer Architecture: A Quantitative Approach (6th ed.). Morgan Kaufmann.
