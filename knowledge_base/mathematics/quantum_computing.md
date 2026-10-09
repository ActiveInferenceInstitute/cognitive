---
title: Quantum Computing
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - quantum_computing
  - quantum_circuits
  - complexity
semantic_relations:
  - type: foundation
    links:
      - "[[quantum_information]]"
      - "[[quantum_mechanics]]"
  - type: relates
    links:
      - "[[quantum_computation]]"
      - "[[computational_complexity]]"
      - "[[optimization_algorithms]]"
---

# Quantum Computing

## Overview
Quantum computing performs computation by manipulating quantum states: $n$ qubits occupy a $2^n$-dimensional state space, operations are unitary gates, and the answer is read out by measurement. Known speedups are specific to certain problem structures; it is not a general acceleration of every computation.

## Definition / Formalism
In the circuit model a computation is a sequence of unitaries on an initial state $|0\rangle^{\otimes n}$ followed by measurement in the computational basis. A universal gate set includes the Hadamard $H=\tfrac1{\sqrt2}\begin{pmatrix}1&1\\1&-1\end{pmatrix}$, a phase gate, and the two-qubit CNOT. Example: applying $H$ to the first qubit of $|00\rangle$ and then CNOT gives $(|00\rangle+|11\rangle)/\sqrt2$, the Bell state (verified by direct matrix multiplication).

Complexity landscape: BQP is the class of problems solvable by a quantum computer in polynomial time with bounded error. It contains factoring (Shor's algorithm, polynomial time versus the best known classical sub-exponential methods) and simulation of quantum dynamics. Grover's algorithm searches an unstructured list of $N$ items with $O(\sqrt N)$ queries, a quadratic speedup that is provably optimal for that task. BQP is not known to contain NP-complete problems, and it is widely believed not to. Quantum error correction is required because noise accumulates; fault-tolerant machines need many physical qubits per logical qubit. Near-term hybrid approaches, such as variational quantum eigensolvers and the quantum approximate optimisation algorithm, optimise circuit parameters with a classical optimiser against a measured cost.

## Role in Active Inference
Speculative. Proposals exist to use variational quantum circuits for Bayesian inference and free-energy minimisation (the vault's research page on quantum computation collects them), and quantum algorithms for linear algebra could in principle accelerate parts of inference. Such speedups come with substantial caveats about data loading and readout, and no demonstrated advantage for active inference workloads exists to my knowledge. The repo's agents run on classical numpy code, and nothing in the theory requires a quantum implementation.

## Related concepts
- [[quantum_information]] — entropy, entanglement and channels underlying the model
- [[quantum_mechanics]] — the physical theory being exploited
- [[quantum_computation]] — research page on quantum approaches to Active Inference
- [[computational_complexity]] — classes such as BQP, NP and PSPACE
- [[optimization_algorithms]] — classical optimisers used in variational quantum algorithms
- [[inference_algorithms]] — the classical inference tasks proposed for acceleration

## References
- Nielsen & Chuang (2000), Quantum Computation and Quantum Information.
- Shor (1994/1997), Polynomial-time algorithms for prime factorization and discrete logarithms on a quantum computer.
- Grover (1996), A fast quantum mechanical algorithm for database search.
