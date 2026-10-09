---
title: Quantum Computation
type: knowledge_base
status: stable
created: 2026-02-07
updated: 2026-10-09
tags:
  - research
  - active_inference
semantic_relations:
  - type: relates_to
    links:
      - "[[knowledge_base/mathematics/probability_theory]]"
      - "[[knowledge_base/research/quantum_systems]]"
      - "[[knowledge_base/mathematics/variational_methods]]"
      - "[[knowledge_base/mathematics/variational_inference]]"
      - "[[knowledge_base/research/quantum_cognition]]"
---

# Quantum Computation

## Overview

Quantum computation processes information using qubits, whose state is a unit vector in a two-dimensional complex Hilbert space, and unitary gates, followed by measurement. A register of $n$ qubits has a state space of dimension $2^n$. Computation is a circuit of gates; the output is a sample from the measurement distribution. Advantage over classical computers is established for specific problems, such as Shor's factoring algorithm (1994) and Grover's search with a quadratic speedup (1996), and not for computation in general.

## Hardware context

Present devices are noisy and small. Preskill (2018) named this regime NISQ, for noisy intermediate-scale quantum. Hybrid methods try to make use of such devices by delegating a short parameterised circuit to the quantum processor and optimising its parameters on a classical computer. The variational quantum eigensolver (Peruzzo et al., 2014) and the quantum approximate optimisation algorithm (Farhi, Goldstone and Gutmann, 2014) are the standard examples. A known difficulty is that for many circuit designs the gradients of the objective vanish exponentially with system size, the "barren plateau" phenomenon described by McClean and colleagues (2018).

## Inference on quantum hardware

Probabilistic inference is a natural target. Low, Yoder and Chuang (2014) described quantum rejection sampling on Bayesian networks, with a speedup in some regimes. Parameterised circuits also define generative models in their own right, since measuring them gives samples from a distribution that can be trained, as in quantum circuit Born machines. Variational methods apply to both settings, because the objective being minimised is a divergence or bound with the same form as classical variational inference.

## Connection to Active Inference

The connection is exploratory. Active Inference agents minimise variational and expected free energy, which are optimisation problems over beliefs and policies. In principle some inference sub-steps could be mapped onto quantum sampling or variational circuits. No speedup for Active Inference has been demonstrated, and several practical obstacles stand in the way: loading classical data into a quantum state can cost as much as the saving, noise limits circuit depth, and expected free energy requires evaluating many policies over many time steps. For present purposes, quantum computation is best treated as a possible future substrate and as a source of formal tools, not as a component of existing Active Inference implementations.

It is also separate from the claim that biological systems compute quantum-mechanically, and from the quantum formulation of the free energy principle covered in [[knowledge_base/research/quantum_systems]].

## In this repository

Probabilistic foundations are in [[knowledge_base/mathematics/probability_theory]]. Variational objectives are in [[knowledge_base/mathematics/variational_methods]] and [[knowledge_base/mathematics/variational_inference]]. A study path is [[active_inference_quantum_learning_path]]. For the contrast with human judgement models, see [[knowledge_base/research/quantum_cognition]].

## References

- Shor, P. W. (1994). Algorithms for quantum computation: discrete logarithms and factoring. *Proceedings of the 35th Annual Symposium on Foundations of Computer Science*.
- Grover, L. K. (1996). A fast quantum mechanical algorithm for database search. *Proceedings of STOC*.
- Preskill, J. (2018). Quantum computing in the NISQ era and beyond. *Quantum*, 2, 79.
- McClean, J. R., et al. (2018). Barren plateaus in quantum neural network training landscapes. *Nature Communications*, 9, 4812.

## Related Resources

- [[knowledge_base/mathematics/probability_theory]]
- [[knowledge_base/research/quantum_systems]]
