---
title: Quantum Information
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - quantum_information
  - entanglement
  - von_neumann_entropy
semantic_relations:
  - type: foundation
    links:
      - "[[quantum_mechanics]]"
      - "[[information_theory]]"
  - type: relates
    links:
      - "[[entropy]]"
      - "[[kl_divergence]]"
      - "[[quantum_computing]]"
---

# Quantum Information

## Overview
Quantum information theory studies how information is stored, transformed and transmitted in quantum systems. It generalises classical probability distributions to density operators and classical entropy to von Neumann entropy, and it identifies resources, such as entanglement, with no classical counterpart.

## Definition / Formalism
A qubit state is a density operator $\rho$ on $\mathbb C^2$; pure states satisfy $\rho=|\psi\rangle\langle\psi|$. The von Neumann entropy is

$$S(\rho)=-\operatorname{Tr}\,\rho\ln\rho=-\sum_i\lambda_i\ln\lambda_i,$$

with $\lambda_i$ the eigenvalues of $\rho$; it equals the Shannon entropy of the spectrum, is zero for pure states and equals $\ln d$ for the maximally mixed state in dimension $d$. The quantum relative entropy $S(\rho\|\sigma)=\operatorname{Tr}\,\rho(\ln\rho-\ln\sigma)$ is non-negative (Klein's inequality) and reduces to KL divergence when $\rho,\sigma$ commute.

Entanglement example: the Bell state $|\Phi^+\rangle=(|00\rangle+|11\rangle)/\sqrt2$ is pure, so $S=0$ for the pair, yet the reduced state of either qubit is $I/2$ with $S=\ln2$ (checked by partial trace); for a pure bipartite state this local entropy measures entanglement. Other basic results: the no-cloning theorem forbids copying unknown states, quantum channels are completely positive trace-preserving maps, and the Holevo bound limits the classical information extractable from $n$ qubits to at most $n$ bits. Unlike classical joint entropy, conditional quantum entropy $S(A|B)=S(AB)-S(B)$ can be negative.

## Role in Active Inference
Classical active inference uses Shannon quantities (entropy, KL divergence, mutual information) throughout, and these are the standard-case limits of the objects above. The quantum extension is research-stage: the free energy principle has been reformulated for quantum systems using reference frames and Markov blankets as interfaces (Fields, Friston, Glazebrook, Levin, 2022), and quantum-probability models of cognition use density operators for belief states. Whether such models have predictive advantages over classical ones for cognition is unsettled. Nothing in this repository's inference code depends on quantum information.

## Related concepts
- [[quantum_mechanics]] — states, observables and measurement
- [[quantum_computing]] — computation using these resources
- [[information_theory]] — the classical theory that this generalises
- [[entropy]] — Shannon entropy as the commutative special case
- [[kl_divergence]] — classical relative entropy
- [[mutual_information]] — classical correlation measure, with quantum analogue

## References
- Nielsen & Chuang (2000), Quantum Computation and Quantum Information.
- Wilde, Quantum Information Theory.
