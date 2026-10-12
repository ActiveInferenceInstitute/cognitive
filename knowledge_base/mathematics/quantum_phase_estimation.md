---
title: Quantum Phase Estimation
description: A quantum algorithm that estimates the eigenphase of a unitary operator by combining controlled powers of the operator with an inverse quantum Fourier transform, and that underlies order finding and eigenvalue estimation.
type: mathematical_concept
status: stable
created: 2026-10-12
updated: 2026-10-12
tags:
  - quantum_computing
  - algorithms
  - fourier_transform
semantic_relations:
  - type: relates
    links:
      - "[[quantum_computing]]"
      - "[[quantum_information]]"
      - "[[quantum_error_correction]]"
      - "[[quantum_mechanics]]"
      - "[[linear_algebra]]"
---

# Quantum Phase Estimation

## Overview
Quantum phase estimation (QPE) is a subroutine of [[quantum_computing]]. Given a unitary operator $U$ and a state $|u\rangle$ with $U|u\rangle = e^{2\pi i \varphi}|u\rangle$ and $\varphi\in[0,1)$, it returns an approximation of $\varphi$ to a chosen number of bits. Because the eigenvalues of a unitary lie on the unit circle, the whole information content of an eigenvalue is its phase. Kitaev (1995) introduced the idea as a measurement procedure for an eigenvalue of a unitary operator, and Cleve, Ekert, Macchiavello and Mosca (1998) showed that several quantum algorithms share this structure.

## The circuit
Use a control register of $t$ qubits and a target register holding $|u\rangle$.
1. Prepare $|0\rangle^{\otimes t}|u\rangle$ and apply a Hadamard gate to every control qubit, giving $2^{-t/2}\sum_{k=0}^{2^t-1}|k\rangle|u\rangle$.
2. Apply $U^{2^{j}}$ to the target, controlled on control qubit $j$, for $j=0,\dots,t-1$. By phase kickback the state becomes
$$\frac{1}{2^{t/2}}\sum_{k=0}^{2^t-1} e^{2\pi i \varphi k}\,|k\rangle|u\rangle .$$
3. Apply the inverse quantum Fourier transform to the control register and measure it in the computational basis, obtaining an integer $m$ with $m/2^t\approx\varphi$.

If $\varphi = m/2^t$ exactly, the outcome is $m$ with certainty. Otherwise the outcome distribution peaks near $\varphi$. The nearest $t$-bit approximation is returned with probability at least $4/\pi^2$, and taking $t = n + \lceil \log_2(2 + 1/(2\epsilon)) \rceil$ qubits gives the first $n$ bits correctly with probability at least $1-\epsilon$ (Nielsen and Chuang, 2000).

## Cost
Reaching precision $2^{-t}$ uses $2^t-1$ applications of controlled-$U$ in total, so the number of applications grows as the inverse of the precision. This is the sense in which phase estimation is said to reach the Heisenberg scaling. Kitaev's variant replaces the large control register with a single control qubit that is measured repeatedly, which trades register size for more rounds.

## Where it is used
- **Order finding and factoring.** Taking $U|x\rangle = |ax \bmod N\rangle$ makes the eigenphases multiples of $1/r$, where $r$ is the multiplicative order of $a$ modulo $N$. Estimating them reveals $r$, which is the quantum core of Shor's factoring algorithm.
- **Eigenvalue estimation.** With $U = e^{-iHt}$ for a Hamiltonian $H$, the phases encode energies, which is the standard route to ground-state energies in quantum simulation.
- **Amplitude and counting problems.** Phase estimation applied to a Grover iteration estimates amplitudes.

On noisy hardware the circuit needs deep, coherent sequences of gates, which is why it is usually paired with [[quantum_error_correction]].

## Relation to Active Inference
Quantum phase estimation has no established role in Active Inference. The link is thematic only: [[quantum_free_energy]] and quantum-cognition proposals extend free-energy ideas to quantum state spaces, and any such scheme would run on the machinery described in [[quantum_computing]] and [[quantum_information]]. Phase estimation does not appear in the standard discrete or continuous formulations of the framework.

## Related concepts
- [[quantum_computing]] — the circuit model in which the algorithm is expressed
- [[quantum_information]] — states, measurement and entanglement
- [[quantum_error_correction]] — protection needed for long coherent circuits
- [[quantum_mechanics]] — unitary evolution and eigenvalue problems
- [[quantum_free_energy]] — free-energy ideas extended to quantum systems

## References
- Kitaev, A. Yu. (1995). Quantum measurements and the Abelian Stabilizer Problem. arXiv:quant-ph/9511026.
- Cleve, R., Ekert, A., Macchiavello, C., & Mosca, M. (1998). Quantum algorithms revisited. Proceedings of the Royal Society of London. Series A, 454(1969), 339-354. https://doi.org/10.1098/rspa.1998.0164
- Nielsen, M. A., & Chuang, I. L. (2000). Quantum Computation and Quantum Information. Cambridge University Press.
- Cuffaro, M., & Hagar, A. Quantum Computing. Stanford Encyclopedia of Philosophy. https://plato.stanford.edu/entries/qt-quantcomp/
