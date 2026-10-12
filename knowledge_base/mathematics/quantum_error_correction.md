---
title: Quantum Error Correction
description: Methods for protecting quantum information from noise by encoding logical qubits into entangled states of many physical qubits and diagnosing errors through syndrome measurements that do not reveal the encoded data.
type: mathematical_concept
status: stable
created: 2026-10-12
updated: 2026-10-12
tags:
  - quantum_computing
  - error_correction
  - stabilizer_codes
semantic_relations:
  - type: relates
    links:
      - "[[quantum_computing]]"
      - "[[quantum_information]]"
      - "[[quantum_phase_estimation]]"
      - "[[fault_tolerance]]"
      - "[[information_theory]]"
      - "[[belief_propagation]]"
---

# Quantum Error Correction

## Overview
Quantum error correction (QEC) protects quantum states from decoherence and imperfect gates. Two facts make it harder than classical error correction. A quantum state cannot be copied (the no-cloning theorem), so simple repetition is not available, and measuring a qubit to look for errors would disturb the superposition being protected. Shor (1995) and Steane (1996) showed that both obstacles can be avoided: the information is spread over an entangled state of several physical qubits, and only parity-type questions about the qubits are measured.

## A worked example: the three-qubit bit-flip code
Encode $\alpha|0\rangle+\beta|1\rangle$ as $\alpha|000\rangle+\beta|111\rangle$. Measuring the parities $Z_1Z_2$ and $Z_2Z_3$ gives a two-bit syndrome that identifies which qubit, if any, suffered an $X$ (bit-flip) error, without revealing $\alpha$ or $\beta$. Applying $X$ to that qubit repairs the state. This code does not protect against phase flips ($Z$ errors). Conjugating by Hadamard gates gives the analogous phase-flip code, and concatenating the two gives Shor's nine-qubit code, which corrects any single-qubit error.

Arbitrary single-qubit errors are handled because the Pauli operators $I, X, Y, Z$ form a basis for operators on one qubit. Correcting the discrete set $\{X, Y, Z\}$ on each qubit therefore corrects any linear combination of them, so the continuum of possible errors is reduced to a finite set.

## General conditions and stabilizer codes
A code with projector $P$ onto the code space corrects a set of error operators $\{E_a\}$ if $P E_a^\dagger E_b P = c_{ab}P$ for some Hermitian matrix $(c_{ab})$ (the Knill-Laflamme condition; see Nielsen and Chuang, 2000). Most practical codes are **stabilizer codes** (Gottesman, 1997). The code space is the joint $+1$ eigenspace of an abelian subgroup $S$ of the Pauli group that does not contain $-I$. The generators of $S$ are measured to produce the syndrome. An $[\![n,k,d]\!]$ code encodes $k$ logical qubits in $n$ physical qubits and has distance $d$, so it corrects up to $\lfloor (d-1)/2\rfloor$ arbitrary errors.

**CSS codes** (Calderbank and Shor, 1996; Steane, 1996) build a quantum code from two classical linear codes with one contained in the dual of the other. Steane's $[[7,1,3]]$ code comes from the classical $[7,4,3]$ Hamming code.

## Fault tolerance
Encoded circuits must tolerate errors in the correction step itself. The threshold theorem states that, if the physical error rate per operation is below a constant threshold that depends on the architecture and noise model, arbitrarily long quantum computations can be carried out with overhead that grows only polylogarithmically in the computation size (Nielsen and Chuang, 2000). Algorithms such as [[quantum_phase_estimation]] need coherent circuits of this depth for large instances.

## Relation to Active Inference
The connection is an analogy. Decoding a syndrome is an inference problem: from indirect, noisy observations the decoder infers the most probable hidden error and acts to restore the intended state, and for some codes decoders use message passing of the kind described in [[belief_propagation]]. This resembles perception and action under a generative model of noise, but quantum error correction is a theory of engineered devices, not a part of the Active Inference framework, and no derivation links them.

## Related concepts
- [[quantum_computing]] — the setting in which logical qubits and gates are defined
- [[quantum_information]] — entanglement and measurement used in syndromes
- [[fault_tolerance]] — the classical engineering notion of surviving component failures
- [[information_theory]] — classical coding theory, the starting point for CSS codes
- [[belief_propagation]] — message-passing decoders

## References
- Shor, P. W. (1995). Scheme for reducing decoherence in quantum computer memory. Physical Review A, 52(4), R2493-R2496. https://doi.org/10.1103/PhysRevA.52.R2493
- Steane, A. (1996). Error correcting codes in quantum theory. Physical Review Letters, 77(5), 793-797. https://doi.org/10.1103/PhysRevLett.77.793
- Calderbank, A. R., & Shor, P. W. (1996). Good quantum error-correcting codes exist. Physical Review A, 54(2), 1098-1105. https://doi.org/10.1103/PhysRevA.54.1098
- Gottesman, D. (1997). Stabilizer Codes and Quantum Error Correction (doctoral thesis, California Institute of Technology). arXiv:quant-ph/9705052.
- Nielsen, M. A., & Chuang, I. L. (2000). Quantum Computation and Quantum Information. Cambridge University Press.
