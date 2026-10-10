---
title: Quantum Free Energy
description: "The Helmholtz free energy of a quantum state, its variational characterisation by the Gibbs state and relative entropy, and how it parallels, without replacing, variational free energy."
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - quantum
  - thermodynamics
  - free_energy
semantic_relations:
  - type: relates
    links:
      - "[[free_energy]]"
      - "[[quantum_mechanics]]"
      - "[[quantum_information]]"
      - "[[variational_free_energy]]"
---

# Quantum Free Energy

## Overview
In quantum statistical mechanics the free energy of a density operator $\rho$ with Hamiltonian $H$ at temperature $T$ is $F[\rho]=\operatorname{Tr}(\rho H)-T\,S(\rho)$, where $S(\rho)=-k_B\operatorname{Tr}\rho\ln\rho$ is the von Neumann entropy (units with $k_B=1$ below). It generalises the classical Helmholtz free energy by replacing Shannon entropy with von Neumann entropy and probability vectors with density matrices.

## Formalism
**Gibbs state.** $F$ is minimised over all states by $\rho_\beta=e^{-\beta H}/Z$, with $Z=\operatorname{Tr}e^{-\beta H}$ and $\beta=1/T$, and the minimum is $F_{\min}=-T\ln Z$.

**Relative-entropy form.** For any $\rho$,
$$F[\rho]-F[\rho_\beta]=T\,D(\rho\,\|\,\rho_\beta),\qquad D(\rho\|\sigma)=\operatorname{Tr}\rho(\ln\rho-\ln\sigma)\ge 0.$$
I verified this for a random 4-level Hamiltonian: both sides equal 3.2524 at $T=0.7$. Nonnegativity of the quantum relative entropy (Klein's inequality) then proves the minimum.

**Gibbs-Bogoliubov bound.** For trial Hamiltonian $H_0$, $F\le F_0+\langle H-H_0\rangle_0$, the quantum form of a variational bound used in mean-field theory.

**Commuting case.** If $\rho$ commutes with $H$, everything reduces to classical probabilities and $F=\sum_i p_iE_i+T\sum_ip_i\ln p_i$, the classical free energy.

## Role in Active Inference
The structural parallel is exact in form: variational free energy $F[q]=\mathbb E_q[-\ln p(o,s)]-H[q]=D_{KL}(q\,\|\,p(s\mid o))-\ln p(o)$ has the same shape as $F[\rho]$, with the surprisal $-\ln p(o,s)$ playing the role of energy and $q$ the role of the state. Quantum free energy is not used in standard Active Inference. The extension in the literature (Fields, Friston, Glazebrook and Levin, 2022) argues that the free energy principle applies to any quantum system separated by a holographic boundary from its environment; that claim is a theoretical proposal that has not been tested empirically. Quantum-like probability models of cognition use similar mathematics but do not assume quantum physics in the brain.

## Related concepts
- [[free_energy]] — classical Helmholtz and variational forms
- [[partition_function]] — $Z$ in the Gibbs state
- [[thermodynamics]] — origin of the free-energy variational principle
- [[quantum_control]] — steering states, where free energy can serve as a cost
- [[kl_divergence]] — classical counterpart of relative entropy

## References
- Fields, C., Friston, K., Glazebrook, J. F., Levin, M. (2022). A free energy principle for generic quantum systems. Progress in Biophysics and Molecular Biology.
