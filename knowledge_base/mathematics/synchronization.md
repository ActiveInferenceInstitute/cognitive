---
title: Synchronization
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - synchronization
  - kuramoto
  - coupled_oscillators
semantic_relations:
  - type: foundation
    links:
      - "[[dynamical_systems]]"
      - "[[nonlinear_dynamics]]"
  - type: relates
    links:
      - "[[neural_synchrony]]"
      - "[[phase_transitions]]"
      - "[[self_organization]]"
aliases:
  - synchronization_dynamics
---

# Synchronization

## Overview
Synchronization is the adjustment of the rhythms of interacting oscillators until they share a common frequency or maintain a fixed phase relation. It occurs in coupled pendulum clocks, fireflies, cardiac pacemaker cells, power grids and neural populations.

## Definition / Formalism
The Kuramoto model describes $N$ phase oscillators with natural frequencies $\omega_i$ drawn from a density $g(\omega)$:

$$\dot\theta_i=\omega_i+\frac{K}{N}\sum_{j=1}^N\sin(\theta_j-\theta_i).$$

Define the complex order parameter $r e^{i\psi}=\frac1N\sum_j e^{i\theta_j}$, where $r\in[0,1]$ measures coherence. The equations become $\dot\theta_i=\omega_i+Kr\sin(\psi-\theta_i)$, a mean-field form. For symmetric unimodal $g$ the incoherent state loses stability at

$$K_c=\frac{2}{\pi g(0)}.$$

For a Lorentzian of half-width $\gamma$, $K_c=2\gamma$ and for $K>K_c$ the stationary coherence is $r=\sqrt{1-K_c/K}$. A simulation of 4000 oscillators with $\gamma=1$, $K=3$ gives $r\approx0.573$ versus the predicted $\sqrt{1/3}\approx0.577$. The onset of coherence is a continuous transition with order parameter exponent $1/2$. Two-oscillator locking occurs when $|\omega_1-\omega_2|\le 2K'$ for the pairwise coupling $K'$ (Adler equation). Generalised synchronisation, where one system's state is a function of another's, is a weaker notion applicable to non-identical or chaotic systems.

## Role in Active Inference
Synchronisation appears in two roles. As a model of neural dynamics, oscillatory coherence is discussed in relation to message passing and attentional gain in predictive coding, though the specific mapping from synchrony to precision weighting is a hypothesis. As a model of agent interaction, Friston and Frith (2015) simulated two agents that infer each other's states with a shared generative model and showed generalised synchrony of their dynamics, interpreted as communication. This is an illustrative simulation of a mechanism, not a measurement of human communication. The Kuramoto order parameter is also a convenient collective variable for swarm and social-insect models in this vault.

## Related concepts
- [[neural_synchrony]] — synchrony in neural populations
- [[phase_transitions]] — the onset of coherence as a transition
- [[self_organization]] — collective order arising from local coupling
- [[nonlinear_dynamics]] — limit cycles and phase locking
- [[swarm_intelligence]] — collective coordination in agent groups
- [[pattern_formation]] — spatial counterpart of temporal coordination

## References
- Kuramoto (1984), Chemical Oscillations, Waves, and Turbulence.
- Strogatz (2000), From Kuramoto to Crawford: exploring the onset of synchronization in populations of coupled oscillators, Physica D 143.
