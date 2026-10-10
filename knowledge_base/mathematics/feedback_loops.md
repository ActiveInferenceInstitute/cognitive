---
title: Feedback Loops
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - feedback
  - systems
  - stability
  - bistability
aliases:
  - feedback
  - feedback_systems
  - negative_feedback
  - positive_feedback
semantic_relations:
  - type: foundation
    links:
      - "[[dynamical_systems]]"
      - "[[systems_theory]]"
  - type: relates
    links:
      - "[[circular_causality]]"
      - "[[homeostasis]]"
      - "[[active_inference_loop]]"
---

# Feedback Loops

## Overview
A feedback loop is a causal cycle in which a variable influences itself, directly or through other variables. Negative (balancing) loops oppose change and tend to stabilise; positive (reinforcing) loops amplify change and tend to drive growth, collapse or switching between states.

## Definition / Formalism
In a linear scalar system $\dot x=\lambda x$, the sign of $\lambda$ is the loop sign: $\lambda>0$ gives exponential growth (positive feedback), $\lambda<0$ gives exponential decay toward zero (negative feedback). In a signed causal graph the loop polarity is the product of its edge signs. Nonlinear saturation turns unbounded amplification into bistability or limit cycles.

Examples with checked eigenvalues:
- Logistic growth $\dot x=rx(1-x/K)$ combines positive feedback at low $x$ with negative feedback near capacity; its fixed point $x=K$ has eigenvalue $-r$, so it is stable, and $x=0$ has eigenvalue $+r$.
- $\dot x=x-x^3$ has a self-reinforcing linear term and a saturating cubic term; the fixed point $0$ is unstable (eigenvalue $+1$) and $x=\pm1$ are stable (eigenvalue $-2$), so the loop acts as a bistable switch.

A negative loop with delay $\tau$ can oscillate: $\dot x=-kx(t-\tau)$ is stable for $k\tau<\pi/2$ and loses stability to oscillation beyond that. Positive loops with mutual repression yield genetic toggle switches. Coupled loops generate richer behaviour, which system-dynamics methods represent as stock-and-flow diagrams and which causal-loop analysis classifies into archetypes.

## Role in Active Inference
The action-perception cycle is a closed loop: actions change the world, the world changes observations, observations update beliefs, and beliefs select actions. Free-energy minimisation operates on both arms of that loop; perceptual inference reduces prediction error by changing beliefs and action reduces it by changing the sensory input. This closure is the concrete content of the claim that agents and environments are coupled systems. Homeostasis and allostasis are the negative-feedback instances. Positive feedback matters for self-organisation, where reinforcing loops (for example stigmergic trail reinforcement in colonies) amplify fluctuations into structure. The loops themselves are standard systems theory; active inference supplies one particular account of why biological agents contain the negative ones.

## Related concepts
- [[circular_causality]] — mutual constraint between levels of a system
- [[feedback_control]] — engineered feedback with stability analysis
- [[homeostasis]] — negative feedback maintaining internal variables
- [[active_inference_loop]] — the perception-action cycle in this repo
- [[self_organization]] — amplification by positive feedback
- [[stability_analysis]] — eigenvalue and Lyapunov tests used above
