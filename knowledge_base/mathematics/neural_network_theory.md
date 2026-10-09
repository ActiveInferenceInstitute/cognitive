---
title: Neural Network Theory
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - neural_networks
  - universal_approximation
  - hopfield
  - backpropagation
semantic_relations:
  - type: foundation
    links:
      - "[[optimization_algorithms]]"
      - "[[automatic_differentiation]]"
  - type: relates
    links:
      - "[[adaptive_resonance_theory]]"
      - "[[predictive_coding]]"
      - "[[neural_computation]]"
---

# Neural Network Theory

## Overview
Neural network theory studies the mathematical properties of networks of simple parametrised units: what functions they can represent, how they can be trained, how they store and retrieve patterns, and how they generalise. Artificial networks are loosely inspired by neurons; the theory concerns the mathematical objects, not claims about brains.

## Definition / Formalism
A feedforward layer computes $h=\sigma(Wx+b)$ with a nonlinearity $\sigma$; a network composes such layers. **Expressivity:** the universal approximation theorems (Cybenko, 1989, for sigmoids; Hornik, 1991, more generally) state that a single hidden layer with enough units can approximate any continuous function on a compact set to arbitrary accuracy. This is an existence result and says nothing about the number of units needed or about whether training finds the weights. **Training:** parameters are fit by gradient descent on a loss, with gradients computed by backpropagation (reverse-mode differentiation; Rumelhart, Hinton and Williams, 1986).

**Attractor memory:** a Hopfield network (1982) has binary units $s_i$ and symmetric weights $w_{ij}$ with $w_{ii}=0$ and energy

$$E(s)=-\tfrac12\sum_{i,j}w_{ij}s_is_j.$$

Asynchronous updates $s_i\leftarrow\operatorname{sign}\big(\sum_jw_{ij}s_j\big)$ never increase $E$, so dynamics reach a local minimum; with Hebbian weights $w_{ij}=\frac1N\sum_\mu\xi_i^\mu\xi_j^\mu$, stored patterns are (approximately) minima. A simulation with $N=50$ and 5 random patterns shows monotone energy decrease and exact recall of a stored pattern. Reliable recall of random patterns holds up to roughly $0.14N$ stored patterns (Amit, Gutfreund and Sompolinsky, 1985). Statistical learning theory (VC dimension, margin and norm-based bounds) addresses generalisation, and classical bounds are often loose for modern overparameterised networks.

## Role in Active Inference
Neural networks enter in several distinct ways. As implementations: deep networks parametrise generative and recognition models in amortised and deep active inference. As models of brain computation: predictive-coding networks use local prediction-error updates, and under certain conditions approximate backpropagation (Whittington and Bogacz, 2017), though this correspondence is approximate and conditional. As comparison class: Hopfield energy minimisation resembles free-energy gradient descent, but the two are not the same objective. Adaptive resonance theory is a different, competitive-learning architecture with its own stability-plasticity analysis. The principle itself is not a claim that the brain is a feedforward network.

## Related concepts
- [[adaptive_resonance_theory]] — resonance-based architecture addressing stability-plasticity
- [[predictive_coding]] — error-propagating network scheme under active inference
- [[neural_computation]] — computational view of neural systems
- [[neural_networks]] — implementations of active inference with neural networks
- [[attractor_dynamics]] — Hopfield memories as attractors
- [[automatic_differentiation]] — mechanism behind backpropagation

## References
- Hopfield (1982), Neural networks and physical systems with emergent collective computational abilities, PNAS 79.
- Cybenko (1989), Approximation by superpositions of a sigmoidal function.
- Rumelhart, Hinton & Williams (1986), Learning representations by back-propagating errors, Nature 323.
- Whittington & Bogacz (2017), An approximation of the error backpropagation algorithm in a predictive coding network with local Hebbian synaptic plasticity, Neural Computation 29.
