---
title: Bio-Inspired Computing
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
      - "[[knowledge_base/cognitive/swarm_intelligence]]"
      - "[[knowledge_base/biology/systems_biology]]"
      - "[[knowledge_base/cognitive/neuromorphic_computing]]"
      - "[[knowledge_base/mathematics/evolutionary_algorithms]]"
      - "[[knowledge_base/cognitive/predictive_coding]]"
      - "[[knowledge_base/cognitive/stigmergic_coordination]]"
      - "[[knowledge_base/cognitive/collective_behavior_ants]]"
      - "[[knowledge_base/research/natural_computation]]"
---

# Bio-Inspired Computing

## Overview

Bio-inspired computing designs algorithms and hardware by borrowing mechanisms from living systems. The inspiration is usually a design heuristic: the resulting method is judged by how well it solves the engineering problem, not by fidelity to biology. The main families are listed below.

## Main families

- **Neural networks.** From the McCulloch-Pitts neuron (1943) and the perceptron to modern deep networks. Contemporary networks keep the idea of weighted units adapting through learning while differing from brains in learning rule, architecture and energy use.
- **Evolutionary computation.** Genetic algorithms (Holland), evolution strategies and genetic programming search by variation and selection over a population of candidates.
- **Swarm intelligence.** Ant colony optimisation (Dorigo) is based on pheromone trails, and particle swarm optimisation (Kennedy and Eberhart, 1995) on flocking. Both solve optimisation problems with simple agents that interact locally.
- **Artificial immune systems**, which borrow self/non-self discrimination for anomaly detection.
- **Neuromorphic hardware**, which implements spiking neurons in silicon to cut energy use for sensory processing.

## Connection to Active Inference

The relation is that of a normative framework to a collection of heuristics, and it is partly established, partly exploratory.

For neural networks, a precise result exists. Whittington and Bogacz (2017) showed that a predictive coding network, which minimises prediction errors using only local computations, can approximate the error back-propagation algorithm under certain conditions. This links a biologically plausible scheme to the dominant learning method in machine learning. The result applies to specific architectures and approximations, and whether the brain learns this way is open.

For swarm intelligence, the proposal is that collective behaviours such as pheromone-based foraging can be described as agents inferring a shared environmental state, with stigmergy serving as a communication channel. The repository implements such models as an Active Inference ant colony. This is an interpretation that matches the behaviour qualitatively. Classical ant colony optimisation remains the standard tool for combinatorial problems, and no benchmark shows the Active Inference version outperforming it.

For evolutionary algorithms, the connection is an analogy at present: selection can be read as a form of Bayesian model selection, although that reading is theoretical. Neuromorphic hardware is of interest because event-driven, local computations align with the message-passing schemes used for inference.

## In this repository

See [[knowledge_base/cognitive/neuromorphic_computing]], [[knowledge_base/mathematics/evolutionary_algorithms]], [[knowledge_base/cognitive/predictive_coding]], [[knowledge_base/cognitive/stigmergic_coordination]], [[knowledge_base/cognitive/collective_behavior_ants]] and [[knowledge_base/research/natural_computation]]. The Ant_Colony agent under `code/Things` provides a runnable example.

## References

- McCulloch, W. S., & Pitts, W. (1943). A logical calculus of the ideas immanent in nervous activity. *Bulletin of Mathematical Biophysics*, 5, 115-133.
- Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. *Proceedings of ICNN'95*.
- Whittington, J. C. R., & Bogacz, R. (2017). An approximation of the error backpropagation algorithm in a predictive coding network with local Hebbian synaptic plasticity. *Neural Computation*, 29, 1229-1262.

## Related Resources

- [[knowledge_base/cognitive/swarm_intelligence]]
- [[knowledge_base/biology/systems_biology]]
