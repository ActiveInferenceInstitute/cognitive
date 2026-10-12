---
title: Perception-Action Cycles
description: The circular loop in which perception guides action and action changes what is perceived, from Neisser's cycle and Fuster's cortical hierarchy to its formal treatment in Active Inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - perception
  - action
  - closed_loop
semantic_relations:
  - type: relates
    links:
      - "[[circular_causality]]"
      - "[[active_inference_loop]]"
      - "[[sensorimotor_contingencies]]"
      - "[[embodied_cognition]]"
      - "[[predictive_processing]]"
---

# Perception-Action Cycles

## Overview
A perception-action cycle is the continuing loop in which an organism senses its environment, acts, and thereby alters the sensory input that it receives next. The cycle emphasises that perception and action are not separate stages in a one-way chain from stimulus to response. Action selects and generates sensory samples, and what is sensed informs subsequent action. The idea appears in ecological psychology, in cybernetics and control theory, and in cognitive neuroscience.

## Formulations
- **Neisser's perceptual cycle.** Neisser (1976) described perception as a cycle in which schemata direct exploration of the environment, exploration samples available information, and the information modifies the schemata.
- **Cortical hierarchy.** Fuster (2004) proposed that the cycle runs through the hierarchy of the cortex, with sensory areas in the posterior cortex and motor areas in the frontal cortex, and that the prefrontal cortex supports the temporal integration needed to link sensations and actions separated in time.
- **Sensorimotor accounts.** Theories of sensorimotor contingencies hold that perceptual experience depends on the lawful relations between movements and the changes in sensation that follow them ([[sensorimotor_contingencies]]).
- **Control-theoretic loops.** In engineering, feedback control uses the sensed outcome of an action to correct the next command, which gives the cycle a quantitative form ([[circular_causality]]).

## Properties
Cycles operate on several timescales, from reflexes to deliberate behaviour spanning minutes. They are closed loops, so the effects of action and perception cannot be separated by studying either alone. Experiments therefore often vary the coupling between action and feedback, for instance by delaying visual feedback of a hand movement.

## Relation to Active Inference
The perception-action cycle is a core structural feature of Active Inference. Perception is the update of beliefs about hidden states given observations, and action is the selection of behaviour that is expected to bring about preferred or informative observations. These two processes are described as minimising a single quantity, variational free energy, either by changing beliefs or by changing sensory input (Friston et al., 2010; see [[active_inference_loop]]). The formal framework therefore expresses the cycle explicitly. Whether the brain implements this scheme exactly remains an empirical question, and the framework is one of several that capture the loop.

## Related concepts
- [[circular_causality]] — mutual causal influence between organism and environment
- [[active_inference_loop]] — the perceive, infer, act cycle of the framework
- [[sensorimotor_contingencies]] — lawful dependence of sensation on movement
- [[embodied_cognition]] — cognition grounded in the body's interaction with the world
- [[predictive_processing]] — the broader family of theories in which prediction guides the cycle

## References
- Neisser, U. (1976). Cognition and Reality: Principles and Implications of Cognitive Psychology. W. H. Freeman.
- Fuster, J. M. (2004). Upper processing stages of the perception-action cycle. Trends in Cognitive Sciences, 8, 143-145.
- Friston, K. J., Daunizeau, J., Kilner, J., & Kiebel, S. J. (2010). Action and behavior: A free-energy formulation. Biological Cybernetics, 102, 227-260.
