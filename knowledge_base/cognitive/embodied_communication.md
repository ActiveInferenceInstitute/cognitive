---
title: Embodied Communication
type: knowledge_base
status: stable
created: 2026-02-07
updated: 2026-10-09
tags:
  - cognitive
  - active_inference
semantic_relations:
  - type: relates_to
    links:
      - "[[knowledge_base/cognitive/enactive_cognition]]"
      - "[[knowledge_base/cognitive/active_inference]]"
      - "[[knowledge_base/cognitive/communication]]"
      - "[[knowledge_base/cognitive/social_cognition]]"
      - "[[knowledge_base/cognitive/multisensory_integration]]"
---

# Embodied Communication

## Overview

Embodied communication is information exchange carried by the body: gesture, posture, gaze, facial expression, and the pitch, timing and loudness of speech (prosody), usually in several channels at once. It contrasts with a picture of communication as encoding and decoding abstract symbols. An Active Inference reading treats a speaker's and a listener's behaviour as actions chosen to make each other's sensory input predictable. The theoretical case is developed in the literature; most computational demonstrations use toy settings and the account is an active research programme rather than a validated model of human conversation.

## Main Ideas

**Communication as shared inference.** Friston and Frith (2015) describe two agents with the same generative model who each infer the hidden state of the other. When the agents predict each other well, their dynamics synchronise, and mutual prediction replaces explicit coding. Their illustration is a pair of birds whose songs lock together. This is a formal argument with simulations, not an empirical finding about human dyads.

**Sensorimotor communication.** Pezzulo, Donnarumma and Dindo (2013) analyse movements that serve a signalling purpose in joint action, such as exaggerating a reach so an observer can tell which object is intended. In Active Inference terms the sender chooses an action whose expected free energy includes a term for the observer's resolved uncertainty, and so a movement can be epistemic for someone else ([[knowledge_base/cognitive/epistemic_foraging]]).

**Gesture and prosody.** Gesture can disambiguate speech and contribute content of its own. Prosody conveys emphasis, affect and turn-taking cues. A natural mapping is that such cues adjust the precision of the listener's predictions: stress marks which part of the message deserves high weight. This mapping is plausible and not a settled result.

**Multimodality.** Because one underlying state (a speaker's intention) generates sound, movement and facial change, listeners combine them by inference ([[knowledge_base/cognitive/multisensory_integration]]). Conflict between channels, such as a cheerful voice with a tense face, appears as raised prediction error.

## Limits

Models in this area tend to use small discrete state spaces, so they show that the logic works, not that people communicate that way. Competing accounts from pragmatics, joint-action research and enactivism cover the same phenomena ([[knowledge_base/cognitive/enactive_cognition]]).

## In This Repository

Related pages: [[knowledge_base/cognitive/communication]], [[knowledge_base/cognitive/language_processing]], [[knowledge_base/cognitive/sensorimotor_coordination]] and the non-human cases [[knowledge_base/biology/bee_communication_systems]] and [[knowledge_base/biology/ant_communication_systems]], where signalling through movement, odour and the environment ([[knowledge_base/cognitive/stigmergic_coordination]]) is documented. The code tree contains no gesture, prosody or dyadic-communication simulation; the multi-agent material under `code/Things` concerns coordination, not embodied signalling.

## References

- Friston, K., Frith, C. (2015). A duet for one. *Consciousness and Cognition*.
- Pezzulo, G., Donnarumma, F., Dindo, H. (2013). Human sensorimotor communication: a theory of signaling in online social interactions. *PLoS ONE*.

## Related Resources

- [[knowledge_base/cognitive/enactive_cognition]]
- [[knowledge_base/cognitive/active_inference]]
