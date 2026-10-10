---
title: Hyperscanning
description: Hyperscanning records brain activity from two or more interacting people at once to study inter-brain coupling during communication, cooperation and joint action, with notable interpretive pitfalls.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - social_neuroscience
  - neuroimaging
  - synchrony
semantic_relations:
  - type: relates
    links:
      - "[[social_neuroscience]]"
      - "[[neural_synchrony]]"
      - "[[cooperation]]"
      - "[[joint_attention]]"
---

# Hyperscanning

## Overview
Hyperscanning is the simultaneous recording of neural activity from multiple people engaged in a shared task. Montague et al. (2002) coined the term for linked fMRI scanners, and later work extended it to EEG, magnetoencephalography and functional near-infrared spectroscopy, which tolerate movement and face-to-face contact better than fMRI. The method treats social interaction as a two-person (or group) process and not as a single brain reacting to a social stimulus.

## Methods and findings
Inter-brain measures quantify statistical dependence between signals from different people: phase-locking value or coherence of oscillations in EEG, wavelet coherence in fNIRS, correlation of fMRI time courses, or directed measures such as Granger causality. Studies report increased inter-brain synchrony during guitar duets, cooperative games, joint attention, turn-taking conversation and teaching, with effects often located over frontal and temporoparietal regions; Dumas et al. (2010) reported it during spontaneous imitation of hand movements, in an alpha-mu network linking right centroparietal regions. Hasson and colleagues (2012) reviewed evidence for brain-to-brain coupling as a mechanism for sharing information.

Interpretive limits are considerable:
- **Common input.** Two people exposed to the same stimulus or rhythm will show correlated signals without any interaction; matching stimulus-driven activity must be separated from interaction-driven activity.
- **Spurious synchrony.** Task structure, motion, and shared event timing can inflate synchrony statistics, so permutation controls with surrogate pairs are needed (Burgess, 2013).
- **Direction of causation.** Higher synchrony may result from successful interaction as much as cause it.
- **Measurement.** fNIRS signals contain systemic physiological components, and EEG hyperscanning in free movement has artefact problems.

Findings are mostly correlational and sample sizes are often small.

## Role in Active Inference
Friston and Frith (2015) described communicating agents with a shared generative model as exhibiting generalised synchrony: each agent infers the other's hidden states, and mutual prediction drives their dynamics toward coherence. In this view, inter-brain synchrony is a signature of shared models and of effective communication, with the same mechanism applying to birdsong duets as to human dialogue. The proposal supplies a principled reason to expect synchrony, but it is a theoretical prediction that has been tested only partly, and alternative accounts of synchrony (entrainment to shared rhythms, common attention) remain viable.

## Related concepts
- [[eeg_recording]] — the most common hyperscanning modality
- [[fmri_studies]] — the original modality, with limited ecological validity
- [[communication]] — dialogue as coupled prediction between speakers
- [[economic_games]] — interactive tasks used with hyperscanning
- [[multi_agent_active_inference]] — models of coupled inference between agents

## References
- Montague, P. R., et al. (2002). Hyperscanning: simultaneous fMRI during linked social interactions. NeuroImage.
- Dumas, G., et al. (2010). Inter-brain synchronization during social interaction. PLoS ONE.
- Hasson, U., et al. (2012). Brain-to-brain coupling: a mechanism for creating and sharing a social world. Trends in Cognitive Sciences.
- Burgess, A. P. (2013). On the interpretation of synchronization in EEG hyperscanning studies: a cautionary note. Frontiers in Human Neuroscience.
- Friston, K., & Frith, C. (2015). A duet for one. Consciousness and Cognition.
