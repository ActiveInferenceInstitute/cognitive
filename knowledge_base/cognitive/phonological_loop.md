---
title: Phonological Loop
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - working_memory
  - language
  - baddeley
semantic_relations:
  - type: relates
    links:
      - "[[working_memory]]"
      - "[[language_processing]]"
      - "[[active_inference]]"
aliases:
  - phonological_memory
---

# Phonological Loop

## Overview
The phonological loop is the component of Baddeley and Hitch's multicomponent model of working memory that temporarily holds speech-based and acoustic information. It is one of the "slave" systems supervised by the central executive, alongside the visuospatial sketchpad (and, in later versions, the episodic buffer). It accounts for the short-term retention of verbal material such as a telephone number.

## Mechanism
The loop has two parts:

- **Phonological store**: a passive store in which speech-based traces decay within roughly two seconds unless refreshed.
- **Articulatory rehearsal process**: a subvocal mechanism that re-circulates the contents, refreshing them and converting visually presented verbal items into phonological form.

Several behavioral effects motivate this structure:

- **Phonological similarity effect**: lists of similar-sounding items are recalled worse than dissimilar ones, consistent with a sound-based code.
- **Word-length effect**: recall span depends on how many items can be spoken in about two seconds, so short words are recalled better than long ones.
- **Articulatory suppression**: repeating an irrelevant word blocks rehearsal, abolishing the word-length effect and impairing recall of visually presented lists.
- **Irrelevant speech effect**: unattended speech disrupts serial recall by intruding on the store.

Neuroimaging and lesion studies associate the store with left inferior parietal cortex and rehearsal with left inferior frontal regions (including Broca's area), though the mapping is approximate. The loop also supports vocabulary learning: patients and children with poor phonological short-term memory have difficulty learning new word forms (Baddeley, Gathercole and Papagno, 1998). The model is influential but not uncontested; alternative accounts treat verbal short-term memory as activated long-term representations governed by interference rather than a dedicated store.

## Role in Active Inference
There is no developed active inference account specific to the phonological loop, and the connection should be read as indirect. Two points are reasonable. First, rehearsal can be described as the generation of predictions that re-instantiate a sequence of phonological states, so maintenance corresponds to sustained, self-generated sequence inference in a hierarchical temporal model (see [[sequence_learning]]). Second, the limited capacity in terms of time rather than item count resembles constraints on the depth of predictive sequences that can be maintained. Both are conceptual mappings, and the empirical effects listed above are explained by the existing working memory literature without needing the free-energy principle. Computational models of speech and reading by Friston and colleagues use hierarchical generative models of this kind, which is the closest relevant work.

## Related concepts
- [[working_memory]] — parent framework
- [[language_processing]] — the function the loop primarily serves
- [[sequence_learning]] — serial order retention and learning
- [[attention_mechanisms]] — central executive control of the slave systems
- [[temporal_models]] — representation of ordered events over time
- [[memory_formation]] — maintenance precedes consolidation

## References
- Baddeley, A. D., & Hitch, G. (1974). Working memory. In *The Psychology of Learning and Motivation*, Vol. 8.
- Baddeley, A., Gathercole, S., & Papagno, C. (1998). The phonological loop as a language learning device. *Psychological Review*.
