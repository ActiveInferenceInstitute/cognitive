---
definition: >-
  Adrenaline (epinephrine), the endogenous (R)-catecholamine hormone secreted by the
  adrenal medulla.
scope_note: >-
  The subject is naturally occurring adrenaline as an endogenous chemical messenger.
  The mappings exclude noradrenaline, racemic adrenaline preparations and
  therapeutic epinephrine products.
external_mappings:
  - target: http://purl.obolibrary.org/obo/CHEBI_28918
    relation: exactMatch
    evidence: https://www.ebi.ac.uk/chebi/CHEBI:28918
    rationale: >-
      The source concerns endogenous epinephrine secreted by adrenal-medullary cells
      in the fight-or-flight response. ChEBI defines (R)-adrenaline as the
      adrenaline enantiomer secreted by adrenal glands in that response. Both denote
      the natural hormone, not noradrenaline, a racemate or a therapeutic
      preparation.
    reviewed_on: "2026-10-20"
  - target: http://purl.obolibrary.org/obo/NCIT_C2292
    relation: exactMatch
    evidence: https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C2292?include=full
    rationale: >-
      The source denotes endogenous adrenaline, also named epinephrine, as an
      adrenal hormone. NCI's Epinephrine concept denotes the hormone and
      neurotransmitter and is distinct from its Therapeutic Epinephrine concept
      C473. This mapping concerns the endogenous chemical, not a manufactured
      medicine.
    reviewed_on: "2026-10-20"
title: Adrenaline
description: The hormone epinephrine released by the adrenal medulla during acute stress, its adrenergic receptor actions, the fight-or-flight response and its effects on memory and arousal.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - hormones
  - stress
  - arousal
semantic_relations:
  - type: relates
    links:
      - "[[noradrenergic_system]]"
      - "[[stress_response]]"
      - "[[cortisol]]"
      - "[[fear]]"
      - "[[amygdala]]"
      - "[[emotional_memory]]"
---

# Adrenaline

## Overview
Adrenaline, also called epinephrine, is a catecholamine produced mainly by chromaffin cells of the adrenal medulla, which are stimulated by the sympathetic nervous system. It is released within seconds of a threat or demanding situation and acts as a hormone carried in the blood. Walter Cannon described the associated syndrome as preparation for "fight or flight" (Cannon, 1915). In the brain, noradrenaline rather than adrenaline is the major catecholamine (see [[noradrenergic_system]]).

## Mechanism and effects
Adrenaline binds adrenergic receptors, which are G protein-coupled receptors of the alpha and beta families. Their distribution across tissues explains its diverse effects:
- It raises heart rate and the force of cardiac contraction through beta-1 receptors.
- It relaxes airway smooth muscle through beta-2 receptors.
- It mobilises energy by promoting breakdown of glycogen in liver and muscle and release of fatty acids.
- It redirects blood flow towards skeletal muscle and away from the gut and skin.

The response is fast and short-lived, and it works together with slower glucocorticoid signalling (see [[cortisol]], [[stress_response]]).

## Memory and arousal
Emotionally arousing events tend to be remembered better than neutral ones. Cahill and McGaugh (1998) reviewed evidence that this enhancement depends on stress hormones and on noradrenergic activity in the basolateral amygdala, which modulates consolidation in other regions. Circulating adrenaline does not readily cross the blood-brain barrier, and the proposed route involves peripheral signals, including vagal afferents, that engage central noradrenergic neurons. Excessive or prolonged arousal can impair memory and flexible thinking, so the relation between arousal and performance is not monotonic.

## Relation to Active Inference
The link is an interpretation. Active Inference accounts of emotion treat arousal as inference about bodily state (see [[interoception]]), and noradrenaline is often proposed to regulate precision or gain in sensory and cortical processing. Adrenaline's role in this framework is largely indirect, since the peripheral hormone changes the bodily signals that are interpreted, and the formal models remain hypotheses.

## Related concepts
- [[noradrenergic_system]] — the central noradrenaline systems that adrenaline signalling engages
- [[stress_response]] — the integrated physiological response to challenge
- [[fear]] — an emotional state accompanied by sympathetic arousal
- [[amygdala]] — a structure that mediates emotional modulation of memory
- [[emotional_memory]] — how arousal shapes what is remembered

## References
- Cannon, W. B. (1915). Bodily Changes in Pain, Hunger, Fear and Rage. D. Appleton and Company.
- Cahill, L., & McGaugh, J. L. (1998). Mechanisms of emotional arousal and lasting declarative memory. Trends in Neurosciences, 21, 294-299.
