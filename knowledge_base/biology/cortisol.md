---
title: Cortisol
description: "Cortisol as the main human glucocorticoid, released through the HPA axis in response to stress and circadian cues, with its physiological effects and an allostatic Active Inference interpretation."
type: biology
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - hormone
  - stress
  - endocrine
semantic_relations:
  - type: relates
    links:
      - "[[allostatic_regulation]]"
      - "[[emotional_processing]]"
      - "[[homeostasis]]"
aliases:
  - cortisol_system
wikidata: Q190875
definition: "The chemical substance cortisol (hydrocortisone), the glucocorticoid steroid hormone produced by the adrenal cortex."
scope_note: "The mapped concept is the chemical substance, not the HPA axis, cortisol secretion, an assay, a drug product, or a particular molecule. Physiological mechanisms and Active Inference interpretations below provide context for this substance."
external_mappings:
  - target: http://purl.obolibrary.org/obo/CHEBI_17650
    relation: exactMatch
    evidence: https://www.ebi.ac.uk/chebi/CHEBI:17650
    rationale: "The page's opening identifies the steroid hormone cortisol produced by the adrenal cortex. ChEBI identifies that same chemical substance, rather than a cortisol assay, secretion process, or signaling pathway."
    reviewed_on: "2026-10-20"
  - target: http://purl.obolibrary.org/obo/NCIT_C2290
    relation: exactMatch
    evidence: https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C2290
    rationale: "NCI defines cortisol as the glucocorticoid hormone produced by the adrenal cortex; this is the substance denoted by the source page, not a particular molecule or administered dose."
    reviewed_on: "2026-10-20"
  - target: http://www.wikidata.org/entity/Q190875
    relation: exactMatch
    evidence: https://www.wikidata.org/wiki/Q190875
    rationale: "The page identifies the glucocorticoid steroid hormone cortisol, the same substance denoted by the target. The assertion concerns the concept, not identity of the Markdown document with a molecule."
    reviewed_on: "2026-10-20"
---

# Cortisol

## Overview
Cortisol is a steroid hormone produced by the adrenal cortex and the principal glucocorticoid in humans. It mobilizes energy, modulates immune activity and shapes brain function, and its release is governed by the hypothalamic-pituitary-adrenal (HPA) axis. In the emotion literature it is the standard peripheral marker of stress, which is how the cognitive pages in this knowledge base use it.

## Mechanism
The HPA axis is a cascade. Neurons in the paraventricular nucleus of the hypothalamus release corticotropin-releasing hormone; the anterior pituitary responds with adrenocorticotropic hormone (ACTH); ACTH stimulates the adrenal cortex to secrete cortisol. Cortisol then feeds back negatively on the hypothalamus, pituitary and hippocampus through glucocorticoid and mineralocorticoid receptors, which limits the response. Secretion follows a circadian rhythm, with a peak shortly after waking and a nadir around the start of sleep, with additional pulses driven by stressors.

Physiological actions include raising blood glucose through gluconeogenesis, redistributing energy stores, and suppressing inflammatory and immune responses. In the brain, cortisol acts on receptors in [[amygdala]], hippocampus and prefrontal cortex. Its effects on memory are often described as an inverted U: moderate elevations can aid consolidation of arousing events, while prolonged or very high exposure is associated with impaired retrieval and hippocampal changes in animal studies. Chronic dysregulation has been studied in relation to mood and anxiety disorders, with mixed and heterogeneous findings; none of this implies a simple causal story, and the research is ongoing.

## Role in Active Inference
The most developed link is through allostasis. Peters, McEwen and Friston argued that stress responses can be framed as the brain's attempt to resolve uncertainty about predicted bodily and environmental states, with sustained unresolved uncertainty carrying a physiological cost. On this view cortisol release is part of the allostatic machinery that reallocates resources when predictions about [[interoception]] and the environment fail ([[allostatic_regulation]], [[allostatic_control]]). This is a theoretical proposal, and quantitative models linking hormone levels to precision or prediction error are still at an early stage.

## Related concepts
- [[homeostasis]] — the regulated-variable framework within which cortisol acts
- [[emotional_processing]] — stress and arousal effects on emotion and memory
- [[anxiety]] — a condition studied in terms of HPA-axis function
- [[serotonin]] — a neuromodulator interacting with stress circuitry
- [[oxytocin]] — a hormone with partly opposing effects on stress reactivity

## References
- Peters, A., McEwen, B. S., & Friston, K. (2017). Uncertainty and stress: Why it causes diseases and how it is mastered by the brain. *Progress in Neurobiology*, 156.
