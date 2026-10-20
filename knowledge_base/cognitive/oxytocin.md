---
definition: >-
  Oxytocin, the mature cyclic nonapeptide hormone and brain neuromodulator produced
  by hypothalamic neurons and released from the posterior pituitary.
scope_note: >-
  The subject is the mature nine-residue oxytocin peptide. The mappings do not
  identify it with the OXT gene, a peptide precursor, the oxytocin receptor, the
  entire oxytocinergic system or a recombinant drug preparation.
external_mappings:
  - target: http://purl.obolibrary.org/obo/CHEBI_7872
    relation: exactMatch
    evidence: https://www.ebi.ac.uk/chebi/CHEBI:7872
    rationale: >-
      The source defines oxytocin as a nine-amino-acid hormone and brain
      neuromodulator associated with uterine contraction and milk ejection. ChEBI
      defines the cyclic nonapeptide CYIQNCPLG with the same endocrine and neural
      roles. Both denote the mature peptide, not its receptor, gene or a drug
      product.
    reviewed_on: "2026-10-20"
  - target: http://purl.obolibrary.org/obo/NCIT_C2283
    relation: exactMatch
    evidence: https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C2283?include=full
    rationale: >-
      The source concerns the mature nine-amino-acid oxytocin peptide with mammary,
      uterine and social-cognition context. NCI defines the nine-residue OXT gene
      product with mammary and uterine smooth-muscle roles and possible
      bonding/cognition roles. Both denote the mature peptide; NCI's Recombinant
      Oxytocin C724 is a separate therapeutic concept.
    reviewed_on: "2026-10-20"
title: Oxytocin
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - neuromodulation
  - social_neuroscience
  - hormones
aliases:
  - oxytocin_system
semantic_relations:
  - type: relates
    links:
      - "[[social_cognition]]"
      - "[[social_neuroscience]]"
      - "[[precision_weighting]]"
      - "[[active_inference]]"
wikidata: Q169960
---

# Oxytocin

## Overview
Oxytocin is a nine-amino-acid peptide synthesised mainly in the paraventricular and supraoptic nuclei of the hypothalamus. It acts as a hormone, released from the posterior pituitary (uterine contraction in birth, milk ejection in lactation), and as a neuromodulator, released within the brain, where it acts on oxytocin receptors expressed in limbic and other regions. This page concerns its proposed role in social cognition; it is not medical advice and makes no treatment claims.

## Mechanism
Animal work links oxytocin signalling to maternal behaviour and to pair bonding in monogamous prairie voles. In humans, most evidence comes from intranasal administration studies, which test whether raising oxytocin changes behaviour or neural responses. Kosfeld and colleagues reported increased trust in an investment game, an influential result that stimulated a large literature. Subsequent reviews found inconsistent and sometimes null effects, small samples, and publication bias, and Nave and colleagues argued that evidence for a trust effect is weak. It is also debated how much intranasally administered peptide reaches the brain, and peripheral and central levels need not correlate.

The social salience hypothesis (Shamay-Tsoory and Abu-Akel) proposes that oxytocin increases the salience of social cues in context-dependent ways, accounting for reports of both prosocial effects and increased in-group bias or envy. Imaging studies report modulated amygdala responses to social stimuli. Genetic association studies of the receptor gene and social traits have likewise yielded inconsistent results. The popular label of a "love hormone" is not supported by this mixed evidence.

## Role in Active Inference
Neuromodulators are widely hypothesised in Active Inference to encode or set precision (confidence) of prediction errors and priors at different levels of a hierarchy. By analogy, oxytocin could adjust the precision attached to social and interoceptive signals, increasing the weight of social cues in belief updating ([[precision_weighting]], [[social_cognition]]). Quattrocki and Friston proposed such a role in a hypothesis linking oxytocin, interoception, and [[autism]]. This is speculative: no experiments have yet established oxytocin's effect on precision as such, and the same observations are explained by the salience account and by other accounts.

## Related concepts
- [[social_neuroscience]] — the field in which oxytocin research sits
- [[social_cognition]] — the processes it is hypothesised to modulate
- [[precision_mechanisms]] — neuromodulatory implementation of precision
- [[cooperation]] — behaviour studied in trust and economic-game paradigms
- [[autism]] — condition where oxytocin hypotheses have been proposed and debated
- [[emotional_processing]] — affective processes also modulated by the peptide

## References
- Kosfeld, M., Heinrichs, M., Zak, P. J., Fischbacher, U., & Fehr, E. (2005). Oxytocin increases trust in humans. *Nature*.
- Nave, G., Camerer, C., & McCullough, M. (2015). Does oxytocin increase trust in humans? A critical review of research. *Perspectives on Psychological Science*.
- Shamay-Tsoory, S. G., & Abu-Akel, A. (2016). The social salience hypothesis of oxytocin. *Biological Psychiatry*.
- Quattrocki, E., & Friston, K. (2014). Autism, oxytocin and interoception. *Neuroscience & Biobehavioral Reviews*.
