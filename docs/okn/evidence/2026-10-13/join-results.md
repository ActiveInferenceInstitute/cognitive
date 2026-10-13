---
title: Captured OKN cross-graph join results
type: reference
status: stable
updated: 2026-10-13
---

# Captured OKN cross-graph join results

Observed at 2026-10-13T22:32:42.376166+00:00. Local cognitive-kb was joined with live OKN
identifier-mappings and Ubergraph responses, then replayed offline.

| Measurement | Value |
|---|---|
| Joined rows | 43 |
| Matched concept pages | 39 |
| Labeled external OBO terms | 43 |

The first edge is a candidate `skos:closeMatch` from an explanatory page
to Wikidata. The second is the mapping graph's `skos:exactMatch` to an
OBO identifier. Joining these edges does not establish identity between
the page and the ontology class. Labels are discovery signposts, not
clinical or scientific validation. Unmatched pages are omitted.

| Source page | Wikidata join key | External term |
|---|---|---|
| [Axon Guidance](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/biology/axon_guidance.md) | [Q2034716](http://www.wikidata.org/entity/Q2034716) | [axon guidance](http://purl.obolibrary.org/obo/GO_0007411) |
| [Biofilm Formation](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/biology/biofilm_formation.md) | [Q21759980](http://www.wikidata.org/entity/Q21759980) | [biofilm formation](http://purl.obolibrary.org/obo/GO_0042710) |
| [Cell Migration](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/biology/cell_migration.md) | [Q189092](http://www.wikidata.org/entity/Q189092) | [cell migration](http://purl.obolibrary.org/obo/GO_0016477) |
| [Cortisol](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/biology/cortisol.md) | [Q190875](http://www.wikidata.org/entity/Q190875) | [cortisol](http://purl.obolibrary.org/obo/CHEBI_17650) |
| [Cytoskeleton](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/biology/cytoskeleton.md) | [Q154626](http://www.wikidata.org/entity/Q154626) | [cytoskeleton](http://purl.obolibrary.org/obo/GO_0005856) |
| [Gene Expression](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/biology/gene_expression.md) | [Q26972](http://www.wikidata.org/entity/Q26972) | [gene expression](http://purl.obolibrary.org/obo/GO_0010467) |
| [Locomotion](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/biology/locomotion.md) | [Q2633862](http://www.wikidata.org/entity/Q2633862) | [locomotion](http://purl.obolibrary.org/obo/GO_0040011) |
| [Long-Term Potentiation](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/biology/ltp.md) | [Q1805481](http://www.wikidata.org/entity/Q1805481) | [long-term synaptic potentiation](http://purl.obolibrary.org/obo/GO_0060291) |
| [Myelination](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/biology/myelination.md) | [Q14876135](http://www.wikidata.org/entity/Q14876135) | [myelination](http://purl.obolibrary.org/obo/GO_0042552) |
| [Neuromuscular Junction](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/biology/neuromuscular_junction.md) | [Q776995](http://www.wikidata.org/entity/Q776995) | [neuromuscular junction](http://purl.obolibrary.org/obo/GO_0031594) |
| [Pollination Biology](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/biology/pollination_biology.md) | [Q134624](http://www.wikidata.org/entity/Q134624) | [pollination](http://purl.obolibrary.org/obo/GO_0009856) |
| [Serotonin](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/biology/serotonin.md) | [Q167934](http://www.wikidata.org/entity/Q167934) | [serotonin](http://purl.obolibrary.org/obo/CHEBI_28790) |
| [Symbiosis](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/biology/symbiosis.md) | [Q121610](http://www.wikidata.org/entity/Q121610) | [biological process involved in symbiotic interaction](http://purl.obolibrary.org/obo/GO_0044403) |
| [Testosterone](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/biology/testosterone.md) | [Q1318776](http://www.wikidata.org/entity/Q1318776) | [testosterone](http://purl.obolibrary.org/obo/CHEBI_17347) |
| [Wound Healing](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/biology/wound_healing.md) | [Q1509074](http://www.wikidata.org/entity/Q1509074) | [wound healing](http://purl.obolibrary.org/obo/GO_0042060) |
| [Agnosia](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/cognitive/agnosia.md) | [Q220322](http://www.wikidata.org/entity/Q220322) | [Disturbed sensory perception](http://purl.obolibrary.org/obo/HP_0010524) |
| [Agnosia](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/cognitive/agnosia.md) | [Q220322](http://www.wikidata.org/entity/Q220322) | [agnosia](http://purl.obolibrary.org/obo/MONDO_0005638) |
| [Anterograde Amnesia](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/cognitive/anterograde_amnesia.md) | [Q572111](http://www.wikidata.org/entity/Q572111) | [anterograde amnesia](http://purl.obolibrary.org/obo/MONDO_0003398) |
| [Anxiety](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/cognitive/anxiety.md) | [Q154430](http://www.wikidata.org/entity/Q154430) | [Anxiety](http://purl.obolibrary.org/obo/HP_0000739) |
| [Aphasia](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/cognitive/aphasia.md) | [Q2836](http://www.wikidata.org/entity/Q2836) | [obsolete Dysphasia](http://purl.obolibrary.org/obo/HP_0002357) |
| [Aphasia](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/cognitive/aphasia.md) | [Q2836](http://www.wikidata.org/entity/Q2836) | [aphasia](http://purl.obolibrary.org/obo/MONDO_0000598) |
| [Associative Learning](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/cognitive/associative_learning.md) | [Q14330970](http://www.wikidata.org/entity/Q14330970) | [associative learning](http://purl.obolibrary.org/obo/GO_0008306) |
| [Autism](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/cognitive/autism.md) | [Q38404](http://www.wikidata.org/entity/Q38404) | [Autism](http://purl.obolibrary.org/obo/HP_0000717) |
| [Coma](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/cognitive/coma.md) | [Q159557](http://www.wikidata.org/entity/Q159557) | [Coma](http://purl.obolibrary.org/obo/HP_0001259) |
| [Dark Adaptation](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/cognitive/dark_adaptation.md) | [Q22273766](http://www.wikidata.org/entity/Q22273766) | [dark adaptation](http://purl.obolibrary.org/obo/GO_1990603) |
| [Habituation](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/cognitive/habituation.md) | [Q1136816](http://www.wikidata.org/entity/Q1136816) | [habituation](http://purl.obolibrary.org/obo/GO_0046959) |
| [Impulsivity](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/cognitive/impulsivity.md) | [Q1549918](http://www.wikidata.org/entity/Q1549918) | [Impulsivity](http://purl.obolibrary.org/obo/HP_0100710) |
| [Interoception](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/cognitive/interoception.md) | [Q1668091](http://www.wikidata.org/entity/Q1668091) | [interoceptive system](http://purl.obolibrary.org/obo/UBERON_0036255) |
| [Movement Disorders](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/cognitive/movement_disorders.md) | [Q2608695](http://www.wikidata.org/entity/Q2608695) | [obsolete rare movement disorder](http://purl.obolibrary.org/obo/MONDO_0015143) |
| [Neurogenesis](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/cognitive/neurogenesis.md) | [Q1456827](http://www.wikidata.org/entity/Q1456827) | [neurogenesis](http://purl.obolibrary.org/obo/GO_0022008) |
| [Observational Learning](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/cognitive/observational_learning.md) | [Q1143725](http://www.wikidata.org/entity/Q1143725) | [observational learning](http://purl.obolibrary.org/obo/GO_0098597) |
| [Operant Conditioning](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/cognitive/operant_conditioning.md) | [Q847079](http://www.wikidata.org/entity/Q847079) | [operant conditioning](http://purl.obolibrary.org/obo/GO_0035106) |
| [Oxytocin](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/cognitive/oxytocin.md) | [Q169960](http://www.wikidata.org/entity/Q169960) | [oxytocin](http://purl.obolibrary.org/obo/CHEBI_7872) |
| [Proprioception](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/cognitive/proprioception.md) | [Q1129066](http://www.wikidata.org/entity/Q1129066) | [proprioception](http://purl.obolibrary.org/obo/GO_0019230) |
| [Simultanagnosia](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/cognitive/simultanagnosia.md) | [Q2046811](http://www.wikidata.org/entity/Q2046811) | [simultanagnosia](http://purl.obolibrary.org/obo/MONDO_0000678) |
| [Sleep](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/cognitive/sleep.md) | [Q35831](http://www.wikidata.org/entity/Q35831) | [sleep](http://purl.obolibrary.org/obo/GO_0030431) |
| [Sleep](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/cognitive/sleep.md) | [Q35831](http://www.wikidata.org/entity/Q35831) | [Drowsiness](http://purl.obolibrary.org/obo/HP_0002329) |
| [Specific Language Impairment](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/cognitive/specific_language_impairment.md) | [Q775593](http://www.wikidata.org/entity/Q775593) | [specific language impairment](http://purl.obolibrary.org/obo/MONDO_0000724) |
| [Tremor](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/cognitive/tremor.md) | [Q223907](http://www.wikidata.org/entity/Q223907) | [Tremor](http://purl.obolibrary.org/obo/HP_0001337) |
| [Tremor](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/cognitive/tremor.md) | [Q223907](http://www.wikidata.org/entity/Q223907) | [obsolete rare tremor disorder](http://purl.obolibrary.org/obo/MONDO_0017644) |
| [Visual Agnosia](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/cognitive/visual_agnosia.md) | [Q18742](http://www.wikidata.org/entity/Q18742) | [visual agnosia](http://purl.obolibrary.org/obo/MONDO_0000685) |
| [Visual Perception](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/cognitive/visual_perception.md) | [Q162668](http://www.wikidata.org/entity/Q162668) | [visual perception](http://purl.obolibrary.org/obo/GO_0007601) |
| [Environmental Monitoring](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/systems/environmental_monitoring.md) | [Q1749732](http://www.wikidata.org/entity/Q1749732) | [environmental monitoring](http://purl.obolibrary.org/obo/ENVO_02500041) |
