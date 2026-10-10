---
title: Reconsolidation
description: Reconsolidation is the proposal that retrieving a consolidated memory returns it to a labile state requiring restabilization, allowing updating, and this page covers evidence, boundary conditions, and a predictive-processing reading.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - memory
  - consolidation
  - updating
semantic_relations:
  - type: relates
    links:
      - "[[memory_consolidation]]"
      - "[[synaptic_consolidation]]"
      - "[[memory_systems]]"
---

# Reconsolidation

## Overview
Consolidation stabilizes a new memory over hours to days. Reconsolidation is the hypothesis that under certain conditions a retrieved memory becomes labile again and must be restabilized, creating a window in which it can be strengthened, weakened, or modified. It challenges the view that a consolidated memory is fixed, and it connects memory retrieval with memory updating ([[memory_consolidation]]).

## Evidence
- **Animal work**: Nader, Schafe, and LeDoux (2000) reported that infusing a protein synthesis inhibitor into the rat amygdala shortly after retrieving an auditory fear memory impaired the memory at later testing, whereas the same infusion without retrieval did not. This resembled the effect of the inhibitor during original consolidation. Related findings from electroconvulsive shock after retrieval date to the late 1960s. Similar effects have been reported across tasks and species, though not universally.
- **Human work**: studies have reported that behavioural or pharmacological manipulations after reminders can alter later expression of fear and some declarative memories. Replication is mixed, and effect sizes and generality are actively disputed. Clinical applications are hypotheses under research and should not be treated as established interventions.
- **Boundary conditions**: reactivation does not always destabilize a memory. Conditions proposed to matter include memory age and strength, the duration of the reminder, and, importantly, whether the reminder contains a prediction error, meaning a mismatch between what was expected and what occurred. Memories that are fully predicted tend to be retrieved without updating.
- **Alternative accounts**: some effects have been explained as impaired retrieval, new learning of extinction, or state-dependent expression instead of a restabilization of the original trace.

## Role in Active Inference
The prediction-error boundary condition is the point of contact. In a generative-model account, a model is revised when its predictions fail, so a reminder that matches expectation leaves the model unchanged, while a surprising outcome during retrieval increases the free-energy gradient on the parameters involved and makes them open to modification. Reconsolidation can then be viewed as a case of precision-dependent belief updating during recall, with the labile window corresponding to a period of increased plasticity ([[synaptic_plasticity]], [[prediction_error]], [[belief_updating]]). This reading is interpretive and untested as a quantitative model; the molecular and systems-level mechanisms are described independently of the framework.

## Related concepts
- [[memory_consolidation]] — initial stabilization that reconsolidation revisits
- [[synaptic_consolidation]] — cellular protein-synthesis-dependent stabilization
- [[systems_consolidation]] — slower reorganization across brain regions
- [[memory_systems]] — memory types to which the effect may apply
- [[interference]] — competing traces that can alter retrieved memories
- [[prediction_error]] — mismatch signal proposed to trigger updating

## References
- Nader, K., Schafe, G. E., & LeDoux, J. E. (2000). Fear memories require protein synthesis in the amygdala for reconsolidation after retrieval. *Nature*, 406, 722-726.
