---
title: Metaplasticity
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - plasticity
  - learning
  - regulation
semantic_relations:
  - type: relates
    links:
      - "[[active_inference]]"
      - "[[synaptic_plasticity]]"
      - "[[stability_plasticity]]"
wikidata: Q1420291
---

# Metaplasticity

## Overview
Metaplasticity is the plasticity of synaptic plasticity: a change in the capacity of synapses to undergo LTP or LTD that depends on their prior activity or state. Abraham and Bear (1996) introduced the term. The same stimulus can produce different outcomes depending on the history of activity, so the rules governing change are themselves modifiable.

## Mechanism
The best-known formalization is the BCM theory, in which the sign of synaptic change depends on postsynaptic activity relative to a modification threshold $\theta_M$: $\Delta w \propto x\,y\,(y - \theta_M)$. The threshold slides with recent average postsynaptic activity, rising when the neuron has been very active and falling when it has been quiet. High past activity thus makes LTP harder and LTD easier, which stabilizes learning without a separate normalization step. Experimental findings consistent with this include:

- Prior synaptic or network activity shifting the frequency-response curve for LTP and LTD.
- Priming by NMDA receptor activation or other prior stimulation changing the induction threshold.
- Sensory deprivation (for example dark rearing) lowering the LTP threshold in visual cortex.

Candidate molecular substrates include changes in NMDA receptor subunit composition (the NR2A/NR2B ratio), calcium dynamics, kinase and phosphatase activity, and inhibitory tone. Metaplasticity is distinct from homeostatic scaling: scaling multiplies existing weights to hold firing rate steady, whereas metaplasticity changes the rule by which future Hebbian change occurs (see [[homeostatic_plasticity]]), although the two interact and are sometimes hard to separate experimentally.

## Role in Active Inference
There is a clear computational analogy, but it is an analogy. In Bayesian learning with conjugate Dirichlet parameters, the effective learning rate on a parameter is inversely related to the total accumulated count: a well-evidenced parameter moves little in response to a new observation, a poorly evidenced one moves a lot. Learning therefore regulates its own subsequent plasticity as a consequence of the update rule, much as metaplasticity does biologically. In hierarchical models, the precision assigned to prior beliefs about parameters likewise sets how readily they are revised (see [[precision_weighting]]). Whether the brain's sliding thresholds implement such precision dynamics is not established. A limitation worth stating is that a purely count-based rule keeps slowing down, which suits a stationary world but not a changing one (see [[online_learning]]).

## Related concepts
- [[synaptic_plasticity]] — the process whose rules metaplasticity modulates
- [[stability_plasticity]] — the trade-off metaplasticity helps manage
- [[hebbian_learning]] — BCM extends the Hebbian rule with a sliding threshold
- [[precision_weighting]] — computational analogue of regulating revisability
- [[critical_periods]] — developmental windows of altered plasticity
- [[meta_learning]] — learning to learn in the algorithmic sense

## References
- Abraham, W. C., & Bear, M. F. (1996). Metaplasticity: the plasticity of synaptic plasticity. *Trends in Neurosciences*.
- Bienenstock, E., Cooper, L., & Munro, P. (1982). Theory for the development of neuron selectivity. *Journal of Neuroscience*.
