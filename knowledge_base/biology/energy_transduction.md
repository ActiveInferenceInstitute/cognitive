---
title: Energy Transduction
description: The conversion of energy between forms in living cells, such as light to chemical gradients, gradients to ATP, and ATP to mechanical work, with the chemiosmotic mechanism as its central example.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - bioenergetics
  - thermodynamics
  - cell_biology
semantic_relations:
  - type: relates
    links:
      - "[[thermodynamics]]"
      - "[[metabolic_efficiency]]"
      - "[[molecular_motors]]"
      - "[[ion_channels]]"
      - "[[dissipative_structures]]"
      - "[[information_thermodynamics]]"
---

# Energy Transduction

## Overview
Energy transduction is the conversion of energy from one form to another by biological machinery. Cells take in energy as light or as the chemical energy of food and transform it into a small number of usable forms: ion gradients across membranes, high-energy phosphate bonds in ATP, and reduced coenzymes. These in turn drive synthesis, transport, signalling and movement. The processes obey the laws of [[thermodynamics]]: the conversion is never perfectly efficient, and a living cell remains far from equilibrium by continually dissipating energy.

## Chemiosmotic coupling
In the chemiosmotic theory of Peter Mitchell (1961), electron transfer through respiratory or photosynthetic chains pumps protons across a membrane, producing a proton-motive force. For protons moving back across the membrane from the pumped (outer) side to the inner side, the free energy released per unit charge, expressed as a potential, is
$$\Delta p=(\psi_{\text{out}}-\psi_{\text{in}})+\frac{2.303\,RT}{F}\,(\mathrm{pH}_{\text{in}}-\mathrm{pH}_{\text{out}}),$$
where $\psi$ is the electrical potential on each side, $R$ the gas constant, $T$ the temperature and $F$ the Faraday constant; the prefactor is about 59 mV at 25 degrees Celsius. Both terms are positive when the pumped side is more positive and more acidic. Protons returning through ATP synthase drive the rotation of its central stalk, and the conformational changes in its catalytic sites produce ATP from ADP and phosphate, as described by Boyer's binding-change mechanism.

## Other examples
- **Chemical to mechanical.** [[molecular_motors]] such as myosin, kinesin and the bacterial flagellar motor use ATP hydrolysis or an ion gradient to produce directed movement.
- **Chemical to electrical.** Pumps such as the sodium-potassium ATPase use ATP to maintain ion gradients that [[ion_channels]] then use to generate signals.
- **Light to chemical.** Photosynthetic reaction centres convert absorbed photons into charge separation.

ATP is the shared intermediate that couples favourable reactions to unfavourable ones.

## Relation to Active Inference
The free energy of the free energy principle is the variational free energy of an information-theoretic model, which differs from the thermodynamic free energy relevant to bioenergetics. They are related through the thermodynamics of information, such as Landauer's bound on the cost of erasing a bit, and through work linking inference to non-equilibrium steady states (see [[information_thermodynamics]]), but they are not identical quantities. Treating metabolic cost as a constraint on inference is a proposal, not an established result.

## Related concepts
- [[thermodynamics]] — the laws that limit conversion efficiency
- [[metabolic_efficiency]] — how effectively a cell converts energy to work
- [[molecular_motors]] — conversion of chemical energy into movement
- [[dissipative_structures]] — ordered systems maintained by energy flow
- [[information_thermodynamics]] — the link between information processing and energy cost

## References
- Mitchell, P. (1961). Coupling of phosphorylation to electron and hydrogen transfer by a chemi-osmotic type of mechanism. Nature, 191, 144-148.
- Boyer, P. D. (1997). The ATP synthase: A splendid molecular machine. Annual Review of Biochemistry, 66, 717-749.
- Nicholls, D. G., & Ferguson, S. J. (2013). Bioenergetics (4th ed.). Academic Press.
