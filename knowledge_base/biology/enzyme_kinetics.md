---
title: Enzyme Kinetics
description: The quantitative study of the rates of enzyme-catalysed reactions, including the Michaelis-Menten model, kinetic parameters, inhibition types and cooperativity, with its role in metabolic modelling.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - biochemistry
  - reaction_rates
  - catalysis
semantic_relations:
  - type: relates
    links:
      - "[[biochemistry]]"
      - "[[metabolic_networks]]"
      - "[[thermodynamics]]"
      - "[[protein_folding]]"
      - "[[cell_biology]]"
---

# Enzyme Kinetics

## Overview
Enzyme kinetics measures how fast enzyme-catalysed reactions proceed and how the rate depends on substrate concentration, enzyme concentration, inhibitors, temperature and pH. Enzymes lower the activation energy of a reaction without changing its equilibrium, so kinetic analysis characterises catalytic efficiency and reveals mechanism. It provides the rate laws used in models of [[metabolic_networks]] and in pharmacology.

## The Michaelis-Menten model
For a single substrate, the simplest mechanism is $E + S \rightleftharpoons ES \rightarrow E + P$. Under the steady-state assumption of Briggs and Haldane, in which the concentration of the enzyme-substrate complex stays roughly constant, the initial rate is
$$v = \frac{V_{\max}[S]}{K_m + [S]}.$$
Here $V_{\max}=k_{cat}[E]_T$ is the maximal rate when the enzyme is saturated, $k_{cat}$ is the turnover number (the number of substrate molecules converted per active site per unit time), and $K_m$ is the substrate concentration at which $v = V_{\max}/2$. The ratio $k_{cat}/K_m$ is a measure of catalytic efficiency at low substrate concentration. Parameters are often estimated by nonlinear regression, and linearised plots such as Lineweaver-Burk are mainly used for visual diagnosis because they distort error.

## Inhibition and regulation
- **Competitive inhibitors** bind the free enzyme and raise the apparent $K_m$ without changing $V_{\max}$.
- **Uncompetitive inhibitors** bind only the enzyme-substrate complex and lower both parameters.
- **Noncompetitive (mixed) inhibitors** bind at a separate site and reduce $V_{\max}$.
- **Cooperativity and allostery** occur in multi-subunit enzymes, producing sigmoidal rate curves that the hyperbolic Michaelis-Menten form cannot describe. Models such as the Hill equation capture the steepness of the response.

## Relation to Active Inference
The connection to Active Inference is weak and indirect. Enzyme kinetics is a standard chemical description of the rate processes that sustain a cell's metabolic steady state, and the free energy principle treats such maintained nonequilibrium states as a physical precondition for life (see [[thermodynamics]]). Rate laws can be placed inside stochastic or dynamical models of cells, but kinetics itself does not involve inference, and no result in the field depends on the framework.

## Related concepts
- [[biochemistry]] — the broader discipline that supplies enzyme structure and function
- [[metabolic_networks]] — systems of reactions whose fluxes depend on kinetic parameters
- [[thermodynamics]] — sets the energetic limits on reaction direction and rate
- [[protein_folding]] — determines the structure that gives an enzyme its activity
- [[cell_biology]] — the cellular context in which enzymes operate

## References
- Michaelis, L., & Menten, M. L. (1913). Die Kinetik der Invertinwirkung. Biochemische Zeitschrift, 49, 333-369.
- Briggs, G. E., & Haldane, J. B. S. (1925). A note on the kinetics of enzyme action. Biochemical Journal, 19, 338-339.
- Fersht, A. (1999). Structure and Mechanism in Protein Science. New York: W. H. Freeman.
