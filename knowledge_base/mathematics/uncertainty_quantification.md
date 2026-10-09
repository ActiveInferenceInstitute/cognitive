---
title: Uncertainty Quantification
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - uncertainty
  - entropy
  - aleatoric_epistemic
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/mathematics/entropy]]"
      - "[[knowledge_base/mathematics/mutual_information]]"
      - "[[knowledge_base/mathematics/epistemic_value]]"
      - "[[knowledge_base/mathematics/efe_components]]"
---

# Uncertainty Quantification

## Overview
Uncertainty quantification (UQ) is the practice of representing, propagating and reporting what is not known: noise in the data-generating process, ignorance about parameters or states, and error in the model itself. A central distinction is between aleatoric uncertainty (irreducible variability) and epistemic uncertainty (reducible with more data).

## Definition / Formalism
Let a predictive distribution be a mixture over parameter or model hypotheses $\theta$ with weights $p(\theta)$. The entropy of the prediction splits exactly as

$$\underbrace{H\big[\mathbb E_\theta\,p(y\mid\theta)\big]}_{\text{total}}=\underbrace{\mathbb E_\theta\,H[p(y\mid\theta)]}_{\text{aleatoric}}+\underbrace{I(y;\theta)}_{\text{epistemic}} .$$

The epistemic part is a mutual information, so it is nonnegative and vanishes only when all hypotheses agree. A numerical check with weights $(0.2,0.5,0.3)$ over three binary predictive distributions $(0.9,0.1)$, $(0.5,0.5)$, $(0.2,0.8)$ gave total 0.6929, aleatoric 0.5617, epistemic 0.1312, and the mutual information computed directly matched the difference.

The same split in variance form is the law of total variance, $\mathrm{Var}(y)=\mathbb E[\mathrm{Var}(y\mid\theta)]+\mathrm{Var}(\mathbb E[y\mid\theta])$.

Common tools: credible intervals from a posterior; bootstrap and ensemble spread; forward propagation of input uncertainty by Monte Carlo or linearisation; sensitivity analysis to rank which inputs drive output variance; and calibration checks that a nominal 90% interval covers about 90% of held-out outcomes. Entropy is one summary of uncertainty; it ignores the scale of outcomes, so variance or interval width may be more appropriate for ordered quantities.

## Role in Active Inference
The split above appears directly in this repo's expected free energy. For a state belief $q(s)$ and likelihood $A$, the predictive entropy decomposes as $H[q(o)]=\text{ambiguity}+I(s;o)$, where ambiguity $=\mathbb E_{q(s)}H[P(o\mid s)]$ is the aleatoric part (observation noise even if the state were known) and $I(s;o)$ is the epistemic part (what an observation can reveal about the state). With $A=\begin{pmatrix}0.9&0.1\\0.1&0.9\end{pmatrix}$ and $q(s)=(0.7,0.3)$ the numbers are $0.6410=0.3251+0.3160$. Expected free energy is risk plus ambiguity; the information gain is already inside that sum, as the CHANGELOG correction records, so uncertainty-seeking is not added a second time. The `epistemic` value returned by `DiscreteGenerativeModel.expected_free_energy` is a diagnostic only.

## Related concepts
- [[entropy]] — the measure used to summarise predictive uncertainty
- [[mutual_information]] — the epistemic term
- [[epistemic_value]] — information gain as a driver of action selection
- [[uncertainty_resolution]] — action as uncertainty reduction
- [[sensitivity_analysis]] — propagating input uncertainty through a model
- [[cross_validation]] — empirical calibration against held-out data
- [[precision_parameter]] — inverse-variance weighting in inference
