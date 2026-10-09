---
title: Verification Theory
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - philosophy_of_science
  - epistemology
  - confirmation
semantic_relations:
  - type: relates
    links:
      - "[[logical_positivism]]"
      - "[[active_inference]]"
      - "[[model_comparison]]"
---

# Verification Theory

## Overview
Verification theory is the family of doctrines holding that the meaning, or at least the scientific standing, of a statement depends on how it can be checked against experience. It begins with the verification principle of [[logical_positivism]] and continues, after that principle failed, as the theory of confirmation and its rival, falsification.

## Core claims
**Strong verification.** A statement is meaningful only if conclusively verifiable. This excludes universal laws ("all copper conducts"), since no finite set of observations entails them.

**Weak verification.** Ayer's 1936 formulation requires only that some observation statement be deducible from the statement together with auxiliary hypotheses, without being deducible from the auxiliaries alone. Critics showed this admits almost any sentence, because one can always choose auxiliaries that make an arbitrary statement observationally relevant.

**Falsification.** Popper replaced verifiability as a criterion of meaning with falsifiability as a criterion of demarcation: scientific claims forbid some observable outcomes. The Duhem-Quine point limits this, since an observation refutes only a theory plus its auxiliaries.

**Confirmation.** Carnap and Hempel pursued degrees of confirmation, and the Bayesian version is now standard: evidence E confirms H when P(H|E) > P(H). Hempel's raven paradox and Goodman's grue show that qualitative accounts need care with what counts as an instance.

## Role in Active Inference
Bayesian confirmation is built into the formalism. The variational free energy bounds log model evidence, and its standard decomposition into accuracy minus complexity is a graded, Occam-penalized form of verification: a model is supported to the extent it fits data without being more flexible than necessary. In this repository `ActiveInferenceModel.calculate_free_energy` returns an accuracy term plus a complexity term, the KL divergence of current beliefs from the prior; the accuracy term is a squared-error proxy rather than a categorical log likelihood, so it is a convenience, not full evidence.

At the level of the framework the situation is weaker. Model comparison verifies particular generative models against particular data; it does not obviously verify or falsify the principle itself, which is why the testability dispute in [[dark_room_problem]] and [[logical_positivism]] persists. Marginal likelihood also depends on the prior, so Bayesian verification inherits any arbitrariness in the chosen prior.

## Related concepts
- [[logical_positivism]] — origin of the verification principle
- [[operationalism]] — meaning through measurement operations
- [[model_comparison]] — evidence-based model selection
- [[hypothesis_testing]] — frequentist counterpart of confirmation
- [[bayesian_inference]] — the update rule behind Bayesian confirmation
- [[variational_free_energy]] — the bound on model evidence
- [[scientific_method]] — the wider practice verification serves

## References
- Ayer, A. J. (1936). *Language, Truth and Logic*.
- Popper, K. (1934). *Logik der Forschung*.
- Hempel, C. G. (1945). Studies in the Logic of Confirmation. *Mind*.
- Goodman, N. (1955). *Fact, Fiction, and Forecast*.
