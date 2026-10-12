---
title: Psychometrics
description: The theory and technique of measuring psychological attributes, covering classical test theory, reliability, validity, scale types and item response models, and the use of such methods in computational modelling.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - measurement
  - testing
  - statistics
semantic_relations:
  - type: relates
    links:
      - "[[measurement_theory]]"
      - "[[operationalism]]"
      - "[[psychophysics]]"
      - "[[mathematical_psychology]]"
      - "[[fluid_intelligence]]"
      - "[[educational_psychology]]"
---

# Psychometrics

## Overview
Psychometrics is the field concerned with the construction, evaluation and interpretation of measurements of psychological attributes such as abilities, attitudes, personality traits and symptoms. It supplies the standards by which a standardised test or questionnaire is judged: how consistently it measures, whether it measures what it is intended to, and how scores should be interpreted. It is distinct from the psychophysical measurement of sensory thresholds, although the two share history and statistical methods ([[psychophysics]]).

## Core concepts
- **Scales of measurement.** Stevens (1946) classified measurement scales as nominal, ordinal, interval and ratio, according to which transformations leave the information intact. Whether psychological test scores reach interval level is debated ([[measurement_theory]]).
- **Classical test theory.** An observed score $X$ is treated as a true score $T$ plus error $E$, so $X=T+E$. Reliability is the proportion of observed-score variance that is true-score variance, $\rho=\sigma_T^2/\sigma_X^2$. Common estimates include test-retest correlation and internal consistency, for which Cronbach's alpha is widely used (Cronbach, 1951).
- **Validity.** A score is valid to the extent that evidence supports the interpretation proposed for it. Cronbach and Meehl (1955) introduced construct validity, the idea that a measure of a theoretical attribute is judged by its place in a network of predicted relations. This links psychometrics to [[operationalism]], which holds that a concept is defined by the operations used to measure it, though modern validity theory rejects a purely operational definition.
- **Item response theory.** Models such as the Rasch model relate the probability of a correct response to the difference between a person's latent ability and an item's difficulty. They allow the placement of persons and items on a common scale.
- **Factor analysis.** It identifies latent dimensions that account for correlations among items, as in models of intelligence ([[fluid_intelligence]]).

## Relation to Active Inference
The link is practical rather than theoretical. Computational modelling of behaviour, including Active Inference models fitted to task data, produces parameters intended to measure individual differences, for example in precision or learning rate. Whether such parameters have adequate reliability and validity is a psychometric question, and studies of reinforcement learning and related tasks have reported modest test-retest reliability for some parameters. Evaluating them with the tools above is a recommended practice.

## Related concepts
- [[measurement_theory]] — the formal basis of assigning numbers to attributes
- [[operationalism]] — defining concepts through measurement procedures
- [[mathematical_psychology]] — formal modelling of behaviour, closely tied to measurement
- [[fluid_intelligence]] — an ability construct studied through test batteries
- [[educational_psychology]] — a field that applies tests of aptitude and achievement

## References
- Stevens, S. S. (1946). On the theory of scales of measurement. Science, 103, 677-680.
- Cronbach, L. J. (1951). Coefficient alpha and the internal structure of tests. Psychometrika, 16, 297-334.
- Cronbach, L. J., & Meehl, P. E. (1955). Construct validity in psychological tests. Psychological Bulletin, 52, 281-302.
- Nunnally, J. C., & Bernstein, I. H. (1994). Psychometric Theory (3rd ed.). McGraw-Hill.
