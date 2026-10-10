---
title: Kahneman and Tversky
description: Overview of Daniel Kahneman and Amos Tversky's heuristics-and-biases programme and prospect theory, and how their findings relate to Active Inference accounts of decision making.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - decision-making
  - heuristics
  - prospect-theory
semantic_relations:
  - type: relates
    links:
      - "[[decision_making]]"
      - "[[decision_theory]]"
      - "[[probability_weighting]]"
---

# Kahneman and Tversky

## Overview
Daniel Kahneman and Amos Tversky collaborated from the late 1960s until Tversky's death in 1996 and founded the empirical study of systematic departures from expected-utility theory and probability theory in human judgment. Kahneman received the 2002 Nobel Memorial Prize in Economic Sciences for this work. Their results define the "heuristics and biases" programme and behavioural economics. This page refers to the research programme, not to the two individuals' biographies.

## Core claims
**Heuristics and biases.** Judgments under uncertainty often rely on a few rules of thumb:
- *Representativeness*: probability is judged by similarity to a prototype, which can produce base-rate neglect and the conjunction fallacy (the "Linda" problem).
- *Availability*: frequency or probability is judged by how easily instances come to mind.
- *Anchoring and adjustment*: estimates are pulled toward an initial value, and adjustment is typically insufficient.

**Prospect theory.** Choices under risk are described by a reference-dependent value function and a probability weighting function. Outcomes are coded as gains or losses relative to a reference point; the value function is concave for gains, convex for losses, and steeper for losses (loss aversion). In the 1992 cumulative version, median fitted parameters were roughly 0.88 for curvature and 2.25 for loss aversion, so a loss of 100 is felt about 2.25 times as strongly as a gain of 100. Small probabilities are overweighted and moderate to high ones underweighted (see [[probability_weighting]]).

**Framing.** Logically equivalent descriptions of options, such as lives saved versus lives lost, change preferences, violating description invariance.

Later work by Kahneman popularized a two-system description (fast intuitive and slow deliberate processing). It is a heuristic framing and is contested as a literal account of separate mechanisms. Critics such as Gigerenzer argue that the heuristics are often adaptive in natural environments, not simply errors.

## Role in Active Inference
The link is by comparison, not derivation. Active Inference models choice by minimizing expected free energy, with preferences encoded as prior beliefs over outcomes, so reference dependence and loss aversion would correspond to the shape and location of those priors; no single parameterization is canonical. Heuristic judgment is sometimes read as approximate inference under limited resources, and framing effects as consequences of how context sets priors. These readings are hypotheses under active research, and prospect theory remains a descriptive account, not a normative one.

## Related concepts
- [[decision_making]] — the domain the programme transformed
- [[decision_theory]] — normative baseline that prospect theory departs from
- [[utility_theory]] — expected-utility framework challenged by the findings
- [[probability_weighting]] — the weighting function at the core of prospect theory
- [[risk_assessment]] — applied setting for risk attitudes
- [[expected_free_energy]] — Active Inference objective with risk and ambiguity terms

## References
- Tversky, A., & Kahneman, D. (1974). Judgment under uncertainty: Heuristics and biases. *Science*, 185, 1124-1131.
- Kahneman, D., & Tversky, A. (1979). Prospect theory: An analysis of decision under risk. *Econometrica*, 47, 263-291.
- Tversky, A., & Kahneman, D. (1992). Advances in prospect theory: Cumulative representation of uncertainty. *Journal of Risk and Uncertainty*, 5, 297-323.
