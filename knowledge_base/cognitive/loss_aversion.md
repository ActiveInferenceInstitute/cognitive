---
title: Loss Aversion
description: The tendency for losses to weigh more heavily than equal-sized gains in choice, as formalised in prospect theory, with its evidence, critiques and an analogy to preferences in Active Inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - decision_making
  - behavioural_economics
  - bias
semantic_relations:
  - type: relates
    links:
      - "[[kahneman_tversky]]"
      - "[[decision_making]]"
      - "[[utility_theory]]"
      - "[[probability_weighting]]"
      - "[[heuristics]]"
wikidata: Q2874240
---

# Loss Aversion

## Overview
Loss aversion is the finding that people typically evaluate a loss as more significant than a gain of the same size. Offered a gamble with an equal chance of winning or losing the same amount, most people decline, and they often require a potential gain well above the potential loss before accepting. The idea is a central element of prospect theory, proposed by Kahneman and Tversky (1979) as a descriptive alternative to expected utility theory.

## Formulation
In prospect theory outcomes are coded as gains or losses relative to a reference point, not as final states of wealth. The value function $v(x)$ is defined over these changes, is concave for gains, convex for losses, and steeper for losses than for gains. A common parametrisation is
$$v(x) = \begin{cases} x^{\alpha} & x \ge 0 \\ -\lambda(-x)^{\beta} & x < 0, \end{cases}$$
where $\lambda > 1$ is the loss-aversion coefficient. Tversky and Kahneman (1992) estimated $\lambda$ at about 2.25 (a median) in their sample, and later studies report a range of values, so the figure is not a universal constant.

## Evidence and critique
Related phenomena include the endowment effect and status-quo bias, which have been attributed to loss aversion. The evidence is not uniform. Some researchers argue that the effect depends on stake size, framing and experience, and that reported demonstrations in riskless exchange can be explained by other mechanisms. Others hold that the core finding is robust. The reference point itself is often unobserved, which complicates testing.

## Relation to Active Inference
Active Inference does not derive loss aversion from the free energy principle. Preferences enter through prior beliefs over outcomes, encoded in the $C$ vector of discrete models, and expected free energy scores policies by risk, the divergence between predicted and preferred outcomes, plus ambiguity. An asymmetry like loss aversion could be written into those preferences, or could arise from precision and uncertainty in outcome beliefs, but this would be a modelling choice and not an established result. The correspondence is therefore an analogy that has been explored informally, not a settled theory.

## Related concepts
- [[kahneman_tversky]] — the researchers who introduced prospect theory
- [[decision_making]] — the broader study of choice under risk and uncertainty
- [[utility_theory]] — the normative framework prospect theory modifies
- [[probability_weighting]] — the other main component of prospect theory
- [[heuristics]] — related judgment shortcuts and biases

## References
- Kahneman, D., & Tversky, A. (1979). Prospect theory: An analysis of decision under risk. Econometrica, 47, 263-291.
- Tversky, A., & Kahneman, D. (1991). Loss aversion in riskless choice: A reference-dependent model. Quarterly Journal of Economics, 106, 1039-1061.
- Tversky, A., & Kahneman, D. (1992). Advances in prospect theory: Cumulative representation of uncertainty. Journal of Risk and Uncertainty, 5, 297-323.
