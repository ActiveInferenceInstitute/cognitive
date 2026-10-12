---
title: Information Cascade
description: A sequential social-learning outcome in which individuals rationally follow the actions of predecessors and ignore their own private information, so that the group stops aggregating evidence and can lock onto a wrong choice.
type: concept
status: stable
created: 2026-10-12
updated: 2026-10-12
tags:
  - social_learning
  - herding
  - bayesian_inference
  - collective_behavior
semantic_relations:
  - type: relates
    links:
      - "[[social_learning]]"
      - "[[collective_behavior]]"
      - "[[bayesian_inference]]"
      - "[[social_norms]]"
      - "[[cultural_transmission]]"
      - "[[swarm_intelligence]]"
---

# Information Cascade

## Overview
An information cascade occurs when people decide one after another, each observing the choices of those before them but not their private evidence, and each rationally concludes that the observed pattern outweighs their own signal. From that point their actions carry no new information, so the crowd's behaviour reflects the first few decisions and not the evidence available to all members. Cascades give a formal account of how conformity, fads and fashions can arise without any preference for conforming, and of how a large group can be confidently wrong.

## The basic model
The standard formulation was given independently by Bikhchandani, Hirshleifer and Welch (1992) and Banerjee (1992). Individuals choose in a fixed order between adopting and rejecting an option whose value is unknown. Each receives a private signal that is correct with probability $p>1/2$, and observes the earlier choices but not the earlier signals. A decision-maker applies Bayes' rule to the public history and the private signal, so that

$$\frac{P(\text{good}\mid \text{history},s)}{P(\text{bad}\mid \text{history},s)}=\frac{P(\text{good})}{P(\text{bad})}\cdot\frac{P(\text{history}\mid\text{good})}{P(\text{history}\mid\text{bad})}\cdot\frac{P(s\mid\text{good})}{P(s\mid\text{bad})}.$$

In the simple binary version, once the earlier choices lean in one direction by two adoptions or two rejections, the public likelihood ratio outweighs any single private signal, and each later individual follows the pattern whatever their own signal says. Because a follower's action does not depend on their signal, later followers learn nothing new from it. The tendency to start in the wrong direction is real: a few early signals that happen to mislead can set a cascade off.

## Properties
- **Inefficiency.** Information is not aggregated, so the outcome can be worse than if signals were pooled. Cascades can be incorrect even though every individual acts rationally.
- **Fragility.** Because only a little information supports it, a cascade can be broken by a small new public signal, such as an expert announcement, and fashions can shift quickly.
- **Distinction from herding by preference.** Following others because of payoff externalities or social pressure is a different mechanism, and laboratory work has tried to separate cascades from other herd behaviour (Çelen and Kariv, 2004).

## Relation to Active Inference
The model is Bayesian inference over a hidden state with the choices of others treated as observations, which is the generic setting of Active Inference (see [[bayesian_inference]]). Reading a cascade as precision weighting, in which the public history is given more precision than the private signal, is an interpretive translation and not a result of the framework. The standard cascade model does not use free energy minimisation, and fuller Active Inference treatments of social learning add action selection with epistemic value, which may favour acting on private information to resolve uncertainty (see [[social_learning]] and [[epistemic_foraging]]).

## References
- Bikhchandani, S., Hirshleifer, D., & Welch, I. (1992). A Theory of Fads, Fashion, Custom, and Cultural Change as Informational Cascades. Journal of Political Economy, 100(5), 992-1026. doi:10.1086/261849
- Banerjee, A. V. (1992). A Simple Model of Herd Behavior. The Quarterly Journal of Economics, 107(3), 797-817. doi:10.2307/2118364
- Bikhchandani, S., Hirshleifer, D., & Welch, I. (1998). Learning from the Behavior of Others: Conformity, Fads, and Informational Cascades. Journal of Economic Perspectives, 12(3), 151-170. doi:10.1257/jep.12.3.151
- Çelen, B., & Kariv, S. (2004). Distinguishing Informational Cascades from Herd Behavior in the Laboratory. American Economic Review, 94(3), 484-498. doi:10.1257/0002828041464461
