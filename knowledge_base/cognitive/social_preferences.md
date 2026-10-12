---
title: Social Preferences
description: Preferences that depend on the outcomes or intentions of other people, including inequity aversion, reciprocity and altruism, with formal models and their relation to preference priors in Active Inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - behavioural_economics
  - fairness
  - preferences
semantic_relations:
  - type: relates
    links:
      - "[[game_theory]]"
      - "[[cooperation]]"
      - "[[social_reward]]"
      - "[[trust_games]]"
      - "[[social_punishment]]"
wikidata: Q7551312
---

# Social Preferences

## Overview
Social preferences are preferences over outcomes that depend on what happens to others as well as on one's own payoff. Standard economic models assume self-interested agents who care only about their own material outcome. Experiments with games such as the ultimatum, dictator and trust games show that many participants give up resources to reward fairness or punish unfairness, which motivated models that add other-regarding terms to utility.

## Main forms
- **Altruism**: positive weight on another's welfare.
- **Inequity aversion**: disutility from differences between one's own payoff and others'. In the Fehr and Schmidt (1999) model, the utility of player $i$ among $n$ players is
$$U_i(x)=x_i-\frac{\alpha_i}{n-1}\sum_{j\ne i}\max(x_j-x_i,0)-\frac{\beta_i}{n-1}\sum_{j\ne i}\max(x_i-x_j,0),$$
  with $\beta_i\le\alpha_i$ and $0\le\beta_i<1$, so disadvantageous inequality is penalised at least as strongly as advantageous inequality.
- **Reciprocity**: kind or unkind responses depend on the perceived intentions of others, as in the models of Rabin (1993) and later work, so that identical outcomes can be treated differently depending on how they arose.
- **Concern for efficiency or total welfare**, as in the mixed model of Charness and Rabin (2002).

Preferences vary substantially across individuals and cultures, and behaviour can also reflect beliefs about norms, reputation and strategic motives, so identifying a pure preference from choices is difficult. Neuroeconomic studies associate processing of fairness and social reward with valuation circuitry, a topic discussed under [[social_reward]].

## Relation to Active Inference
In Active Inference, preferences are encoded as prior beliefs about preferred outcomes, often denoted $C$ in discrete models, which enter expected free energy. Social preferences could be represented by priors over outcomes of other agents or over the joint state of an interaction, so that agents expect, and therefore act to bring about, fair or cooperative states. This is a modelling possibility rather than an established result: existing demonstrations use hand-specified priors, and the account does not by itself explain where such priors come from.

## Related concepts
- [[game_theory]] — formal framework in which social preferences are tested
- [[cooperation]] — behaviour influenced by other-regarding concerns
- [[social_reward]] — valuation of social outcomes in the brain
- [[trust_games]] — experiments that reveal reciprocity
- [[social_punishment]] — costly sanctions of unfair behaviour
- [[prisoners_dilemma]] — a dilemma in which other-regarding preferences change equilibria

## References
- Fehr, E., & Schmidt, K. M. (1999). A theory of fairness, competition, and cooperation. Quarterly Journal of Economics, 114, 817-868.
- Bolton, G. E., & Ockenfels, A. (2000). ERC: A theory of equity, reciprocity, and competition. American Economic Review, 90, 166-193.
- Charness, G., & Rabin, M. (2002). Understanding social preferences with simple tests. Quarterly Journal of Economics, 117, 817-869.
- Rabin, M. (1993). Incorporating fairness into game theory and economics. American Economic Review, 83, 1281-1302.
