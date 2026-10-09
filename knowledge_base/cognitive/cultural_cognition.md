---
title: Cultural Cognition
type: knowledge_base
status: stable
created: 2026-02-07
updated: 2026-10-09
tags:
  - cognitive
  - active_inference
semantic_relations:
  - type: relates_to
    links:
      - "[[knowledge_base/cognitive/active_inference]]"
      - "[[knowledge_base/research/cultural_dynamics]]"
      - "[[knowledge_base/cognitive/social_cognition]]"
      - "[[knowledge_base/cognitive/social_learning]]"
      - "[[knowledge_base/biology/niche_construction]]"
---

# Cultural Cognition

## Overview

Cultural cognition asks how shared practices, norms and artefacts shape what individuals perceive, expect and do, and how individual cognition in turn reproduces culture. The phrase also names a separate research line in the study of risk perception, in which people's group values shape their beliefs about contested factual questions. This page uses it in the broader sense, as the study of cognition in cultural context. Reading it through Active Inference is an active research programme that is mostly conceptual, with few quantitative models fitted to cross-cultural data.

## Main Ideas

**Cultural priors.** Cultural learning supplies an individual with priors that no single lifetime could acquire: what counts as a plausible object, a polite gesture or a likely next move. In Active Inference terms, these are parameters of the generative model that are transmitted socially and then refined by experience.

**Shared generative models.** The variational account of culture proposed by Veissière, Constant, Ramstead, Friston and Kirmayer (2020) treats members of a community as agents who minimise free energy partly by aligning their expectations with each other. Expecting others to act in conventional ways, and acting conventionally oneself, makes everyone more predictable. Norms then function as policy priors: conventional actions are assigned high prior probability and deviations carry a surprise cost.

**Niche construction.** Cultures alter the environment, through tools, institutions and written language, so that future members face a world already structured to be predictable. Constant et al. (2018) formalise niche construction as agents shaping the sensory input they will later infer from ([[knowledge_base/biology/niche_construction]]).

**Cross-cultural differences.** If priors and precision settings differ between communities, perception and attention should differ too. Claims such as holistic versus analytic attention are discussed in the psychology literature, but effect sizes and generalisability are contested, and Active Inference does not by itself resolve that dispute. It offers a vocabulary for stating the hypothesis as a difference in specific model parameters.

## Strengths and Cautions

The framework unifies learning, conformity and environmental design under one objective. It is also flexible enough that nearly any observed pattern can be redescribed after the fact, so useful work states which prior or precision differs, and predicts something that could fail. Describing a culture as a single generative model is an idealisation that hides internal disagreement.

## In This Repository

See [[knowledge_base/research/cultural_dynamics]] and [[knowledge_base/research/cultural_evolution]] for the dynamics of cultural change, [[knowledge_base/cognitive/multi_agent_active_inference]] for the multi-agent formalism, and [[knowledge_base/cognitive/communication]] for signalling between agents. The code tree contains no cultural-cognition simulation; the `code/Things/KG_Multi_Agent` documentation describes cultural analytics of language in agent conversations, which is text analysis and not an Active Inference model of culture.

## References

- Veissière, S. P. L., Constant, A., Ramstead, M. J. D., Friston, K. J., Kirmayer, L. J. (2020). Thinking through other minds: a variational approach to cognition and culture. *Behavioral and Brain Sciences*.
- Constant, A., Ramstead, M. J. D., Veissière, S. P. L., Campbell, J. O., Friston, K. J. (2018). A variational approach to niche construction. *Journal of the Royal Society Interface*.

## Related Resources

- [[knowledge_base/cognitive/active_inference]]
- [[knowledge_base/research/cultural_dynamics]]
