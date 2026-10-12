---
title: Language Modeling
description: The task of assigning probabilities to word or token sequences, from n-gram models to neural and transformer-based models, with perplexity as the standard evaluation measure and a comparison to predictive processing.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - language
  - probabilistic_models
  - sequence_models
semantic_relations:
  - type: relates
    links:
      - "[[language_processing_expanded]]"
      - "[[cross_entropy]]"
      - "[[transformer_networks]]"
      - "[[attention_mechanisms]]"
      - "[[predictive_coding]]"
---

# Language Modeling

## Overview
A language model is a probability distribution over sequences of symbols. For a sequence $w_1,\dots,w_T$ the chain rule of probability gives
$$P(w_1,\dots,w_T)=\prod_{t=1}^{T}P(w_t\mid w_1,\dots,w_{t-1}),$$
so modelling a language reduces to predicting the next token from its history. Such models are used for speech recognition, machine translation, text generation and as general-purpose components in language technology.

## Model families
- **n-gram models** truncate the history to the previous $n-1$ tokens and estimate conditional probabilities from counts. Unseen sequences make raw counts unreliable, so smoothing and back-off methods redistribute probability mass.
- **Neural models** represent words as learned vectors and compute the conditional distribution with a neural network. Bengio and colleagues (2003) used a feed-forward network with a fixed context window, and recurrent networks later removed the fixed window, though they struggle with distant dependencies (see [[long_term_dependencies]]).
- **Transformer models** replace recurrence with attention over all earlier tokens (see [[transformer_networks]] and [[attention_mechanisms]]) and are trained at scale on large text collections.

Models are trained by minimising the cross-entropy between the empirical next-token distribution and the model prediction, which is equivalent to maximising likelihood. The standard intrinsic measure is perplexity, $\mathrm{PPL}=\exp\!\big(-\tfrac1T\sum_t\ln P(w_t\mid w_{<t})\big)$, the exponentiated average negative log-probability per token. Lower perplexity means the model finds held-out text less surprising. Perplexity is comparable only between models that share the same tokenisation. Shannon (1951) estimated the entropy of printed English by human prediction experiments, an early form of the same idea.

## Relation to Active Inference
Autoregressive language models minimise a form of prediction error on the next token, which parallels the role of prediction in predictive processing accounts of language comprehension. The parallel is an analogy. A standard language model is trained offline on text, has no actions and no preferences over outcomes, and does not minimise expected free energy. Active Inference accounts of communication model speaker and listener as agents with generative models over hidden states and intentions, which is a separate and more demanding formulation than next-token prediction.

## Related concepts
- [[language_processing_expanded]] — human language processing that models are compared to
- [[language_comprehension]] — prediction during comprehension
- [[cross_entropy]] — the training loss and basis of perplexity
- [[transformer_networks]] — the dominant current architecture
- [[predictive_coding]] — prediction-error account of cortical processing
- [[neural_translation]] — conditional language modeling for translation

## References
- Shannon, C. E. (1951). Prediction and entropy of printed English. Bell System Technical Journal, 30, 50-64.
- Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). A neural probabilistic language model. Journal of Machine Learning Research, 3, 1137-1155.
- Jurafsky, D., & Martin, J. H. (2009). Speech and Language Processing (2nd ed.). Prentice Hall.
