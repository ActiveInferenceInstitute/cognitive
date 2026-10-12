---
title: Neural Translation
description: Machine translation by neural sequence-to-sequence models, from recurrent encoder-decoders with attention to transformers, with the probabilistic formulation, training objective and evaluation by BLEU.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - machine_translation
  - sequence_models
  - language
semantic_relations:
  - type: relates
    links:
      - "[[language_modeling]]"
      - "[[attention_mechanisms]]"
      - "[[transformer_networks]]"
      - "[[language_processing_expanded]]"
      - "[[cross_entropy]]"
---

# Neural Translation

## Overview
Neural machine translation maps a source sentence $x=(x_1,\dots,x_S)$ to a target sentence $y=(y_1,\dots,y_T)$ with a single neural network trained end to end. The model defines a conditional distribution
$$P(y\mid x)=\prod_{t=1}^{T}P(y_t\mid y_{<t},x),$$
so translation is conditional language modeling: the target is generated one token at a time given the source and the tokens already produced. At inference, a high-probability output is found by greedy decoding or beam search.

## Architectures
- **Encoder-decoder recurrent networks.** An encoder network reads the source into a vector summary and a decoder network generates the target from it (Sutskever et al., 2014). A single fixed-length vector limits the amount of source information, which hurts long sentences.
- **Attention.** Bahdanau et al. (2015) let the decoder compute, at each step, a weighted average of the encoder states, with weights learned jointly with the rest of the model. This also yields soft word alignments between source and target.
- **Transformers.** Vaswani et al. (2017) removed recurrence and used attention within and between source and target, which allows parallel training over positions and shorter paths between distant words (see [[transformer_networks]]).

Parameters are fitted by minimising the cross-entropy of reference translations, a maximum likelihood objective, usually with subword tokenisation to handle rare words. Output quality is commonly scored by BLEU, a measure of n-gram overlap with reference translations (Papineni et al., 2002). BLEU correlates only imperfectly with human judgement, and human evaluation remains the more reliable test.

## Relation to Active Inference
There is no direct derivation connecting neural translation to Active Inference. The model is a supervised, passive predictor of text conditioned on text, with no actions, preferences or free energy objective. Accounts of human translation and interpreting would treat the translator as an agent inferring communicative intentions, a more demanding model than conditional next-token prediction. Any connection is at the level of the shared idea of prediction in language processing.

## Related concepts
- [[language_modeling]] — the unconditional version of the task
- [[attention_mechanisms]] — the mechanism that removed the fixed-vector bottleneck
- [[transformer_networks]] — the dominant architecture
- [[recurrent_networks]] — the earlier encoder-decoder components
- [[cross_entropy]] — the training objective
- [[language_processing_expanded]] — human language processing

## References
- Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. Advances in Neural Information Processing Systems, 27, 3104-3112.
- Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. International Conference on Learning Representations.
- Vaswani, A., et al. (2017). Attention is all you need. Advances in Neural Information Processing Systems, 30, 5998-6008.
- Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). BLEU: a method for automatic evaluation of machine translation. Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, 311-318.
