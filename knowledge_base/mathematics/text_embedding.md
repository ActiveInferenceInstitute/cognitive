---
title: Text Embedding
description: Methods that map words, sentences or documents to fixed-length numeric vectors, from sparse term weighting to static word vectors and contextual transformer encoders, and how the vectors are compared.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - natural_language_processing
  - representation_learning
  - vector_spaces
semantic_relations:
  - type: relates
    links:
      - "[[semantic_embeddings]]"
      - "[[semantic_search]]"
      - "[[language_modeling]]"
      - "[[transformer_networks]]"
      - "[[machine_readability]]"
---

# Text Embedding

## Overview
A text embedding is a function that converts a piece of text into a vector in $\mathbb{R}^d$, so that texts can be compared, clustered, classified or retrieved by vector operations. It is the text-specific case of [[semantic_embeddings]]. Embeddings exist at several granularities: tokens, words, sentences, passages and whole documents.

## Families of methods
- **Sparse term weighting.** A document is a vector of length equal to the vocabulary, with entries such as term frequency times inverse document frequency, $\mathrm{tf}(t,d)\cdot\log(N/n_t)$, where $N$ is the number of documents and $n_t$ the number containing term $t$. These vectors are high-dimensional and capture lexical overlap, not synonymy.
- **Static word vectors.** Word2vec and GloVe learn one vector per word type from co-occurrence statistics. A sentence vector can be formed by averaging, which discards word order.
- **Contextual encoders.** Transformer models such as BERT (see [[transformer_networks]]) compute token vectors conditioned on the whole input. A fixed-length text vector is obtained by pooling, for example by averaging the token vectors or by using a special classification token.
- **Sentence encoders trained for similarity.** Siamese or contrastive training pulls paired texts together and pushes unpaired ones apart, commonly with a loss of the InfoNCE type, $-\log\frac{\exp(\mathrm{sim}(u,v^{+})/\tau)}{\sum_j\exp(\mathrm{sim}(u,v_j)/\tau)}$ with temperature $\tau$. Models without such training often give poor sentence-level similarity from raw pooled vectors.

## Practical considerations
Texts are limited by the encoder's maximum input length, so long documents are split into chunks. Vectors from different models are not comparable, and an index must be rebuilt when the model changes. Normalising vectors to unit length makes the inner product equal to the cosine. Embeddings reflect the biases and domain of their training data.

## Relation to Active Inference
The link is by analogy and by use. A text embedding is a compressed latent code from which similarity, and in some models the text itself, can be recovered, which parallels the role of latent states in a generative model. Contrastive predictive coding borrows the term "predictive" from a different tradition and is not an implementation of predictive coding in the neuroscientific sense. In practice, embeddings serve as the interface by which [[language_modeling]] systems and knowledge bases such as this one are searched.

## Related concepts
- [[semantic_embeddings]] — the general construction for any modality
- [[semantic_search]] — the main application, retrieval by vector similarity
- [[language_modeling]] — the training setting that produces many encoders
- [[transformer_networks]] — the architecture of contextual encoders
- [[machine_readability]] — why consistent text structure benefits automated retrieval

## References
- Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. Advances in Neural Information Processing Systems, 26, 3111-3119.
- Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation. Proceedings of EMNLP 2014, 1532-1543.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. Proceedings of NAACL-HLT 2019, 4171-4186.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. Proceedings of EMNLP-IJCNLP 2019, 3982-3992.
