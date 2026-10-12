---
title: Semantic Embeddings
description: Vector representations that place words, sentences, images or concepts in a continuous space so that geometric proximity reflects similarity of meaning, with the main constructions and similarity measures.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - representation_learning
  - vector_spaces
  - distributional_semantics
semantic_relations:
  - type: relates
    links:
      - "[[text_embedding]]"
      - "[[semantic_search]]"
      - "[[dimension_reduction]]"
      - "[[semantic_memory]]"
      - "[[hyperspace_ontology]]"
---

# Semantic Embeddings

## Overview
A semantic embedding is a map $f:\mathcal{X}\to\mathbb{R}^d$ from discrete or structured items, such as words, documents, images or graph nodes, into a dense vector space in which distance or angle between vectors approximates semantic relatedness. The dimension $d$ is usually far smaller than the size of the vocabulary, so embeddings are a form of [[dimension_reduction]] that is learned from data rather than specified by hand.

## Foundations
The theoretical basis is the distributional hypothesis, associated with Harris and Firth: items that occur in similar contexts tend to have similar meanings. Three families of construction follow from it.

- **Count-based and factorisation methods.** A word-context co-occurrence matrix is reweighted, for example by pointwise mutual information, and reduced by singular value decomposition. Latent semantic analysis is the classical case.
- **Predictive neural models.** Skip-gram and related models learn vectors by predicting context words from a target word. With negative sampling, skip-gram can be interpreted as implicitly factorising a shifted word-context pointwise mutual information matrix.
- **Contextual models.** Transformer encoders produce a different vector for each occurrence of a word, depending on its sentence, so that one surface form can occupy different regions for different senses.

Similarity is most often measured by the cosine, $\cos\theta = \dfrac{\mathbf{x}\cdot\mathbf{y}}{\lVert\mathbf{x}\rVert\,\lVert\mathbf{y}\rVert}$, which ignores vector length. Some embeddings show approximate linear regularities, such as the offset between "king" and "queen" resembling that between "man" and "woman", but this holds only roughly and depends on the training corpus and the evaluation protocol.

## Limits
Embeddings inherit statistical biases from their training data, compress polysemy into a single point in static models, and offer no guarantee that geometric neighbours are semantically substitutable. Distances in high dimensions also concentrate, which weakens nearest-neighbour contrasts.

## Relation to Active Inference
The link is an analogy and an engineering convenience, not a derivation. A learned embedding plays a role similar to a latent state in a generative model: a compact code from which observations can be predicted, as in [[variational_autoencoders]]. Active Inference treats such codes as beliefs with uncertainty, whereas most embeddings are point estimates. Embeddings are mainly relevant here as a way to give discrete concepts a continuous geometry, as in [[hyperspace_ontology]].

## Related concepts
- [[text_embedding]] — the specialisation to text, with practical pooling and training choices
- [[semantic_search]] — retrieval by nearest neighbours in embedding space
- [[semantic_memory]] — the cognitive counterpart of stored conceptual knowledge
- [[semantic_networks]] — a symbolic, graph-based alternative to vector representations
- [[dimension_reduction]] — the general problem of compressing high-dimensional data

## References
- Harris, Z. S. (1954). Distributional structure. Word, 10, 146-162.
- Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K., & Harshman, R. (1990). Indexing by latent semantic analysis. Journal of the American Society for Information Science, 41, 391-407.
- Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. Advances in Neural Information Processing Systems, 26, 3111-3119.
