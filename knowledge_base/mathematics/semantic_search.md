---
title: Semantic Search
description: Retrieval of documents by meaning rather than keyword overlap, using vector embeddings, similarity scoring and approximate nearest-neighbour indexes, and how it differs from lexical search.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - information_retrieval
  - embeddings
  - nearest_neighbours
semantic_relations:
  - type: relates
    links:
      - "[[semantic_embeddings]]"
      - "[[text_embedding]]"
      - "[[retrieval]]"
      - "[[machine_readability]]"
      - "[[knowledge_graphs]]"
wikidata: Q1891170
---

# Semantic Search

## Overview
Semantic search retrieves items whose meaning matches a query, even when they share few or no words with it. A query $q$ and each document $d$ are mapped by an encoder to vectors $\mathbf{q},\mathbf{d}\in\mathbb{R}^n$ (see [[semantic_embeddings]] and [[text_embedding]]), and documents are ranked by a similarity score, most often the cosine $\mathbf{q}\cdot\mathbf{d}/(\lVert\mathbf{q}\rVert\lVert\mathbf{d}\rVert)$ or the inner product.

## Method
1. **Indexing.** Every document, or every passage of a long document, is encoded once and the vectors are stored.
2. **Querying.** The query is encoded with the same model and the $k$ nearest vectors are returned.
3. **Approximate search.** Exact search costs $O(Nn)$ per query for $N$ vectors. Large collections use approximate nearest-neighbour structures, such as the hierarchical navigable small world graph, which trade a small loss in recall for sub-linear query time.
4. **Re-ranking.** A slower cross-encoder or a lexical scorer may reorder the top candidates.

Lexical search, in contrast, scores documents by weighted term matches, for example TF-IDF or BM25, over a sparse vector with one dimension per vocabulary term. It is precise for rare names, identifiers and exact phrases, and weak when the query uses synonyms or paraphrase. Semantic search has the opposite profile, so practical systems often combine the two scores.

## Evaluation and limits
Quality is measured on labelled relevance judgements with precision and recall at $k$, mean reciprocal rank or normalised discounted cumulative gain. Dense retrievers can fail out of domain, on negation, and on queries that depend on exact tokens. Chunking choices matter, because an embedding of a long passage averages its topics.

## Relation to Active Inference and the vault
The link is practical rather than theoretical. Semantic search supplies the machine-readable access layer for knowledge bases such as this one, which is the concern of [[machine_readability]]. Informally, retrieval resembles cue-driven recall from [[semantic_memory]], and a retrieved passage can be treated as additional evidence that updates an agent's beliefs. No derivation from the free energy principle is implied.

## Related concepts
- [[semantic_embeddings]] — the vector representations that define similarity
- [[text_embedding]] — encoders that map passages and queries to vectors
- [[retrieval]] — the broader cognitive notion of recalling stored information
- [[knowledge_graphs]] — structured alternative that supports explicit queries over relations
- [[machine_readability]] — why consistent structure makes a corpus searchable

## References
- Salton, G., Wong, A., & Yang, C. S. (1975). A vector space model for automatic indexing. Communications of the ACM, 18, 613-620.
- Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval. Cambridge University Press.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. Proceedings of EMNLP-IJCNLP 2019, 3982-3992.
- Malkov, Y. A., & Yashunin, D. A. (2020). Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. IEEE Transactions on Pattern Analysis and Machine Intelligence, 42, 824-836.
