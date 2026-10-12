---
title: Collocation Analysis
description: The corpus-based study of words that occur together more often than chance predicts, using association measures such as pointwise mutual information, the t-score and the log-likelihood ratio.
type: concept
status: stable
created: 2026-10-12
updated: 2026-10-12
tags:
  - corpus_linguistics
  - language
  - statistics
semantic_relations:
  - type: relates
    links:
      - "[[language_processing]]"
      - "[[semantic_networks]]"
      - "[[statistical_learning]]"
      - "[[semantic_embeddings]]"
      - "[[mutual_information]]"
---

# Collocation Analysis

## Overview
A collocation is a combination of words that co-occur more often than would be expected if words were placed independently, such as "strong tea" or "heavy rain". Collocation analysis finds and ranks such combinations in a text collection (a corpus). The idea goes back to Firth (1957), who emphasised that a word is characterised by the company it keeps, and it became a quantitative method with the use of large corpora in lexicography (Sinclair, 1991).

## Procedure
1. **Count.** For a target word, record the words that appear within a window around it (a few words on each side) or in a specified grammatical relation, and record how often each word occurs overall.
2. **Compare with chance.** For a word pair $(x,y)$, arrange the counts in a $2\times2$ contingency table with observed frequencies $O_{ij}$ and total $N$. Under independence, the expected count of a cell is the product of its row and column totals divided by $N$, written $E_{ij}$.
3. **Score and rank.** Compute an association measure for each pair and list the highest-scoring pairs.

## Association measures
- **Pointwise mutual information (PMI)**, applied to word association by Church and Hanks (1990):
$$\mathrm{PMI}(x,y)=\log_2\frac{P(x,y)}{P(x)\,P(y)} .$$
It is positive when the pair is more frequent than independence predicts. It tends to give high scores to rare pairs, because small counts produce unstable ratios.
- **t-score**, $(O_{11}-E_{11})/\sqrt{O_{11}}$, which emphasises frequent, well-attested pairs. Manning and Schütze (1999) discuss it with the chi-square statistic.
- **Log-likelihood ratio**, $G^2=2\sum_{ij}O_{ij}\ln(O_{ij}/E_{ij})$. Dunning (1993) argued that tests that assume approximately normal counts perform badly on the rare events that dominate text, and proposed likelihood ratio tests as a more reliable alternative.

Different measures rank pairs differently, so the choice depends on whether the aim is to find idiomatic rare combinations or the typical frequent ones.

## Uses
Collocation analysis supports lexicography, where it shows which words a dictionary entry should list as typical partners, and language teaching, where it identifies natural word combinations. In natural language processing, collocation scores are used to find multiword expressions (Manning and Schütze, 1999), and the same co-occurrence counts underlie distributional approaches to meaning and word embeddings (see [[semantic_embeddings]]).

## Relation to Active Inference
The link is an interpretation of what is being measured. Collocations are statistical regularities of language use, and a learner exposed to text is a system that can extract such regularities, as in [[statistical_learning]]. A generative model of language in the Active Inference sense would need to represent them to predict upcoming words. Collocation analysis itself is a descriptive corpus method, and it makes no claims about how the brain stores or learns these regularities.

## Related concepts
- [[language_processing]] — how language is comprehended and produced
- [[semantic_networks]] — networks of related word meanings
- [[statistical_learning]] — extraction of regularities from experience
- [[semantic_embeddings]] — vector representations built from co-occurrence
- [[mutual_information]] — the quantity of which PMI is the pointwise form

## References
- Firth, J. R. (1957). A synopsis of linguistic theory, 1930-1955. In Studies in Linguistic Analysis (Special volume of the Philological Society). Blackwell.
- Church, K. W., & Hanks, P. (1990). Word association norms, mutual information, and lexicography. Computational Linguistics, 16(1), 22-29. https://aclanthology.org/J90-1003/
- Sinclair, J. (1991). Corpus, Concordance, Collocation. Oxford University Press.
- Dunning, T. (1993). Accurate methods for the statistics of surprise and coincidence. Computational Linguistics, 19(1), 61-74. https://aclanthology.org/J93-1003/
- Manning, C. D., & Schütze, H. (1999). Foundations of Statistical Natural Language Processing. MIT Press.
