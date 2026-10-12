---
title: Structured Prediction
description: Supervised learning in which the output is a structured object such as a sequence, tree or labelling of a graph, with the scoring, inference and training methods that exploit output dependencies.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - machine_learning
  - graphical_models
  - inference
semantic_relations:
  - type: relates
    links:
      - "[[conditional_random_fields]]"
      - "[[message_passing]]"
      - "[[factor_graphs]]"
      - "[[probabilistic_graphical_models]]"
      - "[[markov_random_fields]]"
---

# Structured Prediction

## Overview
Structured prediction is the problem of learning a mapping from an input $x$ to an output $y$ drawn from a set $\mathcal{Y}$ that is very large and has internal structure, such as a label sequence for a sentence, a parse tree, or a pixel labelling. The number of candidate outputs grows exponentially with the size of the object, so outputs cannot be treated as independent classes. Their components depend on each other, and the learner must represent and exploit those dependencies.

## Formulation
A common formulation uses a feature map $\phi(x,y)$ and a score $s(x,y)=\mathbf{w}^{\top}\phi(x,y)$. Prediction solves
$$\hat y=\arg\max_{y\in\mathcal{Y}} s(x,y).$$
Tractability comes from decomposing $\phi$ over small parts, for example the cliques of a graph, so that the score is a sum of local terms. If the graph is a chain or tree, maximisation is carried out exactly by dynamic programming (the Viterbi algorithm) and marginals by the forward-backward algorithm, both instances of [[message_passing]] on a [[factor_graphs]] representation. On graphs with cycles, approximate inference or relaxations are used.

## Training
- **Probabilistic.** A conditional random field defines $p(y\mid x)=\exp(\mathbf{w}^{\top}\phi(x,y))/Z(x)$ and maximises conditional log-likelihood, which needs the partition function $Z(x)$ and hence marginal inference (see [[conditional_random_fields]]).
- **Margin-based.** Structured support vector machines require the correct output to beat every alternative by a margin that grows with a task loss such as Hamming distance, and need only loss-augmented maximisation.
- **Neural.** Learned scoring functions can be combined with structured decoding layers, although the global argmax then typically relies on approximate search.

## Relation to Active Inference
The link is shared machinery, not a common theory. Generative models in Active Inference with several hidden-state factors across time are also factor graphs, and belief updating by variational message passing uses the same local-computation idea (see [[variational_message_passing]]). The objectives differ: structured prediction is typically discriminative, optimising $p(y\mid x)$ for a supervised task, whereas Active Inference scores a generative joint model and selects actions by expected free energy.

## Related concepts
- [[conditional_random_fields]] — the standard probabilistic model for structured outputs
- [[message_passing]] — exact and approximate inference over the output graph
- [[factor_graphs]] — the representation that makes decomposition explicit
- [[markov_random_fields]] — undirected models underlying the score decomposition
- [[probabilistic_graphical_models]] — the general framework

## References
- Lafferty, J., McCallum, A., & Pereira, F. (2001). Conditional random fields: Probabilistic models for segmenting and labeling sequence data. Proceedings of the 18th International Conference on Machine Learning, 282-289.
- Tsochantaridis, I., Joachims, T., Hofmann, T., & Altun, Y. (2005). Large margin methods for structured and interdependent output variables. Journal of Machine Learning Research, 6, 1453-1484.
- Koller, D., & Friedman, N. (2009). Probabilistic Graphical Models: Principles and Techniques. MIT Press.
