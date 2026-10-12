---
title: Domain Adaptation
description: Learning a predictor from a labelled source distribution that performs well on a differently distributed target domain, covering covariate shift, importance weighting and a classical generalisation bound.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - machine_learning
  - distribution_shift
  - generalization
semantic_relations:
  - type: relates
    links:
      - "[[transfer_learning]]"
      - "[[generalization]]"
      - "[[supervised_learning]]"
      - "[[importance_sampling]]"
      - "[[continual_learning]]"
wikidata: Q19246213
---

# Domain Adaptation

## Overview
Standard learning theory assumes that training and test data come from the same distribution. Domain adaptation addresses the case where they do not: a model is trained on a source domain with distribution $P_S$ and must be deployed on a target domain with distribution $P_T$, for which labels are scarce or absent. It is a special case of transfer learning in which the task is unchanged and the data distribution differs.

## Types of shift
- **Covariate shift.** The input distribution changes, $P_S(x)\ne P_T(x)$, while $P(y\mid x)$ is shared. The risk on the target can be estimated from source data by weighting each example with $w(x)=P_T(x)/P_S(x)$, as in importance sampling; the weights are unbounded when the supports differ poorly (Shimodaira, 2000).
- **Label (prior) shift.** $P(y)$ changes while $P(x\mid y)$ is shared.
- **Concept shift.** $P(y\mid x)$ itself changes, which no reweighting of inputs can correct.

## A generalisation bound
Ben-David and colleagues (2010) bounded the target error $\epsilon_T(h)$ of a hypothesis $h$ in a class $\mathcal{H}$ by
$$\epsilon_T(h)\le\epsilon_S(h)+\tfrac12\,d_{\mathcal{H}\Delta\mathcal{H}}(P_S,P_T)+\lambda,$$
where $d_{\mathcal{H}\Delta\mathcal{H}}$ measures how well a classifier from the class can tell the two domains apart and $\lambda$ is the smallest combined error of any single hypothesis on both domains. The bound motivates methods that learn representations in which the domains are hard to distinguish, such as adversarial feature alignment.

## Relation to Active Inference
The link is an analogy rather than a derivation. A hierarchical generative model can treat a change of context as a hidden cause to be inferred, so that lower-level mappings are reused while higher-level context variables adapt. This resembles adaptation by representation reuse, but Active Inference does not supply the bounds or the algorithms above. Any claim that a given model performs domain adaptation needs to be tested on shifted data.

## Related concepts
- [[transfer_learning]] — the broader family of methods that reuse knowledge across tasks
- [[generalization]] — performance on unseen data, the target of the theory above
- [[importance_sampling]] — the reweighting technique used under covariate shift
- [[continual_learning]] — adaptation to changing distributions over time
- [[supervised_learning]] — the setting in which source labels are available

## References
- Shimodaira, H. (2000). Improving predictive inference under covariate shift by weighting the log-likelihood function. Journal of Statistical Planning and Inference, 90, 227-244.
- Ben-David, S., Blitzer, J., Crammer, K., Kulesza, A., Pereira, F., & Vaughan, J. W. (2010). A theory of learning from different domains. Machine Learning, 79, 151-175.
- Pan, S. J., & Yang, Q. (2010). A survey on transfer learning. IEEE Transactions on Knowledge and Data Engineering, 22, 1345-1359.
