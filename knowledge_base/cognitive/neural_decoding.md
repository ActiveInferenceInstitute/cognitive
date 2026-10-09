---
title: Neural Decoding
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - neural-coding
  - statistics
  - neurotechnology
semantic_relations:
  - type: relates
    links:
      - "[[neural_coding]]"
      - "[[bayesian_inference]]"
      - "[[information_theory]]"
---

# Neural Decoding

## Overview
Neural decoding is the inference of a stimulus, behavior or cognitive state from recorded neural activity. It is the inverse of encoding, which describes how stimuli determine neural responses. Decoding quantifies what information a population carries and underlies brain-machine interfaces.

## Mechanism
The encoding model gives $p(r\mid s)$, the probability of responses $r$ given a stimulus $s$. Bayesian decoding inverts it: $p(s\mid r)\propto p(r\mid s)\,p(s)$. For independent Poisson neurons with tuning curves $f_i(s)$ and counts $r_i$ in a window, the log-posterior is, up to a constant, $\sum_i \big(r_i \log f_i(s) - f_i(s)\big) + \log p(s)$. Maximum-likelihood and maximum a posteriori estimates follow, and the posterior width quantifies uncertainty. Rieke and colleagues present decoding of spike trains in this information-theoretic frame.

Practical decoders include the population vector, linear regression and optimal linear estimators, Kalman filters for time-varying states such as hand position (used by Wu and colleagues in motor cortex), and classifiers or deep networks for high-dimensional data. In human neuroimaging, multivariate pattern analysis decodes categories or perceived and imagined contents from fMRI voxel patterns: Haxby and colleagues showed distributed category information in ventral temporal cortex, and Kamitani and Tong decoded orientation and attended features from early visual cortex.

Decoding accuracy has a limited interpretation. That information can be read out by an analyst does not show that downstream brain areas use it; the decoder may exploit features the brain ignores, and a good decoder does not identify the neural code. Controls such as cross-validation and permutation tests guard against overfitting, which is a serious risk with many features and few trials.

## Role in Active Inference
The brain itself is cast in the framework as a decoder: perception is inversion of a generative model, producing a posterior over causes given sensory data, which is formally the same operation as Bayesian decoding by an experimenter. The experimenter's decoder, however, is a separate analysis tool. Decoding of neural data can be used to test predictions of the framework, for instance whether population activity carries uncertainty in line with posterior beliefs, but a successful decoder is evidence about information content only, not about the free-energy principle.

## Related concepts
- [[neural_coding]] — encoding principles decoders rely on
- [[population_coding]] — joint activity read out by decoders
- [[brain_machine_interfaces]] — real-time application of decoding
- [[bayesian_inference]] — statistical basis of probabilistic decoders
- [[information_theory]] — quantifying information in responses
- [[perceptual_inference]] — the brain's own inversion problem

## References
- Rieke, F., Warland, D., de Ruyter van Steveninck, R., & Bialek, W. (1997). Spikes: Exploring the Neural Code. MIT Press.
- Haxby, J. V., et al. (2001). Distributed and overlapping representations of faces and objects in ventral temporal cortex. Science.
- Kamitani, Y., & Tong, F. (2005). Decoding the visual and subjective contents of the human brain. Nature Neuroscience.
