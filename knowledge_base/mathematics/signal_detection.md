---
title: Signal Detection
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - signal_detection_theory
  - decision_theory
  - psychophysics
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/mathematics/bayes_theorem]]"
      - "[[knowledge_base/mathematics/probability_distributions]]"
      - "[[knowledge_base/cognitive/decision_making]]"
      - "[[knowledge_base/cognitive/perceptual_inference]]"
---

# Signal Detection

## Overview
Signal detection theory (SDT) analyses binary decisions made on noisy evidence, such as whether a stimulus was present. It separates two things that raw accuracy conflates: sensitivity, how well the evidence distributions for signal and noise can be told apart, and the decision criterion, how much evidence the observer demands before responding "signal".

## Definition / Formalism
In the standard equal-variance Gaussian model the internal evidence is $x\sim\mathcal N(0,1)$ on noise trials and $x\sim\mathcal N(d',1)$ on signal trials. A response "yes" is given when $x>\lambda$. Then

$$H=\Phi(d'-\lambda),\qquad F=\Phi(-\lambda),\qquad d'=\Phi^{-1}(H)-\Phi^{-1}(F),\qquad c=-\tfrac12\big[\Phi^{-1}(H)+\Phi^{-1}(F)\big],$$

where $H$ is the hit rate, $F$ the false-alarm rate, $\Phi$ the standard normal CDF, and $c$ the criterion measured from the unbiased point $d'/2$. For $H=0.84$ and $F=0.16$, $d'=1.989$ and $c=0$.

The likelihood ratio at evidence $x$ is $\ell(x)=p(x\mid\text{signal})/p(x\mid\text{noise})=\exp\!\big[d'(x-d'/2)\big]$ in this model; at $d'=1,\,x=0.8$ it is 1.350. The Bayes-optimal rule responds "yes" when $\ell(x)>\frac{P(\text{noise})}{P(\text{signal})}\cdot\frac{u_{\text{CR}}-u_{\text{FA}}}{u_{\text{H}}-u_{\text{M}}}$, with utilities $u$ for correct rejection, false alarm, hit and miss. With equal priors and symmetric payoffs this is $\lambda=d'/2$ and accuracy $\Phi(d'/2)$; at $d'=1$ that is $0.6915$, which a $10^6$-trial simulation reproduced (0.6916).

Sweeping $\lambda$ traces the ROC curve, whose area is $\Phi(d'/\sqrt2)$ in this model (0.7602 for $d'=1$, simulated 0.7605). The equal-variance assumption is testable: unequal slopes of the z-transformed ROC indicate unequal variances, in which case $d'$ is not a single number.

## Role in Active Inference
The tie is that SDT is the special case of Bayesian decision-making with two hypotheses. The posterior log odds equal log prior odds plus the log likelihood ratio, so a criterion is an encoded prior and a payoff structure. In active inference the same quantities appear as the prior $D$, the likelihood $A$, and preferences $C$ over outcomes, and a two-state, two-observation model with $A=\begin{pmatrix}0.9&0.1\\0.1&0.9\end{pmatrix}$ is a discrete-observation analogue of SDT. The repository does not implement SDT analyses; it is a way to interpret fitted behaviour, with the caution that an observer's criterion and the agent's precision parameters are distinct constructs.

## Related concepts
- [[bayes_theorem]] — posterior odds as prior odds times likelihood ratio
- [[decision_making]] — decisions under uncertainty
- [[perceptual_inference]] — noisy sensory evidence about hidden causes
- [[probability_distributions]] — Gaussian evidence distributions
- [[channel_capacity]] — the information-theoretic limit on discriminability
- [[bayesian_models]] — priors and likelihoods in cognitive models

## References
- Green, D. M. and Swets, J. A. (1966). *Signal Detection Theory and Psychophysics*. Wiley.
- Macmillan, N. A. and Creelman, C. D. (2005). *Detection Theory: A User's Guide*, 2nd ed. Lawrence Erlbaum.
