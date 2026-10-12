---
title: Gumbel-Softmax
description: The Gumbel-softmax, or Concrete, distribution is a continuous relaxation of categorical sampling that allows gradients to pass through discrete random choices, using Gumbel noise and a temperature-controlled softmax.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - deep_learning
  - sampling
  - reparameterization
semantic_relations:
  - type: relates
    links:
      - "[[softmax_function]]"
      - "[[temperature_parameter]]"
      - "[[variational_autoencoders]]"
      - "[[policy_selection]]"
      - "[[probability_distributions]]"
---

# Gumbel-Softmax

## Overview
Training a model by gradient descent requires derivatives through every operation, but drawing a sample from a categorical distribution is a discrete, non-differentiable step. The Gumbel-softmax estimator, proposed independently as the Gumbel-softmax and the Concrete distribution, replaces the discrete sample with a continuous vector on the probability simplex. This permits reparameterised gradient estimates for models with categorical latent variables or discrete choices.

## Gumbel-max trick
Let a categorical distribution have probabilities $\pi_1,\dots,\pi_K$. Draw $u_i\sim\mathrm{Uniform}(0,1)$ independently and set $g_i=-\log(-\log u_i)$, which is a standard Gumbel variable. Then
$$k=\arg\max_i\,\big(\log\pi_i+g_i\big)$$
is distributed as $\mathrm{Categorical}(\pi)$, so that $P(k=i)=\pi_i$. The sampling is exact and the randomness is separated from the parameters $\pi$, which is the essential property of a reparameterisation. The argmax, however, has zero gradient almost everywhere.

## Relaxation
The Gumbel-softmax replaces the argmax by a softmax with temperature $\tau>0$:
$$y_i=\frac{\exp\big((\log\pi_i+g_i)/\tau\big)}{\sum_{j=1}^{K}\exp\big((\log\pi_j+g_j)/\tau\big)}.$$
The vector $y$ lies on the simplex and is differentiable in $\pi$. As $\tau\to 0$ samples approach one-hot vectors distributed as the original categorical, while the gradients become high in variance. As $\tau\to\infty$ samples tend towards the uniform vector, and the gradients are low in variance but biased relative to the discrete model. Temperature is therefore annealed or tuned. The straight-through variant uses the one-hot argmax in the forward pass and the relaxed gradient in the backward pass, which gives biased gradients but a discrete forward value.

The relaxed density is not the same as the categorical, and the objective of a model trained with relaxed samples differs from the one with discrete samples, so evaluation is usually performed with true discrete sampling.

## Relation to Active Inference
Discrete Active Inference selects actions by sampling from a softmax of negative expected free energy with a precision parameter, and the Gumbel-max trick shows that this softmax sampling is exactly the argmax of perturbed scores. The temperature of the relaxation plays the role of the inverse precision. Gumbel-softmax is used in deep active inference implementations to make the sampling of actions or discrete latent states differentiable. This is an engineering tool and not part of the theoretical formalism.

## Related concepts
- [[softmax_function]] — the normalised exponential at the heart of the relaxation
- [[temperature_parameter]] — controls sharpness and the bias-variance trade-off
- [[variational_autoencoders]] — models that use the estimator for discrete latents
- [[policy_selection]] — softmax sampling over policies in Active Inference
- [[action_selection]] — the mapping from policy beliefs to an action
- [[probability_distributions]] — the categorical and Gumbel families involved

## References
- Jang, E., Gu, S., & Poole, B. (2017). Categorical reparameterization with Gumbel-softmax. International Conference on Learning Representations.
- Maddison, C. J., Mnih, A., & Teh, Y. W. (2017). The Concrete distribution: A continuous relaxation of discrete random variables. International Conference on Learning Representations.
- Gumbel, E. J. (1954). Statistical Theory of Extreme Values and Some Practical Applications. National Bureau of Standards Applied Mathematics Series 33.
