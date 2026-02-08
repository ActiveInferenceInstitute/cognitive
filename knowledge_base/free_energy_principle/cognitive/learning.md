---
title: "Learning as Model Parameter Optimization Under the FEP"
type: concept
status: active
created: 2025-01-01
updated: 2025-02-06
tags:
  - free_energy_principle
  - learning
  - synaptic_plasticity
  - structure_learning
  - bayesian_model_comparison
  - active_inference
semantic_relations:
  - type: foundation
    links:
      - [[knowledge_base/free_energy_principle/mathematics/core_principle|Core Principle]]
      - [[knowledge_base/free_energy_principle/mathematics/variational_free_energy|Variational Free Energy]]
  - type: relates
    links:
      - [[perception|Perception]]
      - [[attention|Attention]]
      - [[decision_making|Decision Making]]
      - [[knowledge_base/free_energy_principle/biology/neural_systems|Neural Systems]]
  - type: implements
    links:
      - [[knowledge_base/free_energy_principle/mathematics/information_geometry|Information Geometry]]
---

# Learning as Model Parameter Optimization Under the FEP

## Overview

Under the Free Energy Principle, learning is the process of updating the **parameters** of the generative model to improve its predictive accuracy over time. While perception updates **states** (beliefs about the current hidden causes of sensory input) on a fast timescale, learning updates **parameters** (the structure and tuning of the mapping between causes and observations) on a slower timescale.

This distinction maps onto a fundamental separation in neuroscience:
- **Perception** (fast): Changes in neural activity (firing rates, population codes)
- **Learning** (slow): Changes in synaptic weights (long-term potentiation/depression, structural plasticity)

The FEP provides a unified framework that derives specific learning rules -- including Hebbian learning, error-driven learning, and Bayesian model selection -- as different manifestations of the same objective: minimizing variational free energy with respect to model parameters.

## Mathematical Foundation

### Parameter Learning as Free Energy Minimization

Given observations `o`, hidden states `s`, and model parameters `theta`, the variational free energy is:

```
F[q(s), theta] = D_KL[q(s) || p(s | theta)] - E_q[ln p(o | s, theta)]
```

Learning minimizes F with respect to theta (on a slow timescale):

```
dtheta/dt = -eta * partial F / partial theta
```

Where `eta` is a learning rate (much smaller than the inference rate).

Expanding the gradient:

```
partial F / partial theta = partial/partial theta { D_KL[q(s) || p(s|theta)] - E_q[ln p(o|s,theta)] }
```

This decomposes into:
- **Prior gradient**: `partial D_KL / partial theta` -- how to change theta to make the prior more compatible with the posterior
- **Likelihood gradient**: `-partial E_q[ln p(o|s,theta)] / partial theta` -- how to change theta to improve data fit

### Expectation-Maximization (EM) Connection

The FEP learning process is mathematically equivalent to the **Expectation-Maximization** algorithm:

**E-step** (perception): Fix theta, optimize q(s):
```
q*(s) = argmin_q F[q, theta] = p(s | o, theta)
```

**M-step** (learning): Fix q, optimize theta:
```
theta* = argmin_theta F[q*, theta] = argmax_theta E_{q*}[ln p(o, s | theta)]
```

In the brain, these steps run continuously and simultaneously, not in strict alternation. Perception is faster than learning, so by the time parameters change appreciably, the perceptual inference has already converged -- an approximation to the idealized EM procedure.

## Deriving Specific Learning Rules

### Hebbian Learning

Consider a linear Gaussian generative model:

```
o = W * s + z,    z ~ N(0, Pi^{-1})
p(s) = N(0, I)
```

Where `W` is the weight matrix (parameters to learn).

The free energy gradient with respect to W:

```
partial F / partial W = -Pi * (o - W * mu_s) * mu_s^T
                      = -Pi * epsilon * mu_s^T
```

Where `epsilon = o - W * mu_s` is the prediction error and `mu_s = E_q[s]` is the inferred state.

The learning rule becomes:

```
dW/dt = eta * Pi * epsilon * mu_s^T
```

This is a **precision-weighted Hebbian learning rule**: the weight change is proportional to the product of the prediction error and the inferred state, scaled by precision.

**Connection to neuroscience**:
- `epsilon` = presynaptic activity (prediction error neurons)
- `mu_s` = postsynaptic activity (state representation neurons)
- `Pi` = neuromodulatory gain (precision/attention)
- The rule is Hebbian (correlated pre- and post-synaptic activity strengthens connections)
- With precision weighting, it becomes **attention-modulated Hebbian learning**

### Error-Driven Learning (Delta Rule)

For a simpler model where states are directly observed:

```
dW/dt = eta * (o - W * s) * s^T = eta * epsilon * s^T
```

This is the **delta rule** -- the fundamental supervised learning algorithm. Under the FEP, it emerges naturally as gradient descent on free energy.

### Bayesian Learning Rules

When we maintain uncertainty about parameters (not just point estimates), we get Bayesian learning:

```
q(theta) = N(mu_theta, Sigma_theta)
```

The update equations become:

```
dmu_theta/dt = -Sigma_theta * partial F / partial mu_theta
dSigma_theta/dt = -partial F / partial Sigma_theta
```

The mean update is the natural gradient (weighted by parameter uncertainty), and the covariance update tracks how certain we are about the parameters.

**Properties of Bayesian learning**:
- Automatic regularization (Occam's razor through the complexity cost)
- Uncertainty tracking (know what we don't know)
- Efficient data use (uncertain parameters change more)
- Protection against overfitting (complexity penalty prevents excessive fitting)

## Types of Learning Under the FEP

### 1. Perceptual Learning

Updating the likelihood model `p(o | s, theta)` to better predict observations from states.

**Example**: Learning to read -- the visual system learns the mapping from letter shapes (states) to retinal images (observations).

```
dtheta_likelihood/dt = -eta * partial/partial theta E_q[ln p(o | s, theta)]
```

**Neural correlate**: Synaptic plasticity in sensory cortices (V1, A1, S1).

### 2. Associative Learning

Updating the prior/transition model `p(s_t | s_{t-1}, theta)` to better predict state transitions.

**Example**: Classical conditioning -- learning that a bell (state 1) predicts food (state 2).

```
dtheta_transition/dt = -eta * partial/partial theta D_KL[q(s_t) || p(s_t | s_{t-1}, theta)]
```

**Neural correlate**: Synaptic plasticity in hippocampus and association cortices.

### 3. Preference Learning

Updating the prior preference model `p(o)` (C vector in discrete models) to adjust what the agent finds rewarding.

**Example**: Developing a taste for coffee -- changing prior preferences about bitter sensory input.

```
dC/dt = -eta * partial F / partial C
```

**Neural correlate**: Dopaminergic modulation of the basal ganglia.

### 4. Precision Learning

Learning the precision (inverse variance) of sensory and prior distributions.

**Example**: Learning that visual information is more reliable than auditory in a noisy environment.

```
dPi/dt = -eta * partial F / partial Pi
```

**Neural correlate**: Long-term changes in neuromodulatory gain (e.g., baseline acetylcholine levels).

## Structure Learning: Bayesian Model Reduction

### Beyond Parameter Estimation

Sometimes the structure of the generative model itself needs to change -- adding or removing state factors, changing the connectivity pattern, or adjusting the model order. This is **structure learning** or **model selection**.

### Bayesian Model Comparison

Given competing models `m_1, m_2, ...`, each with different structures, Bayesian model comparison evaluates:

```
p(m_i | o) proportional to p(o | m_i) * p(m_i)
```

Where `p(o | m_i) = integral p(o | theta, m_i) * p(theta | m_i) d theta` is the model evidence.

The free energy provides an approximate model evidence:

```
ln p(o | m_i) approx -F[q*, m_i]
```

The model with the lowest free energy (highest evidence) is preferred. Since free energy = complexity - accuracy, this automatically penalizes overly complex models (Occam's razor).

### Bayesian Model Reduction (BMR)

**Bayesian model reduction** (Friston & Penny, 2011) is a computationally efficient method for comparing nested models. Instead of re-estimating parameters for each model, BMR analytically computes the change in free energy when parameters are "reduced" (set to zero or to their prior values).

For a full model with posterior `q(theta) = N(mu_theta, Sigma_theta)` and a reduced model with prior `p_0(theta) = N(0, Sigma_0)`:

```
Delta_F = F_reduced - F_full
        approx 1/2 * [mu_theta^T * (Sigma_theta^{-1} - Sigma_0^{-1}) * mu_theta
                       - ln|Sigma_theta * Sigma_0^{-1}|
                       + tr(Sigma_0^{-1} * Sigma_theta) - dim]
```

If `Delta_F < 0`, the reduced model has lower free energy (higher evidence) and should be preferred. This means removing those parameters IMPROVES the model.

**Neural interpretation**: BMR corresponds to **synaptic pruning** -- the elimination of unnecessary connections. During development and sleep, the brain performs a form of BMR, removing synaptic connections whose parameters are close to zero (not contributing to model evidence).

### Sleep and Offline Learning

BMR provides a compelling account of the function of sleep:

1. **During waking**: The brain accumulates experience, updating parameters to fit data (increasing accuracy)
2. **During NREM sleep**: The brain performs BMR, pruning unnecessary complexity (reducing complexity)
3. **During REM sleep**: The brain tests the reduced model by generating synthetic data (dreaming)

This explains:
- Why sleep deprivation impairs learning (no time for pruning)
- Why dreams often involve bizarre combinations (testing reduced models)
- Why morning brings clarity (complexity has been reduced)

## The Concentration Parameter and Learning Rates

### Dirichlet Learning

In discrete state-space models, the likelihood (A matrix) and transition (B matrix) are typically given Dirichlet priors:

```
p(A_j) = Dir(a_j)  -- Dirichlet prior over the j-th column of A
```

Learning updates the concentration parameters `a` based on experience:

```
a_j(new) = a_j(old) + o * q(s=j)^T
```

Where `o` is the one-hot observation vector and `q(s=j)` is the posterior probability of state j.

**Key property**: The learning rate is **adaptive** and **decreasing**. Early in learning (low concentrations), each observation has a large effect. Late in learning (high concentrations), each observation has a diminishing effect. This implements optimal Bayesian updating.

**Connection to developmental critical periods**: Young brains (low concentrations) learn rapidly; mature brains (high concentrations) are more stable. This explains why early experience has outsized influence on adult perception and behavior.

### Learning Rate as a Meta-Parameter

The learning rate `eta` can itself be optimized as a hyperparameter:

```
eta* = argmin_eta F[q, theta(eta)]
```

This creates a hierarchy of timescales:
- **Inference** (milliseconds): Update state beliefs
- **Fast learning** (minutes): Update volatile parameters
- **Slow learning** (days/years): Update stable parameters
- **Meta-learning** (lifetime): Adjust learning rates themselves

## Transfer Learning and Generalization

### The FEP Account

Under the FEP, generalization occurs naturally through the structure of the generative model:

1. **Hierarchical abstraction**: Higher levels encode more abstract, transferable features
2. **Prior transfer**: Priors learned in one context apply to novel contexts (if the generative structure is shared)
3. **Model evidence**: Novel situations that are well-predicted by existing models have high model evidence (low free energy)

### One-Shot Learning

The Bayesian framework enables one-shot learning when the generative model has appropriate structure:

```
p(o_new | o_old, m) = integral p(o_new | theta, m) * p(theta | o_old, m) d theta
```

If the posterior `p(theta | o_old, m)` is informative (the model structure constrains the parameters), then a single example can dramatically update predictions about new instances.

**Example**: After seeing one example of a new object category, the brain's hierarchical model can generate predictions about how that object would look from different angles, in different lighting, at different sizes -- because the generative model encodes these transformations as structural features.

## Active Learning

### Curiosity as EFE Minimization

Under the FEP, learning is not passive but **active**. The agent selects actions that maximize information gain about model parameters:

```
G_learning(pi) = -E_q[D_KL[q(theta | o_tau, pi) || q(theta | pi)]]
```

This is the expected reduction in uncertainty about parameters given observations obtained under policy pi. Policies that lead to maximally informative observations are preferred -- this is **curiosity** at the level of parameters.

### Optimal Experimental Design

Active learning under the FEP is formally equivalent to **Bayesian optimal experimental design** (Lindley, 1956):

```
experiment* = argmax_{experiment} I(theta; o | experiment)
```

Choose the experiment (action) that maximizes mutual information between parameters and observations. The FEP agent naturally performs optimal experimental design as a consequence of EFE minimization.

## Catastrophic Forgetting and Continual Learning

### The Problem

Neural networks suffer from **catastrophic forgetting**: learning new tasks destroys performance on old tasks. The FEP provides a principled solution through the complexity cost.

### Elastic Weight Consolidation (EWC)

EWC (Kirkpatrick et al., 2017) is directly inspired by the FEP. When learning task B after task A:

```
F_B = -E_q[ln p(o_B | s, theta)] + D_KL[q(theta) || p(theta | o_A)]
```

The posterior from task A serves as the prior for task B. Parameters that were important for task A (high Fisher information) resist change during task B learning.

```
L_EWC = L_B(theta) + lambda/2 * sum_i F_i * (theta_i - theta_A,i)^2
```

Where `F_i` is the Fisher information for parameter i (importance weight).

This is a direct application of the complexity-accuracy tradeoff: learning new things while not straying too far from what was previously learned.

## Key References

1. Friston, K., & Penny, W. (2011). Post hoc Bayesian model selection. *NeuroImage*, 56(4), 2089-2099.
2. Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., & Pezzulo, G. (2016). Active inference and learning. *Neuroscience & Biobehavioral Reviews*, 68, 862-879.
3. Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences*, 114(13), 3521-3526.
4. Bogacz, R. (2017). A tutorial on the free-energy framework for modelling perception and learning. *Journal of Mathematical Psychology*, 76, 198-211.
5. Sajid, N., Ball, P. J., Parr, T., & Friston, K. J. (2021). Active inference: demystified and compared. *Neural Computation*, 33(3), 674-712.
6. Da Costa, L., et al. (2020). Active inference on discrete state-spaces: A synthesis. *Journal of Mathematical Psychology*, 99, 102447.
7. Smith, R., Friston, K. J., & Whyte, C. J. (2022). A step-by-step tutorial on active inference and its application to empirical data. *Journal of Mathematical Psychology*, 107, 102632.
