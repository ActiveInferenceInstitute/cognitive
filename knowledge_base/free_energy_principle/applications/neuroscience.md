---
title: "Neuroscience Applications of the Free Energy Principle"
type: application
status: stable
created: 2025-02-06
updated: 2025-02-06
tags:
  - free_energy_principle
  - application
  - neuroscience
  - computational_neuroscience
  - predictive_coding
  - neural_circuits
  - brain_imaging
  - dynamic_causal_modeling
semantic_relations:
  - type: relates
    links:
      - [[knowledge_base/free_energy_principle/mathematics/core_principle|Core Principle]]
      - [[knowledge_base/free_energy_principle/mathematics/variational_free_energy|Variational Free Energy]]
      - [[knowledge_base/free_energy_principle/cognitive/perception|Perception]]
      - [[knowledge_base/free_energy_principle/cognitive/attention|Attention]]
      - [[knowledge_base/free_energy_principle/biology/neural_systems|Neural Systems]]
      - [[psychiatry|Psychiatric Applications]]
---

# Neuroscience Applications of the Free Energy Principle

## Overview

The Free Energy Principle (FEP) originated within computational neuroscience and remains most deeply developed there. It provides a unified account of brain function in which all neural processes -- perception, action, learning, attention -- serve a single imperative: the minimization of variational free energy. This document covers the major neuroscience applications, from microscale neural circuit models through mesoscale electrophysiological signatures to macroscale brain imaging methods.

The FEP's contribution to neuroscience is not merely theoretical. It has generated:
- Quantitative models of cortical microcircuitry (predictive coding)
- Novel analysis methods for neuroimaging data (Dynamic Causal Modeling)
- Specific empirical predictions about neural oscillations and prediction error signals
- Mechanistic accounts of neuromodulation in terms of precision weighting
- A principled framework connecting neural computation to behavior and psychopathology

## Theoretical Framework

### Predictive Coding in Cortical Circuits

The central process theory derived from the FEP for neural computation is **predictive coding**. The cortex is organized as a hierarchical generative model where:

1. **Top-down connections** carry predictions from higher to lower cortical areas
2. **Bottom-up connections** carry prediction errors from lower to higher areas
3. **Lateral connections** encode precision (the confidence in prediction errors)

```
Formal architecture:

Level i+1 (higher cortex):
  mu_{i+1} = current beliefs about slow/abstract causes
  prediction: g_{i+1}(mu_{i+1}) -> expected activity at level i

Level i (lower cortex):
  mu_i = current beliefs about fast/concrete causes
  prediction error: epsilon_i = mu_i - g_{i+1}(mu_{i+1})
  precision-weighted error: Pi_i * epsilon_i

Update rule:
  d(mu_i)/dt = -partial F / partial mu_i
             = D(mu_i) - Pi_{i-1} * epsilon_{i-1}
               + (partial g_i / partial mu_i)^T * Pi_i * epsilon_i
```

This maps onto known cortical anatomy:

| Computational Role | Cortical Implementation | Laminar Location |
|-------------------|------------------------|------------------|
| Predictions | Deep pyramidal neurons | Layers 5/6 |
| Prediction errors | Superficial pyramidal neurons | Layers 2/3 |
| Precision weighting | Gain modulation (neuromodulators) | All layers |
| State estimates | Population firing rates | Layers 2/3, 5/6 |
| Model parameters | Synaptic connection weights | All layers |
| Forward (error) connections | Feedforward projections | Layer 2/3 -> Layer 4 |
| Backward (prediction) connections | Feedback projections | Layer 5/6 -> Layers 1, 5/6 |

### The Canonical Microcircuit

Bastos et al. (2012) proposed that the cortical column implements a **canonical microcircuit** for predictive coding. This circuit includes:

```
Superficial pyramidal cells (SP):
  - Encode prediction errors
  - Project forward to the next cortical level
  - Receive top-down predictions on apical dendrites
  - Receive bottom-up input on basal dendrites

Deep pyramidal cells (DP):
  - Encode predictions (expectations)
  - Project backward to the previous cortical level
  - Receive prediction errors from superficial cells
  - Send predictions to superficial cells via interneurons

Inhibitory interneurons (II):
  - Mediate precision weighting
  - Parvalbumin (PV) interneurons: gain control on prediction errors
  - Somatostatin (SST) interneurons: contextual modulation
  - VIP interneurons: disinhibition (precision enhancement)
```

This canonical microcircuit is repeated across cortical areas, with area-specific generative models encoded in the connectivity patterns and synaptic weights.

## Dopamine as Precision

### The Precision Hypothesis

One of the most influential neuroscience applications of the FEP is the reconceptualization of **dopamine** as encoding the precision of beliefs about policies (action plans):

```
Classical view: Dopamine = reward prediction error
FEP view:      Dopamine = precision over expected free energy of policies

pi(a) = sigma(-G(a) * gamma)

Where:
  pi(a) = probability of selecting policy a
  G(a) = expected free energy of policy a
  gamma = precision parameter (encoded by dopamine)
  sigma = softmax function
```

When dopamine (gamma) is high:
- Policies are selected more deterministically (high confidence in action plan)
- Behavior is more goal-directed and exploitative
- The agent "commits" to the best available policy

When dopamine (gamma) is low:
- Policy selection is more stochastic (low confidence)
- Behavior is more exploratory and random
- The agent "dithers" between competing policies

### Empirical Evidence

This reframing accounts for diverse dopaminergic phenomena:

**Reward prediction errors**: Under the FEP, phasic dopamine signals reflect updates to precision over policies, not reward per se. When an unexpected reward occurs, the precision of the policy that led to that reward increases -- this looks identical to a reward prediction error but has a different computational meaning.

**Motivation and vigor**: Tonic dopamine levels set the baseline precision for policy selection. Low tonic dopamine (as in Parkinson's disease) produces:
- Bradykinesia (slow movements): Low precision -> weak commitment to motor policies
- Apathy: Low precision -> inability to select among competing goals
- Akinesia (difficulty initiating movement): Insufficient precision to "win" over default inaction

**Addiction**: Drugs of abuse artificially inflate precision over drug-seeking policies:
```
gamma_drug >> gamma_natural
-> pi(drug_seeking) -> 1 regardless of G(drug_seeking)
-> Compulsive behavior despite negative consequences
```

See [[psychiatry]] for the full psychiatric account.

## Dynamic Causal Modeling (DCM)

### Overview

Dynamic Causal Modeling is a **Bayesian framework for inferring effective connectivity** from neuroimaging data. It is the primary neuroimaging analysis method derived from the FEP.

```
Core idea:
  Data = y (measured brain activity: fMRI BOLD, EEG, MEG, LFP)
  Model = m (neural mass model with specific connectivity)
  Parameters = theta (connection strengths, time constants)

  DCM inverts a generative model:
  p(theta | y, m) proportional_to p(y | theta, m) * p(theta | m)

  Model comparison via free energy:
  F(m) = ln p(y | m) - D_KL[q(theta) || p(theta | y, m)]

  Best model: m* = argmax_m F(m) approx argmax_m ln p(y | m)
```

### DCM for fMRI

DCM for fMRI models the hemodynamic response as a nonlinear observation function applied to underlying neural dynamics:

```
Neural state equation:
  dz/dt = A * z + sum_j u_j * B_j * z + C * u

Where:
  z = neural states (activity in each region)
  A = intrinsic (endogenous) connectivity matrix
  B_j = modulatory effects of input j on connectivity
  C = driving inputs
  u = experimental inputs (stimuli, task conditions)

Hemodynamic observation model:
  y = lambda(z) + noise

  Where lambda maps neural activity through:
  1. Neurovascular coupling
  2. Balloon model (Buxton et al.)
  3. BOLD signal equation
```

DCM has been applied to study:
- Visual processing pathways (feedforward vs. feedback connectivity)
- Language networks (left lateralization, compensatory reorganization)
- Motor planning and execution circuits
- Default mode network dynamics
- Clinical populations (schizophrenia, autism, depression)

### DCM for Electrophysiology (EEG/MEG)

DCM for electrophysiological data uses neural mass models that generate oscillatory dynamics:

```
Neural mass model (Jansen-Rit or conductance-based):
  Excitatory population: pyramidal cells
  Inhibitory populations: fast (GABA-A) and slow (GABA-B)

  Each population characterized by:
  - Mean membrane potential
  - Mean firing rate (sigmoid function of potential)
  - Synaptic kernel (impulse response)

  Coupled via intrinsic and extrinsic connections
```

This version of DCM can model:
- Event-related potentials (ERPs) as transient prediction errors
- Induced oscillatory responses as precision dynamics
- Steady-state responses as evidence for specific generative models
- Cross-frequency coupling as hierarchical message passing

## EEG/fMRI Signatures of Prediction Error

### Mismatch Negativity (MMN)

The mismatch negativity is an EEG component (peaking ~100-250ms, frontocentral topography) elicited when an auditory stimulus violates a learned regularity. Under the FEP:

```
Standard stimuli: Brain learns p(tone | context) = high probability
Deviant stimulus: Elicits prediction error epsilon = o - g(mu)
MMN amplitude: Proportional to precision-weighted prediction error Pi * epsilon

Key predictions (confirmed empirically):
1. MMN increases with deviance magnitude (larger epsilon)
2. MMN increases with standard probability (stronger prior -> larger error)
3. MMN is reduced by attention withdrawal (lower precision Pi)
4. MMN generators are in auditory cortex + frontal cortex (hierarchical model)
5. MMN adapts with repeated deviants (updating the generative model)
```

Garrido et al. (2009) used DCM to show that MMN involves changes in **backward connectivity** (top-down predictions), not just forward connectivity, supporting the predictive coding account over simpler adaptation models.

### The P300

The P300 (or P3b) is a later EEG component (~300-600ms, parietal maximum) elicited by task-relevant surprising events:

```
FEP interpretation:
  P300 = higher-order prediction error that triggers model updating

  P3a (frontal): Orienting response = epistemic action
    -> Reflects salience of unexpected event
    -> Drives attention (precision reallocation)

  P3b (parietal): Context updating = belief revision
    -> Reflects magnitude of belief update
    -> Scales with information content: I = -ln p(event)
```

The P300's amplitude scales with surprise (Shannon information), consistent with it reflecting the magnitude of belief updating in a hierarchical generative model.

### Repetition Suppression and Expectation Suppression

fMRI studies of visual processing reveal two distinct phenomena:

```
Repetition suppression:
  Repeated stimulus -> reduced BOLD response
  FEP account: Prediction errors decrease as the stimulus is predicted

Expectation suppression:
  Expected stimulus -> reduced BOLD response compared to unexpected
  FEP account: Top-down predictions partially cancel bottom-up input

Critical dissociation (Summerfield et al., 2008):
  Repetition effects: Strongest in lower visual areas (V1, V2)
  Expectation effects: Strongest in higher visual areas (LOC, FFA)
  -> Consistent with hierarchical predictive coding
```

## Neural Oscillations and the FEP

### Oscillatory Signatures of Predictive Coding

The FEP predicts specific roles for neural oscillations in cortical message passing:

```
Gamma oscillations (30-100 Hz):
  Role: Encode prediction errors (bottom-up)
  Mechanism: Superficial pyramidal cell activity
  Evidence: Gamma increases with prediction error magnitude

Alpha/beta oscillations (8-30 Hz):
  Role: Encode predictions (top-down)
  Mechanism: Deep pyramidal cell activity
  Evidence: Alpha/beta increases with predictability

Theta oscillations (4-8 Hz):
  Role: Temporal scaffolding for hierarchical inference
  Mechanism: Phase-amplitude coupling with gamma
  Evidence: Theta modulates gamma in hippocampal-cortical circuits
```

### Asymmetric Directed Coupling

Bastos et al. (2015) demonstrated in macaque visual cortex:

```
Feedforward direction (V1 -> V4 -> FEF):
  - Dominated by gamma-band Granger causality
  - Increases with stimulus drive (bottom-up prediction errors)

Feedback direction (FEF -> V4 -> V1):
  - Dominated by alpha/beta-band Granger causality
  - Increases with predictability (top-down predictions)
```

This frequency-specific asymmetry is a strong prediction of the canonical microcircuit model and has been replicated across species and modalities.

### Cross-Frequency Coupling

Hierarchical predictive coding predicts cross-frequency coupling as the mechanism by which different levels of the hierarchy communicate:

```
Phase-amplitude coupling (PAC):
  Low-frequency phase (theta/alpha) from higher levels
  modulates
  High-frequency amplitude (gamma) at lower levels

Interpretation:
  Higher-level predictions (slow, alpha/beta) modulate the gain
  of lower-level prediction errors (fast, gamma)
  = Precision weighting across hierarchical levels
```

## Hierarchical Cortical Processing

### Visual Hierarchy

The visual processing stream provides the clearest example of hierarchical predictive coding:

```
V1: Edge orientation, spatial frequency
  |  predictions (alpha/beta)        ^ errors (gamma)
V2: Texture boundaries, contour ownership
  |  predictions (alpha/beta)        ^ errors (gamma)
V4: Shape, color, intermediate features
  |  predictions (alpha/beta)        ^ errors (gamma)
IT: Object identity, category
  |  predictions (alpha/beta)        ^ errors (gamma)
PFC: Context, goals, task rules

Receptive field sizes increase at each level
Temporal dynamics slow at each level
Abstraction increases at each level
```

Empirical evidence for this architecture:
- **Feedback modulates feedforward**: V1 responses to identical stimuli change depending on context provided by higher areas
- **Predictable stimuli produce less activation**: Expected visual events produce lower BOLD responses in visual cortex
- **Unpredictable stimuli produce more activation**: Prediction error signals scale with surprise
- **Omission responses**: Visual cortex responds when an expected stimulus is ABSENT -- a signature of prediction, not stimulus processing

### Auditory Hierarchy

The auditory system implements a similar predictive hierarchy:

```
Primary auditory cortex (A1): Spectrotemporal features
  |
Belt areas: Auditory objects, pitch
  |
Parabelt: Sound categories, sequences
  |
Superior temporal sulcus (STS): Speech, complex sounds
  |
Prefrontal cortex: Linguistic structure, meaning

Temporal prediction windows increase up the hierarchy:
  A1: ~25-50ms (single phoneme features)
  Belt: ~100-200ms (syllables, pitch contours)
  STS: ~500ms-2s (words, phrases)
  PFC: seconds to minutes (narrative, discourse)
```

### Somatosensory and Motor Hierarchies

Motor control under the FEP is active inference -- the motor system generates proprioceptive predictions that are fulfilled by spinal reflexes:

```
Motor cortex: Generates predictions of proprioceptive consequences
  |
Spinal cord: Computes prediction errors (desired vs. actual position)
  |
Alpha motor neurons: Reflex arcs minimize prediction errors by moving the limb

Movement = making proprioceptive predictions come true
```

This resolves the classical motor control problem of "how does the brain specify which muscles to activate" -- it does not. It specifies the desired sensory consequences, and peripheral reflexes figure out the muscle activations.

## Current Research

### Computational Phenotyping

Using computational models derived from the FEP to characterize individual differences in neural processing:

```
Approach:
1. Fit predictive coding model to individual's neural data
2. Extract computational parameters (precision weights, learning rates)
3. Use parameters as "computational phenotype"
4. Correlate with behavior, genetics, clinical status

Applications:
- Identifying biomarkers for psychiatric disorders
- Predicting treatment response
- Tracking disease progression
- Personalized medicine
```

### Deep Temporal Models

Extending hierarchical predictive coding to account for temporal structure at multiple timescales:

```
Classical predictive coding: Spatial hierarchy (what)
Deep temporal models: Spatiotemporal hierarchy (what + when)

Each level represents temporal regularities at different scales:
  Level 1: Millisecond dynamics (sensory features)
  Level 2: Hundred-millisecond dynamics (events)
  Level 3: Second-scale dynamics (episodes)
  Level 4: Minute-to-hour dynamics (contexts)
```

### Neuromorphic Implementation

Building hardware that implements predictive coding principles:

```
Neuromorphic chips (Intel Loihi, IBM TrueNorth):
  - Spiking neural networks that naturally implement prediction error minimization
  - Event-driven computation (only process prediction errors)
  - Low power consumption (only surprising events trigger computation)
  - Online learning through spike-timing-dependent plasticity
```

## Open Questions

1. **Granularity of the generative model**: At what level of detail does the cortex implement predictive coding? Single neurons? Cortical columns? Areas? The answer likely varies across brain regions.

2. **Precision estimation**: How exactly does the brain estimate precision? The computational mechanisms for second-order statistics remain unclear.

3. **Generalized coordinates**: The FEP formalism uses generalized coordinates (position, velocity, acceleration...). Whether the brain explicitly represents temporal derivatives is debated.

4. **Non-cortical structures**: How do subcortical structures (basal ganglia, cerebellum, thalamus) fit into the predictive coding framework? The thalamus may play a key role in routing prediction errors, while the cerebellum may encode forward models.

5. **Consciousness**: Does prediction error minimization relate to consciousness? Some proposals link consciousness to the precision of high-level predictions (see [[knowledge_base/free_energy_principle/cognitive/consciousness]]).

6. **Individual differences**: What determines the parameters of an individual's generative model? Genetics, development, experience all contribute, but the mapping is poorly understood.

## References

1. Friston, K. (2005). A theory of cortical responses. *Philosophical Transactions of the Royal Society B*, 360(1456), 815-836.
2. Bastos, A. M., Usrey, W. M., Adams, R. A., Mangun, G. R., Fries, P., & Friston, K. J. (2012). Canonical microcircuits for predictive coding. *Neuron*, 76(4), 695-711.
3. Adams, R. A., Shipp, S., & Friston, K. J. (2013). Predictions not commands: active inference in the motor system. *Brain Structure and Function*, 218(3), 611-643.
4. Shipp, S. (2016). Neural elements for predictive coding. *Frontiers in Psychology*, 7, 1792.
5. Bastos, A. M., Vezoli, J., Bosman, C. A., Schoffelen, J. M., Oostenveld, R., Dowdall, J. R., ... & Fries, P. (2015). Visual areas exert feedforward and feedback influences through distinct frequency channels. *Neuron*, 85(2), 390-401.
6. Garrido, M. I., Kilner, J. M., Stephan, K. E., & Friston, K. J. (2009). The mismatch negativity: a review of underlying mechanisms. *Clinical Neurophysiology*, 120(3), 453-463.
7. Friston, K. J., Harrison, L., & Penny, W. (2003). Dynamic causal modelling. *NeuroImage*, 19(4), 1273-1302.
8. Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., & Pezzulo, G. (2017). Active inference: a process theory. *Neural Computation*, 29(1), 1-49.
9. Summerfield, C., Trittschuh, E. H., Monti, J. M., Mesulam, M. M., & Egner, T. (2008). Neural repetition suppression reflects fulfilled perceptual expectations. *Nature Neuroscience*, 11(9), 1004-1006.
10. Heilbron, M., & Chait, M. (2018). Great expectations: Is there evidence for predictive coding in auditory cortex? *Neuroscience*, 389, 54-73.

## See Also

- [[knowledge_base/free_energy_principle/mathematics/core_principle|Core Mathematical Formulation]]
- [[knowledge_base/free_energy_principle/cognitive/perception|Perception as Free Energy Minimization]]
- [[knowledge_base/free_energy_principle/cognitive/attention|Attention and Precision]]
- [[knowledge_base/free_energy_principle/biology/neural_systems|Neural Systems]]
- [[psychiatry|Psychiatric Applications]]
- [[knowledge_base/free_energy_principle/implementations/robotics|Robotics Implementations]]

## See also

- [[knowledge_base/biology/neuroscience|Neuroscience]]
