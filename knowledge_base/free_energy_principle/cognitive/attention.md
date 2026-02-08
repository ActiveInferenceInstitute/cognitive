---
title: "Attention as Precision Optimization"
type: concept
status: active
created: 2025-01-01
updated: 2025-02-06
tags:
  - free_energy_principle
  - attention
  - precision
  - gain_modulation
  - salience
  - selective_attention
semantic_relations:
  - type: foundation
    links:
      - [[knowledge_base/free_energy_principle/mathematics/variational_free_energy|Variational Free Energy]]
      - [[knowledge_base/free_energy_principle/mathematics/core_principle|Core Principle]]
  - type: relates
    links:
      - [[perception|Perception]]
      - [[decision_making|Decision Making]]
      - [[consciousness|Consciousness]]
      - [[knowledge_base/free_energy_principle/biology/neural_systems|Neural Systems]]
  - type: implements
    links:
      - [[knowledge_base/free_energy_principle/mathematics/information_geometry|Information Geometry]]
---

# Attention as Precision Optimization

## Overview

Under the Free Energy Principle, attention is not a separate cognitive module but a fundamental aspect of inference: the optimization of **precision** (inverse variance) of prediction errors. Attention determines how much weight each source of information receives during belief updating. High precision at a location or feature means prediction errors from that source have greater influence on the brain's beliefs -- that source is "attended."

This account unifies diverse attentional phenomena -- spatial attention, feature-based attention, object-based attention, and divided attention -- under a single computational principle.

## Precision in Predictive Coding

### The Central Role of Precision

In predictive coding, belief updating is driven by precision-weighted prediction errors:

```
dmu/dt = sum_i Pi_i * epsilon_i
```

Where `Pi_i` is the precision and `epsilon_i` is the prediction error from source `i`. The precision determines the **gain** of each prediction error channel:

- **High Pi_i**: Source i strongly influences beliefs (attended)
- **Low Pi_i**: Source i weakly influences beliefs (unattended)
- **Pi_i = 0**: Source i has no influence (completely ignored)

### Precision as a Hidden Variable

Precision is not fixed -- it is itself inferred as a hidden variable through free energy minimization:

```
q(Pi) = argmin_{q(Pi)} F[q(s), q(Pi), o]
```

The optimal precision for source i is:

```
Pi_i* = 1 / E_q[(epsilon_i)^2] = 1 / (expected squared prediction error)
```

**High expected error** -> low precision (unreliable source -> downweight)
**Low expected error** -> high precision (reliable source -> upweight)

This is optimal Bayesian inference about reliability: the brain automatically assigns more weight to more reliable information sources.

### Free Energy Decomposition with Precision

Including precision as a variable, the free energy becomes:

```
F = 1/2 * sum_i [Pi_i * epsilon_i^2 - ln Pi_i] + complexity terms
```

The first term penalizes prediction errors (weighted by precision). The second term penalizes extreme precision values (acts as a regularizer -- prevents infinite precision).

Minimizing with respect to Pi_i:
```
partial F / partial Pi_i = 1/2 * [epsilon_i^2 - 1/Pi_i] = 0
-> Pi_i* = 1/epsilon_i^2
```

This means precision tracks the inverse of the running average of squared prediction errors -- exactly what a Bayesian observer should do.

## Types of Attention Under the FEP

### Spatial Attention

Attending to a spatial location increases the precision of prediction errors at that location:

```
Pi(x_attended) >> Pi(x_unattended)
```

Where `x` is spatial position. This means:
- Prediction errors at attended locations strongly drive inference
- Prediction errors at unattended locations are largely ignored
- Stimuli at attended locations are perceived more accurately and rapidly

**Posner cueing paradigm**: A valid cue pre-allocates precision to the cued location, so targets there benefit from higher gain on prediction errors. Invalid cues allocate precision to the wrong location, requiring a shift in precision allocation (cost of attentional reorienting).

### Feature-Based Attention

Attending to a feature (e.g., "red" or "vertical") increases precision for prediction errors involving that feature across the entire visual field:

```
Pi(feature = attended) >> Pi(feature = unattended)  for all spatial locations
```

This explains feature-based attentional enhancement: searching for a red target makes all red items more salient (higher precision on red prediction errors).

### Object-Based Attention

When attention is directed to an object, precision is increased for all features and locations associated with that object simultaneously:

```
Pi(o_j | object attended) >> Pi(o_j | object unattended)  for all features j
```

This is implemented through the generative model: the object representation `s_object` generates predictions for multiple features, and attending to the object increases precision for all those predictions.

### Temporal Attention (Foreperiod Effects)

Attending to a temporal interval increases precision of prediction errors around the expected time:

```
Pi(t = expected) >> Pi(t != expected)
```

This explains the hazard rate effect: reaction times decrease as the conditional probability of an event increases with time.

## Neural Implementation

### Gain Modulation

Precision weighting is implemented neurally as **gain modulation** -- the multiplicative scaling of neural responses:

```
Response = Gain * (input - prediction) = Pi * epsilon
```

Gain modulation is achieved through:
1. **Neuromodulatory systems**: Acetylcholine, norepinephrine, dopamine, serotonin
2. **Inhibitory interneurons**: GABAergic control of pyramidal cell gain
3. **Oscillatory synchronization**: Gamma-band synchronization increases effective gain
4. **NMDA receptor modulation**: Voltage-dependent gain on synaptic inputs

### Neuromodulatory Control of Precision

| Neuromodulator | Precision Domain | Effect of Increase |
|---------------|-----------------|-------------------|
| **Acetylcholine (ACh)** | Sensory precision | Trust sensory data more; data-driven processing |
| **Norepinephrine (NE)** | State transition precision | Trust predictions more; model-driven processing |
| **Dopamine (DA)** | Policy precision | Commit to action plans more strongly |
| **Serotonin (5-HT)** | Temporal precision | Weight distal outcomes more heavily |

**Evidence**:
- ACh increases sensory gain in V1 (Gil et al., 1997)
- ACh blockade (scopolamine) impairs attention tasks
- NE modulates signal-to-noise ratio in cortical networks
- DA modulates the vigor of action selection

### Cortical Oscillations and Attention

Different frequency bands implement precision at different scales:

| Frequency Band | Function | Precision Role |
|---------------|----------|----------------|
| **Gamma (30-100 Hz)** | Local processing | Bottom-up precision (sensory) |
| **Beta (13-30 Hz)** | Feedback / predictions | Top-down precision (prior) |
| **Alpha (8-12 Hz)** | Inhibition / gating | Precision reduction (suppression) |
| **Theta (4-8 Hz)** | Sequential processing | Temporal precision (working memory) |

**Alpha suppression** at attended locations: Alpha oscillations act as a "gating" mechanism. Decreased alpha at attended locations = increased precision = better processing. Increased alpha at unattended locations = decreased precision = suppression.

## Salience and Surprise

### Bayesian Surprise as Salience

Under the FEP, a stimulus is **salient** (attention-grabbing) when it generates high Bayesian surprise:

```
Salience(o) = D_KL[q(s|o) || q(s)] = Bayesian surprise
```

This is the amount of belief updating caused by the observation. Salient stimuli are those that most change the brain's beliefs.

**Properties**:
- Novel stimuli: High salience (prior doesn't predict them)
- Predicted stimuli: Low salience (prior already accounts for them)
- Repeated stimuli: Decreasing salience (habituation = belief convergence)

### Expected Salience and Attention Deployment

The expected salience of a location or feature guides where attention is deployed:

```
Expected_salience(x) = E_q(o|x)[D_KL[q(s|o,x) || q(s|x)]]
```

This is the expected information gain from sampling at location/feature x. Attention should be deployed where expected information gain is highest -- this is the **epistemic component** of the expected free energy for saccadic selection.

### Precision-Weighted Prediction Error as Neural Salience

In neural terms, salience is proportional to the precision-weighted prediction error:

```
Neural_salience ~ Pi * |epsilon|
```

Large prediction errors with high precision are highly salient (unexpected events in reliable channels).
Large prediction errors with low precision are not salient (noise in unreliable channels).
Small prediction errors with high precision are not salient (expected events in reliable channels).

## Attention Disorders Through the FEP Lens

### ADHD

Attention-Deficit/Hyperactivity Disorder may reflect **imprecise precision estimation**:

```
ADHD: q(Pi) has high variance -> unstable attention deployment
```

The brain cannot reliably estimate which prediction errors are important, leading to:
- Difficulty sustaining attention (precision fluctuates)
- Distractibility (irrelevant stimuli get inappropriately high precision)
- Hyperfocus (occasionally, precision locks onto one source excessively)

### Autism Spectrum

Autism may reflect **elevated sensory precision** with **reduced contextual modulation**:

```
ASD: Pi_sensory systematically too high; Pi_prior too low
```

This produces:
- Heightened sensory sensitivity (sensory prediction errors have too much influence)
- Reduced contextual effects (priors don't modulate perception enough)
- Detail-oriented processing (local precision dominates over global precision)
- Sensory overload (too many high-precision channels compete for processing)

### Anxiety

Anxiety involves **elevated precision on threat-related prediction errors**:

```
Anxiety: Pi_threat >> Pi_neutral
```

This biases inference toward threat interpretations and drives hypervigilance -- excessive attention to potential dangers.

## Attention and Active Inference

### Overt vs. Covert Attention

The FEP distinguishes two types of attention:

**Covert attention** (precision optimization):
```
Changing Pi without changing o -> mental attention shift
```
Adjusting the gain of prediction error channels without moving the body.

**Overt attention** (active sampling):
```
Changing o through action a -> physical attention shift
```
Moving the eyes, head, or body to sample new observations.

Both reduce free energy, but through different mechanisms:
- Covert attention changes how existing data is weighted
- Overt attention changes which data is available

### Saccadic Eye Movements

Eye movements are the paradigmatic case of overt attention as active inference:

```
saccade_target = argmin_target G(target)
              = argmin_target {-I_gain(target) - Pragmatic_value(target)}
```

The eyes move to locations that promise the most information gain (resolving uncertainty about the scene) and/or pragmatic value (viewing task-relevant content).

**Predictions** (well-supported empirically):
1. Fixations cluster at informative regions (edges, features, objects)
2. Fixation patterns depend on the task (pragmatic value varies)
3. Novel scenes attract more fixations than familiar scenes (more uncertainty)
4. Saccade latencies correlate with the expected information gain of the target

## Key References

1. Feldman, H., & Friston, K. (2010). Attention, uncertainty, and free-energy. *Frontiers in Human Neuroscience*, 4, 215.
2. Parr, T., & Friston, K. J. (2019). Attention or salience? *Current Opinion in Psychology*, 29, 1-7.
3. Kanai, R., Komura, Y., Shipp, S., & Friston, K. (2015). Cerebral hierarchies: predictive processing, precision and the pulvinar. *Philosophical Transactions of the Royal Society B*, 370(1668), 20140169.
4. Mirza, M. B., Adams, R. A., Mathys, C., & Friston, K. J. (2018). Human visual exploration reduces uncertainty about the sensed world. *PLoS One*, 13(1), e0190429.
5. Itti, L., & Baldi, P. (2009). Bayesian surprise attracts human attention. *Vision Research*, 49(10), 1295-1306.
6. Hohwy, J. (2012). Attention and conscious perception in the hypothesis testing brain. *Frontiers in Psychology*, 3, 96.
