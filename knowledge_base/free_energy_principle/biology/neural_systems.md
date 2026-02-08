---
title: "Neural Implementation of Free Energy Minimization"
type: concept
status: active
created: 2025-01-01
updated: 2025-02-06
tags:
  - free_energy_principle
  - neural_systems
  - cortical_hierarchy
  - canonical_microcircuit
  - neurotransmitters
  - predictive_coding
semantic_relations:
  - type: foundation
    links:
      - [[knowledge_base/free_energy_principle/mathematics/core_principle|Core Principle]]
      - [[knowledge_base/free_energy_principle/mathematics/variational_free_energy|Variational Free Energy]]
  - type: relates
    links:
      - [[homeostasis|Homeostasis]]
      - [[knowledge_base/free_energy_principle/cognitive/perception|Perception]]
      - [[knowledge_base/free_energy_principle/cognitive/attention|Attention]]
      - [[knowledge_base/free_energy_principle/cognitive/learning|Learning]]
---

# Neural Implementation of Free Energy Minimization

## Overview

The FEP makes strong claims about neural architecture: the brain is organized as a hierarchical generative model that minimizes variational free energy through predictive coding. This document examines how specific neural structures, circuits, and neurotransmitter systems implement the computations required by the FEP.

## The Canonical Microcircuit

### Architecture

Bastos et al. (2012) proposed a **canonical microcircuit** for predictive coding that maps FEP computations onto the six-layered structure of the neocortex:

```
Layer 1:       Receives top-down predictions (apical dendrites)
Layer 2/3:     Superficial pyramidal cells -> PREDICTION ERRORS (ascending)
Layer 4:       Stellate cells -> Input relay (receives feedforward input)
Layer 5:       Deep pyramidal cells -> PREDICTIONS (descending) + motor output
Layer 6:       Deep pyramidal cells -> PREDICTIONS to thalamus
Interneurons:  Precision weighting (gain modulation)
```

### Mapping to FEP Computations

| Computation | Neural Element | Direction |
|-------------|---------------|-----------|
| Predictions `g(mu)` | Deep pyramidal cells (L5/6) | Top-down (descending) |
| Prediction errors `epsilon` | Superficial pyramidal cells (L2/3) | Bottom-up (ascending) |
| State estimates `mu` | Deep pyramidal cells (L5) | Local representation |
| Precision weighting `Pi` | Inhibitory interneurons + neuromodulation | Lateral / diffuse |
| Input relay | Stellate cells (L4) | Feedforward |

### Evidence

1. **Laminar asymmetry**: Feedforward connections originate in superficial layers (prediction errors) and terminate in layer 4. Feedback connections originate in deep layers (predictions) and terminate in layers 1 and 5/6.

2. **Oscillatory signatures**: Feedforward (prediction error) signals are carried by gamma oscillations (30-100 Hz). Feedback (prediction) signals are carried by alpha/beta oscillations (8-30 Hz).

3. **Mismatch responses**: Mismatch negativity (MMN) and other oddball responses are generated in superficial layers, consistent with prediction error signaling.

4. **Repetition suppression**: Repeated stimuli reduce neural responses -- consistent with better predictions (smaller prediction errors).

## Cortical Hierarchies

### The Visual Hierarchy

The visual system is the best-characterized cortical hierarchy:

```
V1 -> V2 -> V4 -> IT -> PFC
  ^         ^       ^       ^
  |         |       |       |
Edges    Textures  Shapes  Objects  Context/Goals
(ms)     (10ms)   (50ms)  (100ms)  (seconds)
```

Each level operates at a different spatiotemporal scale:
- Lower levels: Fine spatial resolution, fast temporal dynamics
- Higher levels: Coarse spatial resolution, slow temporal dynamics

This is a natural consequence of hierarchical generative modeling: higher levels represent more abstract, slowly varying causes that generate the faster, more detailed patterns at lower levels.

### Empirical Priors and the Hierarchy

Higher levels provide **empirical priors** for lower levels:

```
p(s_1) <- prediction from level 2: f_2(mu_2)
p(s_2) <- prediction from level 3: f_3(mu_3)
...
```

These are not fixed priors but dynamic predictions that change with context. The top of the hierarchy represents the most abstract, context-dependent knowledge, which cascades down as increasingly specific predictions.

## Neurotransmitter Systems and Precision

### Acetylcholine (ACh)

**Source**: Nucleus basalis of Meynert, pedunculopontine nucleus
**Role in FEP**: Encodes sensory precision (reliability of bottom-up signals)

```
ACh increase -> Pi_sensory increases -> sensory prediction errors weighted more heavily
-> More data-driven processing
-> Enhanced sensory discrimination
```

**Evidence**:
- ACh enhances sensory-evoked responses in cortex
- ACh blockade (scopolamine) impairs perception and attention
- ACh release increases during novel or demanding sensory tasks

### Dopamine (DA)

**Source**: Ventral tegmental area (VTA), substantia nigra (SNc)
**Role in FEP**: Encodes precision over policies (confidence in action selection)

```
DA increase -> gamma increases -> policy selection more deterministic
-> More confident, goal-directed behavior
-> Increased motivation and approach
```

**Evidence**:
- DA signals encode reward prediction errors (Schultz, 1997)
- DA modulates the vigor and commitment of action
- DA depletion (Parkinson's) -> difficulty initiating actions (low policy precision)
- DA excess (mania, psychosis) -> overconfident, impulsive actions (high policy precision)

### Norepinephrine (NE)

**Source**: Locus coeruleus (LC)
**Role in FEP**: Encodes state transition precision (expected volatility)

```
Tonic NE mode: Low precision on transitions -> exploratory, flexible
Phasic NE mode: High precision on transitions -> focused, precise
```

**Evidence**:
- LC activity tracks unexpected uncertainty (volatility)
- NE modulates the balance between exploitation and exploration
- NE release enhances memory consolidation (precision of learned transitions)

### Serotonin (5-HT)

**Source**: Raphe nuclei
**Role in FEP**: Encodes temporal precision (discounting of future outcomes)

```
5-HT increase -> future outcomes weighted more heavily -> patient, long-term behavior
5-HT decrease -> future outcomes discounted -> impulsive, short-term behavior
```

**Evidence**:
- 5-HT depletion increases impulsivity and temporal discounting
- SSRIs (increasing 5-HT) are used to treat impulsive behavior
- 5-HT modulates the temporal horizon of active inference

### GABA and Glutamate

**Glutamate**: The primary excitatory neurotransmitter, carries prediction errors and predictions.
**GABA**: The primary inhibitory neurotransmitter, implements precision weighting.

```
Excitation (glutamate): Drives prediction error computation and transmission
Inhibition (GABA): Modulates the gain of prediction error units
E/I balance: Determines the effective precision of neural computations
```

Disrupted E/I balance is implicated in many neuropsychiatric conditions (autism, schizophrenia, epilepsy) and can be understood as disordered precision estimation.

## Subcortical Structures

### Thalamus

The thalamus is not merely a relay station but plays a critical role in predictive coding:

```
Thalamic relay nuclei: Gate sensory input (precision control)
Pulvinar: Coordinates cortical prediction error signaling
Thalamic reticular nucleus: Implements attentional selection (precision allocation)
```

The thalamus may implement the "precision matrix" of the predictive coding hierarchy, controlling which prediction errors are amplified and which are suppressed.

### Basal Ganglia

The basal ganglia implement **policy selection** through expected free energy evaluation:

```
Striatum: Encodes expected free energy of policies (input from cortex + DA)
GPi/SNr: Implements policy selection (winner-take-all through inhibition)
STN: Implements the "hold" signal (pause action selection when uncertain)
```

The dopaminergic modulation of the basal ganglia implements the precision parameter gamma over policies.

### Cerebellum

The cerebellum implements a complementary form of predictive coding for motor control:

```
Climbing fibers: Carry prediction errors (complex spikes)
Mossy fibers/granule cells: Carry predictions (context)
Purkinje cells: Compute prediction error correction signals
Deep cerebellar nuclei: Output corrected motor commands
```

Cerebellar prediction is primarily temporal (precise timing of motor sequences) rather than the causal inference of cortical predictive coding.

### Hippocampus

The hippocampus implements **spatial and episodic generative models**:

```
CA3: Pattern completion (recall/prediction from partial cues)
CA1: Comparison of predictions (from CA3) with input (from entorhinal cortex)
Dentate gyrus: Pattern separation (orthogonalization of similar inputs)
Entorhinal cortex: Grid cells as basis functions for spatial generative model
```

Hippocampal replay during sleep may implement Bayesian model reduction -- consolidating and pruning the episodic generative model. See [[knowledge_base/free_energy_principle/cognitive/learning]].

## Neural Oscillations and Message Passing

### Frequency-Specific Communication

Different frequency bands serve different roles in the predictive coding hierarchy:

| Band | Frequency | Direction | FEP Role |
|------|-----------|-----------|----------|
| Delta | 1-4 Hz | Top-down | Slowest predictions (context, goals) |
| Theta | 4-8 Hz | Bidirectional | Sequential processing, memory |
| Alpha | 8-12 Hz | Top-down | Precision suppression (inhibition) |
| Beta | 13-30 Hz | Top-down | Predictions (status quo maintenance) |
| Gamma | 30-100 Hz | Bottom-up | Prediction errors (fast, local) |

### Cross-Frequency Coupling

Different frequency bands interact through **cross-frequency coupling**:

```
Theta-gamma coupling: Gamma bursts (prediction errors) are nested within theta cycles
-> Sequences of prediction error updates organized by theta rhythm
-> Working memory: Each theta cycle carries a different "item"
```

This provides a mechanism for the temporal organization of hierarchical inference.

## Plasticity Rules from the FEP

### Short-Term Plasticity (Inference)

Fast synaptic dynamics implement perceptual inference:
- **Short-term facilitation**: Increases effective connectivity for repeated patterns (prediction confirmation)
- **Short-term depression**: Decreases connectivity for unexpected patterns (adaptation)

### Long-Term Plasticity (Learning)

Slower synaptic changes implement parameter learning:

```
dW/dt = eta * Pi * epsilon * mu^T  (precision-weighted Hebbian rule)
```

This maps onto known plasticity mechanisms:
- **LTP**: Hebbian strengthening (correlated pre/post activity with high precision)
- **LTD**: Anti-Hebbian weakening (uncorrelated activity or low precision)
- **Spike-timing dependent plasticity (STDP)**: Temporal version of prediction error learning
- **Metaplasticity**: Changes in the learning rule itself (precision of learning)

## Key References

1. Bastos, A. M., et al. (2012). Canonical microcircuits for predictive coding. *Neuron*, 76(4), 695-711.
2. Shipp, S. (2016). Neural elements for predictive coding. *Frontiers in Psychology*, 7, 1792.
3. Keller, G. B., & Mrsic-Flogel, T. D. (2018). Predictive processing: a canonical cortical computation. *Neuron*, 100(2), 424-435.
4. Parr, T., & Friston, K. J. (2018). The anatomy of inference: generative models and brain structure. *Frontiers in Computational Neuroscience*, 12, 90.
5. Friston, K. J. (2009). The free-energy principle: a rough guide to the brain? *Trends in Cognitive Sciences*, 13(7), 293-301.
6. Kanai, R., Komura, Y., Shipp, S., & Friston, K. (2015). Cerebral hierarchies: predictive processing, precision and the pulvinar. *Philosophical Transactions of the Royal Society B*, 370(1668), 20140169.
