---
title: "Psychiatric Applications of the Free Energy Principle"
type: application
status: stable
created: 2025-02-06
updated: 2025-02-06
tags:
  - free_energy_principle
  - application
  - psychiatry
  - computational_psychiatry
  - aberrant_precision
  - mental_health
  - psychopathology
semantic_relations:
  - type: relates
    links:
      - [[knowledge_base/free_energy_principle/mathematics/core_principle|Core Principle]]
      - [[knowledge_base/free_energy_principle/mathematics/variational_free_energy|Variational Free Energy]]
      - [[knowledge_base/free_energy_principle/cognitive/perception|Perception]]
      - [[knowledge_base/free_energy_principle/biology/homeostasis|Homeostasis]]
      - [[neuroscience|Neuroscience Applications]]
      - [[knowledge_base/free_energy_principle/cognitive/attention|Attention]]
---

# Psychiatric Applications of the Free Energy Principle

## Overview

Computational psychiatry under the Free Energy Principle reframes mental health disorders as disturbances in the brain's inferential machinery. Rather than viewing psychiatric conditions as discrete disease entities with specific neurochemical causes, the FEP offers a **unified account** in which most major psychiatric disorders arise from **aberrant precision weighting** -- systematic errors in how the brain assigns confidence to its predictions and sensory evidence.

This framework has several advantages over traditional psychiatric nosology:
- It provides a **mechanistic** account of symptoms (not just description)
- It explains **comorbidity** (many disorders share precision-weighting disruptions)
- It generates **quantitative predictions** testable with computational modeling
- It connects **phenomenology** (subjective experience) to **neurobiology** (synaptic mechanisms)
- It suggests **novel therapeutic interventions** targeting specific computational parameters

## Theoretical Framework: Aberrant Precision Weighting

### The Core Idea

In healthy inference, precision is optimally allocated:

```
Healthy inference:
  Pi_sensory   = appropriately weighted sensory precision
  Pi_prior     = appropriately weighted prior precision
  Pi_policy    = appropriately weighted policy precision

  Posterior = optimal balance of data and expectations
```

Psychiatric disorders arise when precision is systematically misallocated:

```
Aberrant inference:
  Pi_sensory   too high or too low
  Pi_prior     too high or too low
  Pi_policy    too high or too low

  Posterior = biased, leading to maladaptive perception, belief, or action
```

### The Precision Landscape

Different disorders map onto different regions of the precision parameter space:

```
                    High sensory precision
                           |
                    Autism  |  PTSD
                           |
Low prior precision -------+------- High prior precision
                           |
              Schizophrenia|  OCD
                           |
                    Low sensory precision
```

This is a simplification -- each disorder involves multiple precision parameters -- but it captures the core intuition.

### Neuromodulatory Basis

Since precision is encoded by neuromodulatory systems, the precision framework connects directly to neurochemistry:

| Neuromodulator | Precision Domain | Disorder Association (Excess) | Disorder Association (Deficit) |
|---------------|-----------------|-------------------------------|-------------------------------|
| Dopamine | Policy precision | Mania, psychosis, addiction | Depression, apathy, Parkinsonism |
| Serotonin | Temporal precision | Compulsivity, rigidity | Impulsivity, aggression |
| Norepinephrine | Volatility estimation | Hypervigilance, anxiety | Inattention, confusion |
| Acetylcholine | Sensory precision | Sensory hypersensitivity | Perceptual deficits, delirium |

## Schizophrenia: False Inference

### The Aberrant Precision Account

Schizophrenia is characterized by a failure to appropriately attenuate sensory precision, leading to **aberrant salience** -- irrelevant sensory signals are treated as if they are highly informative:

```
Healthy:
  epsilon = o - g(mu)            [prediction error]
  Update = Pi_sensory * epsilon  [precision-weighted update]
  Pi_sensory is appropriate      [irrelevant errors attenuated]

Schizophrenia:
  Pi_sensory is inflated for irrelevant signals
  -> Random noise treated as meaningful signal
  -> Aberrant prediction errors drive false beliefs
  -> World feels "changed," significant, threatening
```

### Symptom Mapping

**Positive symptoms** (hallucinations, delusions):

```
Hallucinations:
  Abnormally high precision on internally generated predictions
  + Low precision on sensory evidence
  -> Internal predictions override sensory reality
  -> "Hearing voices" = unconstrained predictions treated as veridical

Delusions:
  Aberrant salience of irrelevant correlations
  + Impaired model updating (prior precision too low to anchor beliefs)
  -> False causal models built from noisy prediction errors
  -> Delusions of reference: "Everything is connected to me"
```

**Negative symptoms** (anhedonia, avolition, flat affect):

```
Reduced precision over policies (low dopaminergic gamma):
  pi(a) = sigma(-G(a) * gamma), gamma -> 0
  -> All policies equally likely
  -> No motivation to select any particular action
  -> Avolition, social withdrawal, poverty of speech
```

**Disorganized symptoms** (thought disorder, bizarre behavior):

```
Unstable precision dynamics:
  Pi fluctuates rapidly and unpredictably
  -> Incoherent shifts between belief states
  -> Thought disorder: associations driven by noise
  -> Behavioral disorganization: actions not guided by stable goals
```

### Pharmacological Evidence

Antipsychotics primarily block dopamine D2 receptors. Under the FEP:

```
D2 blockade -> reduced precision over aberrant policies
-> Positive symptoms improve (aberrant salience reduced)
-> But negative symptoms may worsen (further reduction in policy precision)
```

This explains why antipsychotics help positive but not negative symptoms, and why they can produce akinesia and apathy (Parkinsonian side effects = further precision reduction).

### The Glutamate Connection

NMDA receptor hypofunction (a leading theory of schizophrenia) maps onto the FEP:

```
NMDA receptors: Mediate synaptic plasticity and gain control
  -> NMDA hypofunction reduces precision modulation
  -> Prediction errors cannot be properly gain-controlled
  -> Both sensory and prior precision become noisy and unreliable
  -> Accounts for both positive and negative symptoms
```

Ketamine (NMDA antagonist) produces transient psychotic-like symptoms in healthy individuals, consistent with this account.

## Autism: Overly Precise Sensory Processing

### The High Precision Account

Autism spectrum conditions involve **excessively high sensory precision** relative to prior precision:

```
Neurotypical:
  Pi_sensory / Pi_prior = balanced ratio
  -> Priors smooth over noise, enable generalization, support social inference

Autism:
  Pi_sensory >> Pi_prior
  -> Sensory detail overwhelms top-down prediction
  -> Each experience treated as unique rather than categorized
  -> Reduced contextual modulation of perception
```

### Symptom Mapping

**Sensory hypersensitivity**:
```
High Pi_sensory -> prediction errors from sensory noise are amplified
-> Small deviations from expected stimuli are highly salient
-> Sensory overload in noisy environments
-> Preference for predictable, controlled environments
```

**Insistence on sameness**:
```
Weak priors + high sensory precision
-> Small environmental changes generate large prediction errors
-> To minimize free energy, seek to control the environment
-> Routines and rituals = engineering a predictable sensory stream
```

**Social cognition difficulties**:
```
Social inference requires strong priors (social "intuition"):
  p(intention | facial_expression, context, social_norms)

With weak priors, social inference becomes computationally overwhelming:
  -> Must analytically process each social cue individually
  -> Cannot "fill in" from context (no strong top-down prediction)
  -> Difficulty reading facial expressions, body language, tone
```

**Enhanced perceptual abilities**:
```
High sensory precision -> superior performance on detail-oriented tasks:
  - Perfect pitch (high precision for auditory features)
  - Superior visual search (each item fully processed)
  - Enhanced pattern detection in simple domains
  - Savant abilities (extreme precision in specific modalities)
```

### The HIPPEA Model

Van de Cruys et al. (2014) proposed the **High, Inflexible Precision of Prediction Errors in Autism (HIPPEA)** model:

```
Key claim: Autism involves inflexible precision, not just high precision
  -> Precision cannot be dynamically adjusted to context
  -> Cannot "turn down" sensory precision when appropriate
  -> Cannot "turn up" prior precision for social inference
  -> Leads to context-insensitive, detail-focused processing style
```

## Depression: Learned Helplessness Under the FEP

### The Allostatic Account

Depression can be understood as a state of chronically elevated free energy combined with reduced confidence in one's ability to resolve it:

```
Depression as computational state:
  1. Persistently negative prior preferences: C = ln p(o) is low
     -> Agent expects bad outcomes
  2. Low precision over policies: gamma -> 0
     -> Agent believes no action will help
  3. High complexity cost: D_KL[q || p] is large
     -> Models of the world are effortful to maintain

  Result: G(a) approx constant for all a
  -> No action is worth taking
  -> Withdrawal, anhedonia, psychomotor retardation
```

### Stephan's Allostatic Self-Efficacy Model

Stephan et al. (2016) proposed that depression arises from a metacognitive belief about one's own regulatory capacity:

```
Allostatic self-efficacy = belief that one can successfully
  reduce interoceptive prediction errors through action

Depression = low allostatic self-efficacy:
  p(Delta_F < 0 | action) is low
  -> "Nothing I do will make me feel better"
  -> Withdrawal is optimal under this (false) belief
  -> Confirmation bias: withdrawal prevents disconfirming evidence
```

### Rumination as Failed Inference

```
Rumination under the FEP:
  Repeated mental simulation of negative scenarios
  = Attempting to reduce uncertainty through imagination
  = But without real-world action, no new evidence arrives

  F_rumination = D_KL[q_negative || p_true] + cost_of_simulation
  -> Free energy INCREASES with rumination (divergence grows)
  -> But subjective experience is of "working on the problem"
  -> Agent persists because expected free energy of NOT ruminating
     seems even higher (uncertainty about negative outcomes)
```

### Precision Dynamics in Depression

```
Interoceptive precision: Often elevated
  -> Heightened awareness of negative bodily states
  -> Somatic symptoms (fatigue, pain, appetite changes)

Exteroceptive precision: Often reduced
  -> Reduced engagement with external world
  -> "Nothing seems interesting or vivid"

Policy precision: Reduced
  -> Difficulty deciding, initiating action
  -> Psychomotor retardation
```

## Anxiety: Precision Imbalance

### The Interoceptive Prediction Error Account

Anxiety disorders involve **excessively high precision on interoceptive prediction errors** combined with **inflated expectations of threat**:

```
Anxiety:
  Pi_intero is abnormally high
  -> Normal bodily sensations (heartbeat, breathing) become salient
  -> Prediction errors from bodily fluctuations are amplified
  -> Interpreted as evidence of danger (panic: "heart attack")

  Pi_threat is abnormally high
  -> World modeled as more dangerous than it is
  -> Constant vigilance, scanning for threat
  -> Ambiguous stimuli interpreted as threatening
```

### Panic Disorder

```
Panic cycle as precision runaway:
  1. Random interoceptive fluctuation (slightly elevated heart rate)
  2. High interoceptive precision -> large prediction error
  3. Prediction error interpreted as threat (catastrophic inference)
  4. Sympathetic activation (actual heart rate increase)
  5. Even larger interoceptive prediction error
  6. Positive feedback loop -> panic attack
  7. Panic attack confirms the threatening generative model
```

### Generalized Anxiety Disorder (GAD)

```
GAD as chronic uncertainty intolerance:
  Pi_prior for safety is low (world feels unpredictable)
  Pi_sensory for threat is high (constantly scanning for danger)
  Expected free energy is chronically elevated
  -> Worry = mental simulation of threatening scenarios
     = attempting to reduce uncertainty through imagination
     = but without action, this increases free energy (rumination)
```

## Addiction: Aberrant Precision Over Drug-Related Policies

### The Computational Account

```
Healthy preference learning:
  C_natural = prior preferences for natural rewards
  gamma_natural = precision over policies for natural rewards

Addiction:
  C_drug is massively inflated by pharmacological action
  gamma_drug >> gamma_natural
  -> Drug-seeking policies dominate all others
  -> Expected free energy of drug-seeking appears overwhelmingly favorable
  -> Even when conscious beliefs oppose drug use
```

### Tolerance and Withdrawal

```
Tolerance:
  Generative model adapts to predict drug effects
  -> Drug state becomes the expected state
  -> Drug produces smaller prediction error (less "high")
  -> Need more drug to generate prediction error (dose escalation)

Withdrawal:
  Without drug, massive interoceptive prediction errors
  -> Expected drug state minus actual sober state = large epsilon
  -> High free energy drives drug-seeking to restore predicted state
  -> Homeostatic regulation has been "hijacked"
```

### Habit Formation and Relapse

```
Habit formation in addiction:
  Repeated drug-seeking -> amortized policy (habitual)
  E(pi_drug) >> E(pi_other) -> automatic drug-seeking
  Relapse: Habit prior overrides deliberative evaluation
  -> Patient "knows" drug is harmful but acts anyway
  -> Policy precision (habit) exceeds reflective precision
```

## PTSD: Precision Crystallization

### The Frozen Prior Account

PTSD involves a traumatic generative model with **abnormally high precision** that resists updating:

```
Traumatic encoding:
  Extreme threat -> maximal precision during encoding
  -> Generative model fragment with very high Pi_trauma
  -> This model fragment is resistant to belief updating
  -> Cannot be "overwritten" by subsequent safe experiences

Flashbacks:
  Contextual cue activates traumatic generative model
  -> High-precision predictions of threat
  -> Autonomic arousal (sympathetic activation)
  -> Perceptual re-experiencing (predictions override current sensory input)
  -> Despite current safety (sensory evidence of safety has low precision)
```

### Avoidance as Active Inference

```
Avoidance = active inference to prevent activation of traumatic model:
  Expected free energy of approaching trauma-related contexts is high
  -> Avoid contexts that might trigger traumatic predictions
  -> This prevents model updating (no disconfirming evidence)
  -> Maintains the high-precision traumatic model
  -> Self-perpetuating cycle
```

### Exposure Therapy Under the FEP

```
Exposure therapy = controlled generation of prediction errors:
  1. Present trauma-related cues in safe context
  2. Traumatic model generates high-precision threat predictions
  3. Safe context generates prediction errors: "No danger occurred"
  4. Repeatedly: safety prediction errors accumulate
  5. Eventually: Pi_safety > Pi_trauma for this context
  6. Traumatic model updated: this context is safe
  7. Generalize to other contexts (transfer of updated model)

Key: Exposure must be long enough for safety prediction errors
to overcome the high precision of the traumatic prior.
```

## OCD: Precision Over Interoceptive Error and Uncertainty

### The Uncertainty Intolerance Account

```
OCD cycle:
  1. Intrusive thought (normal; occurs in everyone)
  2. Abnormally high precision on this thought -> treated as significant
  3. Uncertainty about whether the feared outcome will occur
  4. High precision on need-for-certainty prior
  5. Compulsive behavior = active inference to reduce uncertainty
  6. Temporary uncertainty reduction (free energy drops)
  7. But the model that "uncertainty is dangerous" is reinforced
  8. Return to step 1 with even higher precision
```

### Checking and Washing as Free Energy Minimization

```
Checking compulsions:
  "Did I lock the door?" -> Uncertainty about hidden state
  Checking = active inference to reduce uncertainty
  Memory for checking is impaired (low precision on checking memory)
  -> Must check again -> cycle perpetuates

Washing compulsions:
  Contamination = interoceptive prediction error (feeling "dirty")
  Washing = active inference to resolve interoceptive error
  But the generative model predicts contamination with high precision
  -> Cleaning is never "enough" to overcome the prior
```

## Computational Phenotyping

### The Approach

Computational phenotyping uses FEP-derived models to characterize individual patients:

```
Traditional diagnosis:
  Symptoms -> Categorical diagnosis -> Standard treatment

Computational phenotyping:
  Behavioral data + neural data
  -> Fit computational model (active inference parameters)
  -> Extract individual parameter profile:
     {Pi_sensory, Pi_prior, Pi_policy, learning_rate, ...}
  -> Map parameters to mechanisms
  -> Targeted intervention for specific computational deficit
```

### Parameter Recovery

```
Tasks used to estimate parameters:
  - Probabilistic reversal learning: learning rate, volatility sensitivity
  - Social learning tasks: social precision, mentalizing parameters
  - Sensory discrimination: sensory precision, prior influence
  - Decision-making under uncertainty: risk sensitivity, exploration-exploitation

Neuroimaging adds:
  - DCM connectivity parameters
  - Precision-weighted prediction error correlates
  - Network-level dynamics
```

### Clinical Applications

```
Treatment prediction:
  Pre-treatment parameters predict who responds to:
  - CBT (patients with updatable priors respond best)
  - SSRIs (patients with specific serotonergic precision deficits)
  - Antipsychotics (patients with specific dopaminergic precision deficits)

Treatment monitoring:
  Track parameter changes over treatment course
  - Learning rate normalization -> clinical improvement
  - Precision rebalancing -> symptom reduction
```

## Therapeutic Interventions Through the FEP Lens

### Cognitive Behavioral Therapy (CBT)

```
CBT = structured updating of the generative model:
  1. Identify maladaptive priors ("I am worthless")
  2. Reduce precision on these priors (cognitive restructuring)
  3. Increase precision on disconfirming evidence (behavioral experiments)
  4. Update the generative model (belief change)

FEP formulation:
  Before CBT: D_KL[q_maladaptive || p_true] is large
  CBT reduces this divergence by:
  - Lowering precision on maladaptive priors
  - Increasing precision on new evidence
  - Facilitating model updating
```

### Mindfulness and Meditation

```
Mindfulness = meta-awareness of precision allocation:
  1. Observe prediction errors without reacting (reduce action)
  2. Notice precision dynamics (meta-cognition)
  3. Reduce automatic precision allocation to rumination/worry
  4. Increase precision on present-moment sensory experience

Computational effect:
  Reduces Pi_prior for ruminative models
  Increases Pi_sensory for current experience
  -> Breaks the cycle of prediction-error-driven rumination
```

### Psychedelic-Assisted Therapy

```
Psychedelics (psilocybin, LSD) under the FEP:
  - 5-HT2A agonism -> relaxation of high-level priors
  - Reduced precision on entrenched beliefs
  - Increased entropy of posterior beliefs
  - "Anarchic" brain state: flat precision landscape

  F_psychedelic = E_q[-ln p(o,s)] - H_elevated[q(s)]
  -> Elevated entropy -> reduced free energy contribution from rigid priors
  -> Rigid beliefs become plastic, updatable

Therapeutic window (Carhart-Harris & Friston, 2019):
  During the psychedelic state, maladaptive priors lose their grip
  Therapeutic context provides new evidence
  Integration = forming new, more adaptive priors
  The REBUS model: RElaxed Beliefs Under pSychedelics
```

## Current Research

### Transdiagnostic Computational Models

Moving beyond diagnosis-specific models toward shared computational mechanisms:

```
Shared factor: Aberrant precision weighting
  Manifests differently depending on:
  - Which modality (sensory, interoceptive, proprioceptive)
  - Which level of hierarchy (low-level perception vs. high-level belief)
  - Which direction (too high vs. too low)
  - Which temporal dynamics (tonic vs. phasic)
```

### Digital Phenotyping

Using smartphone and wearable data to track computational parameters in real time:

```
Data streams: Movement, sleep, social interaction, typing patterns
-> Infer active inference parameters continuously
-> Detect early warning signs of relapse
-> Trigger just-in-time interventions
```

### Bayesian Clinical Trials

Designing clinical trials using Bayesian adaptive methods inspired by the FEP:

```
Trial as inference:
  Prior: p(treatment_effect | previous_data)
  Evidence: New patient outcomes
  Posterior: Updated belief about treatment efficacy

  Adaptive allocation: More patients assigned to more promising treatments
  Early stopping: When posterior confidence exceeds threshold
```

## Open Questions

1. **Specificity**: Can computational phenotyping reliably distinguish between disorders, or are the precision profiles too overlapping?
2. **Causality**: Do precision abnormalities cause psychiatric disorders, or are they consequences of other processes?
3. **Development**: How do precision-weighting abnormalities develop over the lifespan? What are the critical periods?
4. **Social context**: How do social environments shape precision dynamics? Can dysfunctional social systems create "psychiatric" computational profiles in healthy individuals?
5. **Treatment mechanisms**: Can we precisely target specific precision parameters with pharmacological or psychological interventions?
6. **Cultural variation**: Do optimal precision profiles vary across cultures? Is some "psychopathology" actually adaptive in certain contexts?

## References

1. Friston, K. J. (2017). Computational psychiatry: from synapses to sentience. *Molecular Psychiatry*, 28(1), 256-268.
2. Stephan, K. E., Mathys, C. D., & Friston, K. J. (2016). Allostatic self-efficacy: a metacognitive theory of dyshomeostasis-induced fatigue and depression. *Frontiers in Human Neuroscience*, 10, 550.
3. Adams, R. A., Stephan, K. E., Brown, H. R., Frith, C. D., & Friston, K. J. (2013). The computational anatomy of psychosis. *Frontiers in Psychiatry*, 4, 47.
4. Montague, P. R., Dolan, R. J., Friston, K. J., & Dayan, P. (2012). Computational psychiatry. *Trends in Cognitive Sciences*, 16(1), 72-80.
5. Huys, Q. J. M., Maia, T. V., & Frank, M. J. (2016). Computational psychiatry as a bridge from neuroscience to clinical applications. *Nature Neuroscience*, 19(3), 404-413.
6. Van de Cruys, S., Evers, K., Van der Hallen, R., Van Eylen, L., Boets, B., de-Wit, L., & Wagemans, J. (2014). Precise minds in uncertain worlds: predictive coding in autism. *Psychological Review*, 121(4), 649-675.
7. Paulus, M. P., Feinstein, J. S., & Khalsa, S. S. (2019). An active inference approach to interoceptive psychopathology. *Annual Review of Clinical Psychology*, 15, 97-122.
8. Carhart-Harris, R. L., & Friston, K. J. (2019). REBUS and the anarchic brain: toward a unified model of the brain action of psychedelics. *Pharmacological Reviews*, 71(3), 316-344.
9. Schwartenbeck, P., & Friston, K. (2016). Computational phenotyping in psychiatry: a worked example. *eNeuro*, 3(4), ENEURO.0049-16.2016.
10. Lawson, R. P., Rees, G., & Friston, K. J. (2014). An aberrant precision account of autism. *Frontiers in Human Neuroscience*, 8, 302.

## See Also

- [[neuroscience|Neuroscience Applications]]
- [[knowledge_base/free_energy_principle/cognitive/perception|Perception]]
- [[knowledge_base/free_energy_principle/biology/homeostasis|Homeostasis and Allostasis]]
- [[knowledge_base/free_energy_principle/cognitive/attention|Attention and Precision]]
- [[knowledge_base/free_energy_principle/cognitive/consciousness|Consciousness]]
- [[education|Educational Applications]]
