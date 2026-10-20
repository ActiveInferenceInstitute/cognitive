---
external_mappings:
  - target: http://www.wikidata.org/entity/Q1331926
    relation: exactMatch
    evidence: https://www.wikidata.org/wiki/Q1331926
    rationale: "The overview defines choosing between alternative courses of action; the target defines the cognitive process of choosing an action or belief."
    reviewed_on: "2026-10-20"
title: Decision Making
type: concept
status: stable
tags:
- cognition
- decision
- computation
- behavior
- neural_systems
aliases:
  - decision_models
semantic_relations:
  - type: implements
    links:
      - "[[cognitive_control]]"
  - type: related
    links:
      - "[[value_computation]]"
      - "[[reinforcement_learning]]"
      - "[[action_selection]]"
wikidata: Q1331926
---


# Decision Making


## Overview

Decision Making encompasses the cognitive and neural processes involved in choosing between alternative courses of action. It integrates multiple sources of information including sensory evidence, prior knowledge, value estimates, and uncertainty to guide adaptive behavior.

## Core Components

### Value Systems

- [[value_computation]] - Worth estimation

  - [[reward_processing]] - Benefit evaluation

    - primary rewards - Basic needs

    - secondary rewards - Learned values

  - cost processing - Effort/risk

    - energy costs - Resource use

    - [[opportunity_costs]] - Value forgone by choosing one alternative

### Evidence Processing

- [[evidence_accumulation]] - Information gathering

  - [[sensory_evidence]] - Perceptual input

    - [[signal_detection]] - Separating meaningful signals from background noise

    - noise filtering - Clean-up

  - memory evidence - Stored information

    - [[episodic_recall]] - Past experiences

    - [[semantic_knowledge]] - Facts/concepts

### Choice Mechanisms

- choice selection - Option picking

  - option comparison - Alternative evaluation

    - [[value_comparison]] - Comparing the subjective values of options

    - [[probability_weighting]] - Nonlinear distortion of outcome probabilities

  - [[threshold_crossing]] - Evidence reaching the level that triggers a choice

    - decision bounds - Choice criteria

    - urgency signals - Time pressure

## Neural Implementation

### Brain Networks

- decision networks - Neural systems

  - valuation network - Worth coding

    - [[ventral_striatum]] - Subcortical hub for reward and motivation

    - [[orbitofrontal]] - Value integration

  - [[control_network]] - Choice regulation

    - [[dorsolateral_prefrontal]] - Working memory and executive control region

    - [[anterior_cingulate]] - Conflict and error monitoring region

### Circuit Mechanisms

- decision circuits - Neural computation

  - [[accumulator_circuits]] - Evidence build-up

    - integration neurons - Neurons that accumulate evidence over time

    - threshold units - Units that fire once evidence passes a bound

  - comparison circuits - Option evaluation

    - [[mutual_inhibition]] - Competing populations suppress each other's activity

    - [[winner_take_all]] - Strongest option suppresses all others

### Neuromodulation

- decision modulation - Neurotransmitter influences on choice

  - [[dopamine_system]] - Reward/learning

    - reward prediction - Anticipating the reward an action will bring

    - motivation control - Regulating drive to pursue goals

  - [[serotonin_system]] - Patience/inhibition

    - [[temporal_discounting]] - Devaluing rewards that arrive later

    - [[behavioral_inhibition]] - Restraint

## Computational Framework

### Mathematical Models

- [[decision_models]] - Formal theories

  - [[drift_diffusion]] - Evidence accumulation

    - [[sequential_sampling|drift rate]] - Speed of evidence accumulation toward a bound

    - [[decision_threshold]] - Choice criterion

  - [[sequential_sampling|race models]] - Independent accumulators racing to a bound

    - accumulator race - Parallel build-up

    - [[stopping_rules]] - Choice timing

### Optimization Principles

- decision optimization - Best choice

  - [[utility_maximization]] - Value optimization

    - [[expected_utility]] - Probability-weighted average of outcome values

    - [[risk_preference]] - Uncertainty attitude

  - cost minimization - Choosing options with the lowest total cost

    - effort optimization - Balancing effort spent against expected payoff

    - time optimization - Speed-accuracy

### Learning Processes

- decision learning - Improving choices through experience

  - [[value_learning]] - Learning how rewarding each option is

    - reward learning - Outcome association

    - punishment learning - Learning to avoid actions with bad outcomes

  - strategy learning - Learning which approach suits which situation

    - [[policy_learning]] - Action rules

    - [[meta_learning]] - Learning how to learn and choose better

## Decision Types

### Perceptual Decisions

- perceptual choice - Sensory-based

  - [[detection]] - Presence/absence

    - threshold detection - Detecting the weakest noticeable stimulus

    - change detection - Noticing changes between successive stimuli

  - [[discrimination]] - Difference judgment

    - feature discrimination - Telling stimuli apart by a single attribute

    - pattern discrimination - Telling apart complex multi-feature stimuli

### Value-based Decisions

- value choice - Preference-based

  - economic choice - Resource decisions

    - risk decisions - Choices among options with uncertain outcomes

    - temporal choice - Time trade-offs

  - social choice - Interactive decisions

    - [[cooperation]] - Joint benefit

    - competition - Rivalry for limited resources or rewards

### Motor Decisions

- [[action_choice]] - Movement selection

  - [[target_selection]] - Goal choice

    - spatial target - Choosing a goal by its location

    - object target - Choosing a goal by the object itself

  - movement selection - Choosing which movement to execute

    - trajectory choice - Choosing the route a movement follows

    - timing choice - Choosing when to start an action

## Applications

### Clinical Applications

- decision disorders - Clinical disruptions of choice behavior

  - [[addiction]] - Maladaptive choice

    - [[addiction|substance addiction]] - Drug-related

    - [[addiction|behavioral addiction]] - Action-related

  - [[impulsivity]] - Acting without adequate forethought or restraint

    - premature choice - Choosing before enough evidence is gathered

    - risky choice - Choosing options with excessive risk of loss

### Economic Applications

- economic decisions - Market behavior

  - consumer choice - Buying decisions

    - product choice - What to buy

    - price sensitivity - Responsiveness of demand to price changes

  - investment decisions - Allocating money in pursuit of future returns

    - portfolio choice - Asset mix

    - risk management - Identifying and limiting potential losses

### Technology Applications

- decision technology - Support tools

  - decision support - Choice aids

    - information systems - Systems that collect and present decision data

    - recommendation systems - Suggestions

  - automated decisions - Choices made by algorithms without human input

    - [[autonomous_systems]] - Self-deciding

    - hybrid decisions - Human-AI

## Research Methods

### Behavioral Testing

- decision tasks - Choice measurement

  - [[psychophysics]] - Relating physical stimuli to perceptual judgments

    - [[threshold_measurement]] - Estimating the smallest detectable stimulus

    - bias measurement - Quantifying systematic leanings in choices

  - [[economic_games]] - Tasks with monetary stakes and strategic choices

    - gambling tasks - Tasks probing risk-taking with uncertain payoffs

    - social games - Tasks probing choices between interacting players

### Neural Recording

- decision recording - Brain activity

  - [[single_unit]] - Neuron firing

    - choice probability - How well a neuron's firing predicts the choice

    - [[value_coding]] - Neural signals that represent option value

  - population recording - Group activity

    - [[network_dynamics]] - Time-varying activity patterns across a network

    - [[state_transitions]] - Shifts of neural state as a choice forms

### Computational Modeling

- decision modeling - Theory testing

  - behavioral models - Choice patterns

    - [[sequential_sampling]] - Models of evidence sampled over time until a bound

    - [[reinforcement_learning]] - Learning action values from reward feedback

  - neural models - Brain mechanisms

    - circuit models - Network function

    - system models - Large-scale

## Future Directions

### Current Challenges

- complexity scaling - Real-world

- [[individual_differences]] - Person variation

- context dependence - Situation effects

### Emerging Approaches

- real time tracking - Online monitoring

- [[adaptive_systems]] - Systems that adjust to the user's behavior

- collective decisions - Group choice

## References

- gold shadlen neural

- [[kahneman_tversky]]

- dolan dayan

- rangel hare

## Related Concepts

- [[reinforcement_learning]]

- [[cognitive_control]]

- [[value_computation]]

- [[action_selection]]

- [[motivation]]

- [[emotion]]

### Value Systems

- [[value_computation]] - Estimating the subjective value of options

  - [[reward_processing]] - Evaluating how rewarding an outcome is

    - primary rewards - Innate rewards such as food and water

      - survival value - Value tied to staying alive

      - hedonic value - Value derived from pleasure or enjoyment

    - secondary rewards - Rewards valued through learned association

      - social value - Value gained from social interaction

      - abstract value - Value of ideas, money, and symbolic goods

  - cost processing - Evaluating the costs of an action

    - energy costs - Energy expended to carry out an action

      - metabolic cost - Physical energy demands of an action

      - cognitive cost - Mental effort required to decide or act

    - risk costs - Expected losses from risky outcomes

      - uncertainty cost - Penalty attached to unknown outcomes

      - loss potential - Size of what could be lost

### Evidence Processing

- [[evidence_accumulation]] - Collecting evidence toward a decision

  - [[sensory_evidence]] - Evidence drawn from sensory input

    - perceptual sampling - Repeatedly sampling stimulus information

      - [[feature_extraction]] - Pulling out informative stimulus features

      - [[pattern_recognition]] - Identifying familiar patterns in input

    - noise handling - Managing random variability in the signal

      - [[signal_detection]] - Telling true signals from noise

      - noise filtering - Removing irrelevant fluctuations from input

  - memory evidence - Evidence retrieved from memory

    - [[episodic_recall]] - Recalling specific past events

      - past outcomes - Results of earlier similar choices

      - similar cases - Comparable prior situations

    - [[semantic_knowledge]] - General factual knowledge

      - rule application - Applying learned rules to the choice

      - category knowledge - Knowledge about category membership

### Choice Mechanisms

- choice selection - Selecting among available options

  - [[option_evaluation]] - Assessing the merits of each option

    - [[value_comparison]] - Comparing option values against each other

      - relative value - Value judged against alternatives

      - [[risk_preference|reference points]] - Baselines against which outcomes are judged

    - [[probability_weighting]] - Subjective weighting of outcome probabilities

      - [[subjective_probability]] - Personal estimate of how likely an outcome is

      - risk attitude - Tendency to seek or avoid risk

  - decision criteria - Standards used to settle on a choice

    - threshold setting - Setting how much evidence is needed to choose

      - evidence threshold - Amount of evidence required to commit

      - urgency signals - Growing pressure to decide as time passes

    - strategy selection - Choosing which decision approach to use

      - [[heuristics]] - Fast rule-of-thumb shortcuts

      - [[optimization]] - Searching for the best available option

## Neural Implementation

### Brain Networks

- decision networks - Interacting brain regions underlying choice

  - valuation network - Regions that compute subjective value

    - [[ventral_striatum]] - Reward and motivation hub

      - [[reward_signals]] - Neural signals indicating reward

      - motivation signals - Neural signals driving effort toward goals

    - [[orbitofrontal]] - Region encoding the value of options

      - [[value_coding]] - Neural representation of option value

      - outcome coding - Neural representation of expected outcomes

  - [[control_network]] - Regions that regulate and direct choices

    - [[prefrontal]] - Frontal cortex supporting planning and control

      - [[dorsolateral]] - Prefrontal region for planning and working memory

      - [[anterior_cingulate]] - Region that monitors conflict and errors

    - [[parietal]] - Parietal cortex integrating space and attention

      - [[evidence_accumulation]] - Parietal activity that builds toward a decision bound

      - [[action_planning]] - Preparing which movement to make

### Circuit Mechanisms

- decision circuits - Neural pathways computing choices

  - [[accumulator_circuits]] - Circuits that sum evidence over time

    - [[evidence_integration]] - Combining evidence into a running total

      - [[temporal_integration]] - Summing evidence across time

      - spatial integration - Combining evidence across space or sources

    - threshold mechanisms - Mechanisms that trigger a choice at a bound

      - [[threshold_crossing]] - Moment evidence reaches the decision bound

      - [[winner_take_all]] - Strongest option suppresses its competitors

  - value circuits - Circuits that compute option value

    - reward computation - Computing the reward an option offers

      - utility coding - Neural encoding of subjective utility

      - risk coding - Neural encoding of outcome uncertainty

    - cost computation - Computing the costs an option incurs

      - effort coding - Neural encoding of required effort

      - loss coding - Neural encoding of potential losses

### Neuromodulation

- decision modulation - Neuromodulator effects on decision processes

  - [[dopamine_system]] - Dopamine pathways for reward and motivation

    - reward prediction - Predicting upcoming reward

      - [[prediction_errors]] - Gap between expected and actual outcome

      - [[incentive_salience]] - Motivational pull of reward-related cues

    - action motivation - Drive to initiate effortful actions

      - approach behavior - Moving toward rewarding stimuli

      - [[exploration]] - Sampling unfamiliar options to gain information

  - [[noradrenaline_system]] - Noradrenaline pathways regulating arousal and uncertainty

    - uncertainty signals - Signals that track unreliable information

      - surprise detection - Noticing unexpected events

      - volatility tracking - Tracking how quickly the environment changes

    - arousal control - Regulating alertness level

      - [[vigilance]] - Sustained readiness to detect events

      - [[stress_response]] - Physiological and behavioral reaction to stress

## Processing Principles

### Information Flow

- decision processing - Stages through which a decision unfolds

  - [[bottom_up]] - Processing driven by sensory input

    - evidence gathering - Collecting stimulus-driven evidence

      - sampling processes - Drawing samples of information

      - integration processes - Combining samples into a running estimate

    - value construction - Building value estimates during choice

      - feature valuation - Valuing options by their individual attributes

      - [[context_effects]] - Influence of surrounding options on choice

  - [[top_down]] - Processing guided by goals and expectations

    - goal influence - Goals shaping which evidence gets attention

      - preference bias - Existing preferences tilting the choice

      - strategy bias - Chosen strategy tilting the choice

    - [[control_processes]] - Executive processes that direct decision behavior

      - [[inhibition]] - Suppressing unwanted responses

      - [[flexibility]] - Adapting strategy to changing demands

### Resource Management

- decision resources - Cognitive and motivational capacity available for deciding

  - cognitive resources - Attention and memory capacity for deciding

    - [[attention_allocation]] - Distributing attention among options and cues

      - [[selective_attention]] - Focusing on relevant information

      - [[sustained_attention]] - Maintaining focus over time

    - [[working_memory]] - Holding options in mind briefly

      - option maintenance - Keeping candidate options active in mind

      - comparison process - Weighing options against each other

  - motivational resources - Drive available to sustain effort

    - [[effort_allocation]] - Deciding how much effort to invest

      - energy management - Pacing energy expenditure over time

      - persistence - Continuing effort despite obstacles

    - emotional resources - Capacity to regulate emotion while deciding

      - stress management - Coping with stress during decisions

      - mood effects - Influence of mood on choices

### Integration Mechanisms

- decision integration - Combining multiple inputs into one choice

  - [[multimodal_integration]] - Merging information from several senses

    - [[sensory_fusion]] - Merging sensory signals into a single estimate

      - visual integration - Combining visual cues

      - auditory integration - Combining auditory cues

    - evidence combination - Pooling multiple pieces of evidence

      - weighted averaging - Averaging cues in proportion to reliability

      - bayesian integration - Combining cues and priors by Bayes' rule

  - [[value_integration]] - Combining attribute values into overall value

    - attribute integration - Merging an option's attributes into one score

      - feature combination - Combining separate features into a whole

      - [[dimension_reduction]] - Compressing many attributes into a few

    - [[temporal_integration]] - Combining information across time

      - history effects - Influence of previous trials on current choice

      - future projection - Anticipating likely future outcomes

## Applications

### Clinical Applications

- decision disorders - Conditions that impair choice behavior

  - [[impulsivity]] - Acting without adequate forethought

    - [[response_inhibition]] - Impaired ability to withhold responses

      - premature response - Responding before the evidence warrants

      - [[perseveration]] - Repeating a response despite changed demands

    - risk taking - Seeking options with a high chance of harm

      - reward sensitivity - Heightened drive toward rewards

      - punishment insensitivity - Reduced learning from negative consequences

  - indecision - Difficulty committing to a choice

    - [[choice_paralysis]] - Can't pick

      - overthinking - Excessive deliberation that delays choice

      - avoidance - Evading decisions to escape discomfort

    - [[ambivalence]] - Holding conflicting feelings about options

      - conflict anxiety - Anxiety arising from competing options

      - commitment fear - Fear of being bound to a choice

### Therapeutic Approaches

- decision therapy - Therapies that improve decision-making

  - cognitive strategies - Mental techniques that improve choices

    - [[decision_analysis]] - Structured breakdown of a choice

      - [[option_evaluation]] - Assessing the merits of each option

      - consequence analysis - Considering likely results of each option

    - bias awareness - Recognizing one's own decision biases

      - bias identification - Spotting specific biases in one's choices

      - debiasing techniques - Methods that reduce biased judgment

  - behavioral strategies - Behavior-based methods for better choices

    - exposure training - Gradual confrontation with feared decisions

      - gradual exposure - Stepwise exposure to rising difficulty

      - [[feedback_learning]] - Learning from outcomes of past choices

    - skill building - Developing practical decision skills

      - coping skills - Strategies for managing stress and setbacks

      - planning skills - Skills for setting goals and sequencing steps

### Technology Applications

- decision technology - Technology that supports decision-making

  - decision support - Tools that assist people in choosing

    - [[analysis_tools]] - Software for analyzing options and data

      - option comparison - Tools for side-by-side comparison of options

      - [[risk_assessment]] - Tools that estimate the likelihood and impact of risk

    - recommendation systems - Systems that suggest options based on preferences

      - [[preference_learning]] - Inferring a user's preferences from behavior

      - personalization - Tailoring advice to the individual

  - monitoring systems - Systems that track behavior and outcomes

    - behavior tracking - Recording what people actually do

      - choice patterns - Recurring regularities in a person's choices

      - outcome tracking - Recording the results of decisions

    - adaptation systems - Systems that adjust to changing conditions

      - dynamic adjustment - Updating recommendations as conditions change

      - [[learning_algorithms]] - Algorithms that improve from experience

## Future Directions

### Current Challenges

- measurement issues - Difficulty measuring decision processes reliably

  - preference stability - Whether preferences stay constant over time

  - context dependence - Choices shifting with the situation

- mechanism understanding - Explaining how decision processes operate

  - integration processes - How separate inputs are combined

  - [[individual_differences]] - Variation in decision style across people

### Emerging Approaches

- new methods - Emerging techniques for studying decisions

  - real time measures - Measures taken as decisions unfold

    - [[neural_decoding]] - Inferring choices from neural activity

    - behavior tracking - Recording what people actually do

  - [[computational_models]] - Formal models of decision processes

    - [[learning_models]] - Models of how choices change with experience

    - hybrid models - Models combining several modeling approaches

## References

- [[kahneman_tversky]]

- dolan dayan

- rangel hare

- rushworth decision

## Related Concepts

- [[value_computation]]

- [[reinforcement_learning]]

- [[action_selection]]

- [[emotional_processing]]

- [[cognitive_control]]

- [[working_memory]]

- [[attention]]

- [[motivation]]

## See also

- [[knowledge_base/free_energy_principle/cognitive/decision_making|Decision Making Through Expected Free Energy Minimization]]
