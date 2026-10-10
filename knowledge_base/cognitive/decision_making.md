---
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
---


# Decision Making


## Overview

Decision Making encompasses the cognitive and neural processes involved in choosing between alternative courses of action. It integrates multiple sources of information including sensory evidence, prior knowledge, value estimates, and uncertainty to guide adaptive behavior.

## Core Components

### Value Systems

- [[value_computation]] - Worth estimation

  - [[reward_processing]] - Benefit evaluation

    - [[primary_rewards]] - Basic needs

    - [[secondary_rewards]] - Learned values

  - [[cost_processing]] - Effort/risk

    - [[energy_costs]] - Resource use

    - [[opportunity_costs]] - Value forgone by choosing one alternative

### Evidence Processing

- [[evidence_accumulation]] - Information gathering

  - [[sensory_evidence]] - Perceptual input

    - [[signal_detection]] - Separating meaningful signals from background noise

    - noise filtering - Clean-up

  - [[memory_evidence]] - Stored information

    - [[episodic_recall]] - Past experiences

    - [[semantic_knowledge]] - Facts/concepts

### Choice Mechanisms

- [[choice_selection]] - Option picking

  - [[option_comparison]] - Alternative evaluation

    - [[value_comparison]] - Comparing the subjective values of options

    - [[probability_weighting]] - Nonlinear distortion of outcome probabilities

  - [[threshold_crossing]] - Evidence reaching the level that triggers a choice

    - [[decision_bounds]] - Choice criteria

    - [[urgency_signals]] - Time pressure

## Neural Implementation

### Brain Networks

- [[decision_networks]] - Neural systems

  - [[valuation_network]] - Worth coding

    - [[ventral_striatum]] - Subcortical hub for reward and motivation

    - [[orbitofrontal]] - Value integration

  - [[control_network]] - Choice regulation

    - [[dorsolateral_prefrontal]] - Working memory and executive control region

    - [[anterior_cingulate]] - Conflict and error monitoring region

### Circuit Mechanisms

- [[decision_circuits]] - Neural computation

  - [[accumulator_circuits]] - Evidence build-up

    - [[integration_neurons]] - Neurons that accumulate evidence over time

    - [[threshold_units]] - Units that fire once evidence passes a bound

  - [[comparison_circuits]] - Option evaluation

    - [[mutual_inhibition]] - Competing populations suppress each other's activity

    - [[winner_take_all]] - Strongest option suppresses all others

### Neuromodulation

- [[decision_modulation]] - Neurotransmitter influences on choice

  - [[dopamine_system]] - Reward/learning

    - [[reward_prediction]] - Anticipating the reward an action will bring

    - [[motivation_control]] - Regulating drive to pursue goals

  - [[serotonin_system]] - Patience/inhibition

    - [[temporal_discounting]] - Devaluing rewards that arrive later

    - [[behavioral_inhibition]] - Restraint

## Computational Framework

### Mathematical Models

- [[decision_models]] - Formal theories

  - [[drift_diffusion]] - Evidence accumulation

    - [[drift_rate]] - Speed of evidence accumulation toward a bound

    - [[decision_threshold]] - Choice criterion

  - [[race_models]] - Independent accumulators racing to a bound

    - [[accumulator_race]] - Parallel build-up

    - [[stopping_rules]] - Choice timing

### Optimization Principles

- [[decision_optimization]] - Best choice

  - [[utility_maximization]] - Value optimization

    - [[expected_utility]] - Probability-weighted average of outcome values

    - [[risk_preference]] - Uncertainty attitude

  - [[cost_minimization]] - Choosing options with the lowest total cost

    - [[effort_optimization]] - Balancing effort spent against expected payoff

    - [[time_optimization]] - Speed-accuracy

### Learning Processes

- [[decision_learning]] - Improving choices through experience

  - [[value_learning]] - Learning how rewarding each option is

    - [[reward_learning]] - Outcome association

    - [[punishment_learning]] - Learning to avoid actions with bad outcomes

  - strategy learning - Learning which approach suits which situation

    - [[policy_learning]] - Action rules

    - [[meta_learning]] - Learning how to learn and choose better

## Decision Types

### Perceptual Decisions

- [[perceptual_choice]] - Sensory-based

  - [[detection]] - Presence/absence

    - [[threshold_detection]] - Detecting the weakest noticeable stimulus

    - [[change_detection]] - Noticing changes between successive stimuli

  - [[discrimination]] - Difference judgment

    - [[feature_discrimination]] - Telling stimuli apart by a single attribute

    - [[pattern_discrimination]] - Telling apart complex multi-feature stimuli

### Value-based Decisions

- [[value_choice]] - Preference-based

  - [[economic_choice]] - Resource decisions

    - [[risk_decisions]] - Choices among options with uncertain outcomes

    - [[temporal_choice]] - Time trade-offs

  - [[social_choice]] - Interactive decisions

    - [[cooperation]] - Joint benefit

    - competition - Rivalry for limited resources or rewards

### Motor Decisions

- [[action_choice]] - Movement selection

  - [[target_selection]] - Goal choice

    - [[spatial_target]] - Choosing a goal by its location

    - [[object_target]] - Choosing a goal by the object itself

  - [[movement_selection]] - Choosing which movement to execute

    - [[trajectory_choice]] - Choosing the route a movement follows

    - [[timing_choice]] - Choosing when to start an action

## Applications

### Clinical Applications

- [[decision_disorders]] - Clinical disruptions of choice behavior

  - [[addiction]] - Maladaptive choice

    - [[substance_addiction]] - Drug-related

    - [[behavioral_addiction]] - Action-related

  - [[impulsivity]] - Acting without adequate forethought or restraint

    - [[premature_choice]] - Choosing before enough evidence is gathered

    - [[risky_choice]] - Choosing options with excessive risk of loss

### Economic Applications

- [[economic_decisions]] - Market behavior

  - [[consumer_choice]] - Buying decisions

    - [[product_choice]] - What to buy

    - [[price_sensitivity]] - Responsiveness of demand to price changes

  - [[investment_decisions]] - Allocating money in pursuit of future returns

    - [[portfolio_choice]] - Asset mix

    - [[risk_management]] - Identifying and limiting potential losses

### Technology Applications

- [[decision_technology]] - Support tools

  - decision support - Choice aids

    - [[information_systems]] - Systems that collect and present decision data

    - [[recommendation_systems]] - Suggestions

  - [[automated_decisions]] - Choices made by algorithms without human input

    - [[autonomous_systems]] - Self-deciding

    - [[hybrid_decisions]] - Human-AI

## Research Methods

### Behavioral Testing

- [[decision_tasks]] - Choice measurement

  - [[psychophysics]] - Relating physical stimuli to perceptual judgments

    - [[threshold_measurement]] - Estimating the smallest detectable stimulus

    - [[bias_measurement]] - Quantifying systematic leanings in choices

  - [[economic_games]] - Tasks with monetary stakes and strategic choices

    - [[gambling_tasks]] - Tasks probing risk-taking with uncertain payoffs

    - [[social_games]] - Tasks probing choices between interacting players

### Neural Recording

- [[decision_recording]] - Brain activity

  - [[single_unit]] - Neuron firing

    - [[choice_probability]] - How well a neuron's firing predicts the choice

    - [[value_coding]] - Neural signals that represent option value

  - [[population_recording]] - Group activity

    - [[network_dynamics]] - Time-varying activity patterns across a network

    - [[state_transitions]] - Shifts of neural state as a choice forms

### Computational Modeling

- [[decision_modeling]] - Theory testing

  - [[behavioral_models]] - Choice patterns

    - [[sequential_sampling]] - Models of evidence sampled over time until a bound

    - [[reinforcement_learning]] - Learning action values from reward feedback

  - [[neural_models]] - Brain mechanisms

    - [[circuit_models]] - Network function

    - [[system_models]] - Large-scale

## Future Directions

### Current Challenges

- complexity scaling - Real-world

- [[individual_differences]] - Person variation

- context dependence - Situation effects

### Emerging Approaches

- real time tracking - Online monitoring

- [[adaptive_systems]] - Systems that adjust to the user's behavior

- [[collective_decisions]] - Group choice

## References

- [[gold_shadlen_neural]]

- [[kahneman_tversky]]

- [[dolan_dayan]]

- [[rangel_hare]]

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

    - [[primary_rewards]] - Innate rewards such as food and water

      - [[survival_value]] - Value tied to staying alive

      - [[hedonic_value]] - Value derived from pleasure or enjoyment

    - [[secondary_rewards]] - Rewards valued through learned association

      - [[social_value]] - Value gained from social interaction

      - [[abstract_value]] - Value of ideas, money, and symbolic goods

  - [[cost_processing]] - Evaluating the costs of an action

    - [[energy_costs]] - Energy expended to carry out an action

      - [[metabolic_cost]] - Physical energy demands of an action

      - [[cognitive_cost]] - Mental effort required to decide or act

    - [[risk_costs]] - Expected losses from risky outcomes

      - [[uncertainty_cost]] - Penalty attached to unknown outcomes

      - [[loss_potential]] - Size of what could be lost

### Evidence Processing

- [[evidence_accumulation]] - Collecting evidence toward a decision

  - [[sensory_evidence]] - Evidence drawn from sensory input

    - [[perceptual_sampling]] - Repeatedly sampling stimulus information

      - [[feature_extraction]] - Pulling out informative stimulus features

      - [[pattern_recognition]] - Identifying familiar patterns in input

    - noise handling - Managing random variability in the signal

      - [[signal_detection]] - Telling true signals from noise

      - noise filtering - Removing irrelevant fluctuations from input

  - [[memory_evidence]] - Evidence retrieved from memory

    - [[episodic_recall]] - Recalling specific past events

      - [[past_outcomes]] - Results of earlier similar choices

      - [[similar_cases]] - Comparable prior situations

    - [[semantic_knowledge]] - General factual knowledge

      - [[rule_application]] - Applying learned rules to the choice

      - [[category_knowledge]] - Knowledge about category membership

### Choice Mechanisms

- [[choice_selection]] - Selecting among available options

  - [[option_evaluation]] - Assessing the merits of each option

    - [[value_comparison]] - Comparing option values against each other

      - [[relative_value]] - Value judged against alternatives

      - [[reference_points]] - Baselines against which outcomes are judged

    - [[probability_weighting]] - Subjective weighting of outcome probabilities

      - [[subjective_probability]] - Personal estimate of how likely an outcome is

      - [[risk_attitude]] - Tendency to seek or avoid risk

  - [[decision_criteria]] - Standards used to settle on a choice

    - [[threshold_setting]] - Setting how much evidence is needed to choose

      - [[evidence_threshold]] - Amount of evidence required to commit

      - [[urgency_signals]] - Growing pressure to decide as time passes

    - strategy selection - Choosing which decision approach to use

      - [[heuristics]] - Fast rule-of-thumb shortcuts

      - [[optimization]] - Searching for the best available option

## Neural Implementation

### Brain Networks

- [[decision_networks]] - Interacting brain regions underlying choice

  - [[valuation_network]] - Regions that compute subjective value

    - [[ventral_striatum]] - Reward and motivation hub

      - [[reward_signals]] - Neural signals indicating reward

      - [[motivation_signals]] - Neural signals driving effort toward goals

    - [[orbitofrontal]] - Region encoding the value of options

      - [[value_coding]] - Neural representation of option value

      - [[outcome_coding]] - Neural representation of expected outcomes

  - [[control_network]] - Regions that regulate and direct choices

    - [[prefrontal]] - Frontal cortex supporting planning and control

      - [[dorsolateral]] - Prefrontal region for planning and working memory

      - [[anterior_cingulate]] - Region that monitors conflict and errors

    - [[parietal]] - Parietal cortex integrating space and attention

      - [[evidence_accumulation]] - Parietal activity that builds toward a decision bound

      - [[action_planning]] - Preparing which movement to make

### Circuit Mechanisms

- [[decision_circuits]] - Neural pathways computing choices

  - [[accumulator_circuits]] - Circuits that sum evidence over time

    - [[evidence_integration]] - Combining evidence into a running total

      - [[temporal_integration]] - Summing evidence across time

      - [[spatial_integration]] - Combining evidence across space or sources

    - [[threshold_mechanisms]] - Mechanisms that trigger a choice at a bound

      - [[threshold_crossing]] - Moment evidence reaches the decision bound

      - [[winner_take_all]] - Strongest option suppresses its competitors

  - [[value_circuits]] - Circuits that compute option value

    - [[reward_computation]] - Computing the reward an option offers

      - [[utility_coding]] - Neural encoding of subjective utility

      - [[risk_coding]] - Neural encoding of outcome uncertainty

    - [[cost_computation]] - Computing the costs an option incurs

      - [[effort_coding]] - Neural encoding of required effort

      - [[loss_coding]] - Neural encoding of potential losses

### Neuromodulation

- [[decision_modulation]] - Neuromodulator effects on decision processes

  - [[dopamine_system]] - Dopamine pathways for reward and motivation

    - [[reward_prediction]] - Predicting upcoming reward

      - [[prediction_errors]] - Gap between expected and actual outcome

      - [[incentive_salience]] - Motivational pull of reward-related cues

    - [[action_motivation]] - Drive to initiate effortful actions

      - [[approach_behavior]] - Moving toward rewarding stimuli

      - [[exploration]] - Sampling unfamiliar options to gain information

  - [[noradrenaline_system]] - Noradrenaline pathways regulating arousal and uncertainty

    - [[uncertainty_signals]] - Signals that track unreliable information

      - [[surprise_detection]] - Noticing unexpected events

      - [[volatility_tracking]] - Tracking how quickly the environment changes

    - arousal control - Regulating alertness level

      - [[vigilance]] - Sustained readiness to detect events

      - [[stress_response]] - Physiological and behavioral reaction to stress

## Processing Principles

### Information Flow

- [[decision_processing]] - Stages through which a decision unfolds

  - [[bottom_up]] - Processing driven by sensory input

    - [[evidence_gathering]] - Collecting stimulus-driven evidence

      - [[sampling_processes]] - Drawing samples of information

      - [[integration_processes]] - Combining samples into a running estimate

    - [[value_construction]] - Building value estimates during choice

      - [[feature_valuation]] - Valuing options by their individual attributes

      - [[context_effects]] - Influence of surrounding options on choice

  - [[top_down]] - Processing guided by goals and expectations

    - [[goal_influence]] - Goals shaping which evidence gets attention

      - [[preference_bias]] - Existing preferences tilting the choice

      - [[strategy_bias]] - Chosen strategy tilting the choice

    - [[control_processes]] - Executive processes that direct decision behavior

      - [[inhibition]] - Suppressing unwanted responses

      - [[flexibility]] - Adapting strategy to changing demands

### Resource Management

- [[decision_resources]] - Cognitive and motivational capacity available for deciding

  - [[cognitive_resources]] - Attention and memory capacity for deciding

    - [[attention_allocation]] - Distributing attention among options and cues

      - [[selective_attention]] - Focusing on relevant information

      - [[sustained_attention]] - Maintaining focus over time

    - [[working_memory]] - Holding options in mind briefly

      - [[option_maintenance]] - Keeping candidate options active in mind

      - [[comparison_process]] - Weighing options against each other

  - [[motivational_resources]] - Drive available to sustain effort

    - [[effort_allocation]] - Deciding how much effort to invest

      - [[energy_management]] - Pacing energy expenditure over time

      - persistence - Continuing effort despite obstacles

    - [[emotional_resources]] - Capacity to regulate emotion while deciding

      - stress management - Coping with stress during decisions

      - [[mood_effects]] - Influence of mood on choices

### Integration Mechanisms

- [[decision_integration]] - Combining multiple inputs into one choice

  - [[multimodal_integration]] - Merging information from several senses

    - [[sensory_fusion]] - Merging sensory signals into a single estimate

      - [[visual_integration]] - Combining visual cues

      - [[auditory_integration]] - Combining auditory cues

    - [[evidence_combination]] - Pooling multiple pieces of evidence

      - [[weighted_averaging]] - Averaging cues in proportion to reliability

      - [[bayesian_integration]] - Combining cues and priors by Bayes' rule

  - [[value_integration]] - Combining attribute values into overall value

    - [[attribute_integration]] - Merging an option's attributes into one score

      - [[feature_combination]] - Combining separate features into a whole

      - [[dimension_reduction]] - Compressing many attributes into a few

    - [[temporal_integration]] - Combining information across time

      - [[history_effects]] - Influence of previous trials on current choice

      - [[future_projection]] - Anticipating likely future outcomes

## Applications

### Clinical Applications

- [[decision_disorders]] - Conditions that impair choice behavior

  - [[impulsivity]] - Acting without adequate forethought

    - [[response_inhibition]] - Impaired ability to withhold responses

      - [[premature_response]] - Responding before the evidence warrants

      - [[perseveration]] - Repeating a response despite changed demands

    - [[risk_taking]] - Seeking options with a high chance of harm

      - [[reward_sensitivity]] - Heightened drive toward rewards

      - [[punishment_insensitivity]] - Reduced learning from negative consequences

  - [[indecision]] - Difficulty committing to a choice

    - [[choice_paralysis]] - Can't pick

      - [[overthinking]] - Excessive deliberation that delays choice

      - [[avoidance]] - Evading decisions to escape discomfort

    - [[ambivalence]] - Holding conflicting feelings about options

      - [[conflict_anxiety]] - Anxiety arising from competing options

      - [[commitment_fear]] - Fear of being bound to a choice

### Therapeutic Approaches

- [[decision_therapy]] - Therapies that improve decision-making

  - [[cognitive_strategies]] - Mental techniques that improve choices

    - [[decision_analysis]] - Structured breakdown of a choice

      - [[option_evaluation]] - Assessing the merits of each option

      - [[consequence_analysis]] - Considering likely results of each option

    - [[bias_awareness]] - Recognizing one's own decision biases

      - [[bias_identification]] - Spotting specific biases in one's choices

      - [[debiasing_techniques]] - Methods that reduce biased judgment

  - [[behavioral_strategies]] - Behavior-based methods for better choices

    - [[exposure_training]] - Gradual confrontation with feared decisions

      - [[gradual_exposure]] - Stepwise exposure to rising difficulty

      - [[feedback_learning]] - Learning from outcomes of past choices

    - [[skill_building]] - Developing practical decision skills

      - [[coping_skills]] - Strategies for managing stress and setbacks

      - [[planning_skills]] - Skills for setting goals and sequencing steps

### Technology Applications

- [[decision_technology]] - Technology that supports decision-making

  - decision support - Tools that assist people in choosing

    - [[analysis_tools]] - Software for analyzing options and data

      - [[option_comparison]] - Tools for side-by-side comparison of options

      - [[risk_assessment]] - Tools that estimate the likelihood and impact of risk

    - [[recommendation_systems]] - Systems that suggest options based on preferences

      - [[preference_learning]] - Inferring a user's preferences from behavior

      - [[personalization]] - Tailoring advice to the individual

  - monitoring systems - Systems that track behavior and outcomes

    - behavior tracking - Recording what people actually do

      - [[choice_patterns]] - Recurring regularities in a person's choices

      - [[outcome_tracking]] - Recording the results of decisions

    - [[adaptation_systems]] - Systems that adjust to changing conditions

      - [[dynamic_adjustment]] - Updating recommendations as conditions change

      - [[learning_algorithms]] - Algorithms that improve from experience

## Future Directions

### Current Challenges

- measurement issues - Difficulty measuring decision processes reliably

  - [[preference_stability]] - Whether preferences stay constant over time

  - context dependence - Choices shifting with the situation

- mechanism understanding - Explaining how decision processes operate

  - [[integration_processes]] - How separate inputs are combined

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

- [[dolan_dayan]]

- [[rangel_hare]]

- [[rushworth_decision]]

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
