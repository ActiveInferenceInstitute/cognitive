# Chapter 6: A Recipe for Designing [[active_inference|Active Inference]] Models

> "Give me six hours to chop down a tree and I will spend the first four sharpening the axe."
> — Abraham Lincoln

## 6.1 Introduction

This chapter provides a four-step recipe to construct an [[active_inference|Active Inference]] model, discussing the most important design choices one has to make to realize a model and providing some guidelines for those choices. It serves as an introduction to the second part of the book, which will illustrate several specific [[computational_efficiency|computational models]] using [[active_inference|Active Inference]] and their applications in a variety of [[cognitive_phenomena|cognitive domains]].

🎯 Core Concepts:
- [[active_inference|Active Inference]] as a normative approach
- [[behavioral_biology|Behavioral]] processes
- [[cognitive_phenomena|Cognitive]] processes
- [[neural_architectures|Neural]] processes
- [[free_energy_principle|Free Energy]] minimization
- [[model_architecture|Generative Models]]
- [[probabilistic_inference|Probabilistic Inference]]
- [[belief_updating|Belief Updating]]
- [[uncertainty_estimation|Uncertainty Estimation]]

The [[model_architecture|generative modeling]] approach is used in several disciplines for:
- [[cognitive_modeling_concepts|Cognitive Models]]
- [[bayesian_inference|Statistical Modeling]]
- [[evidence_accumulation|Experimental Data Analysis]]
- [[model_architecture|Machine Learning]]
- [[perceptual_inference|Perceptual Models]]
- [[semantic_memory|Knowledge Representation]]

📋 Design Methodology Examples:
1. [[predictive_coding|Predictive Coding]]: Perception as inference about sensations
2. [[temporal_models|Discrete Time Models]]: Planning as inference about actions
3. [[spatial_attention|Spatial Navigation]]: Planning with spatial variables
4. [[visual_perception|Visual Search]]: Planning with saccades
5. [[cross_modal_perception|Cross-Modal Integration]]: Multisensory inference
6. [[temporal_binding|Temporal Integration]]: Sequential processing

🔍 Key Design Questions:
1. Which [[complex_systems_biology|system]] are we modeling?
2. What is the most appropriate form for the [[generative model]]?
3. How to set up the [[generative model]]?
4. How to set up the [[active_inference_agent|generative process]]?

As [[Active Inference]] is a normative approach, it tries to explain as much as possible about [[behavioral_biology|behavior]], [[cognitive_phenomena|cognitive]], and [[neural_architectures|neural]] processes from first principles. Consistently, the design philosophy of [[Active Inference]] is top-down. Unlike many other approaches to [[neuroscience|computational neuroscience]], the challenge is not to emulate a brain, piece by piece, but to find the [[model_architecture|generative model]] that describes the problem the brain is trying to solve. Once the problem is appropriately formalized in terms of a [[model_architecture|generative model]], the solution to the problem emerges under [[Active Inference]]—with accompanying predictions about brains and minds. In other words, the generative model provides a complete description of a system of interest. The resulting behavior, inference, and neural dynamics can all be derived from a model by minimizing [[Free Energy]].

The generative modeling approach is used in several disciplines for the realization of cognitive models, statistical modeling, experimental data analysis, and machine learning (Hinton 2007b; Lee and Wagenmakers 2014; Pezzulo, Rigoli, and Friston 2015; Allen et al. 2019; Foster 2019). Here, we are primarily interested in designing generative models that engender cognitive processes of interest. We have seen this design methodology in previous chapters. For example, using a generative model for predictive coding, perception was cast as an inference about the most likely cause of sensations; using a generative model that evolves in discrete time, planning was cast as an inference about the most likely course of action. Depending on the problem of interest (e.g., planning during spatial navigation or planning saccades during visual search), one can adapt the form of these generative models to equip them with different structures (e.g., shallow or hierarchical) and variables (e.g., beliefs about allocentric or egocentric spatial locations). Importantly, Active Inference may take on many different guises under different assumptions about the form of the generative model being optimized. For example, assumptions about models that evolve in discrete or continuous time influence the form of the message passing (see chapter 4). This implies that the choice of a generative model corresponds to specific predictions about both behavior and neurobiology.

This flexibility is useful as it allows us to use the same language to describe processes in multiple domains. However, it can also be confusing from a practical perspective, as there are a number of choices that must be made to find the appropriate level of description for the system of interest. In the second part of this book, we will try to resolve this confusion through a series of illustrative examples of Active Inference in silico. This chapter introduces a general recipe for the design of Active Inference models, highlighting some of the key design choices, distinctions, and dichotomies that will appear in the numerical analysis of computational models described in subsequent chapters.

## 6.2 Designing an [[active_inference|Active Inference]] Model: A Recipe in Four Steps

Designing an Active Inference model requires four foundational steps, each resolving a specific design question:
1. Which [[complex_systems_biology|system]] are we modeling? The first choice to make is always the system of interest. This may not be as simple as it seems; it rests on the identification of the boundaries (i.e., Markov blanket) of that system. What counts as an Active Inference agent (generative model), what counts as the external environment (generative process), and what is the interface (sensory data and actions) between them?
2. What is the most appropriate form for the [[generative model]]? The first of the next three practical challenges is deciding whether it is appropriate to think of a process more in terms of categorical (discrete) inferences or continuous inferences, motivating the choice between discrete or continuous-time implementations (or a hybrid) of Active Inference. Then we need to select the most appropriate hierarchical depth, motivating the choice between shallow versus deep models. Finally, we need to consider whether it is necessary to endow generative models with temporal depth and the ability to predict action-contingent observations to support planning.
3. How to set up the [[generative model]]? What are the generative model's most appropriate variables and priors? Which parts are fixed and what must be learned? We emphasize the importance of choosing the right sort of variables and prior beliefs; furthermore, we emphasize a separation in timescales between the (faster) update of state variables that occurs during inference and the (slower) update of model parameters that occurs during learning.
4. How to set up the generative process? What are the elements of the generative process (and how do they differ from the generative model)? These four steps (in most cases) suffice to design an Active Inference model. Once completed, the behavior of the system is determined by the standard schemes of Active Inference: the descent of the active and internal states on the free energy functional associated with the model. From a more practical perspective, once one has specified the generative model and generative process, one can use standard Active Inference software routines to obtain numerical results, as well as to perform data visualization, analysis, and fitting (e.g., model-based data analysis). In what follows, we will review the four design choices in order.

## 6.3 What System Are We Modeling?

🌐 System Boundaries:
- [[neural_architectures|Neural Systems]]
- [[cell_biology|Cellular Systems]]
- [[systems_biology|Biological Systems]]
- [[emergence_self_organization|Emergent Systems]]

🔄 Interaction Components:
- [[sensorimotor_coordination|Sensory Receptors]]
- [[motor_control|Motor Effectors]]
- [[homeostatic_regulation|System Boundaries]]
- [[homeostatic_regulation|Internal States]]

📊 Variable Types:
- [[sensorimotor_coordination|Sensory States]]
- [[action_selection|Active States]]
- [[information_processing|Information States]]
- [[uncertainty_estimation|Uncertainty States]]

A useful first step in applying the formalism of [[active_inference|Active Inference]] is to identify the boundaries of the system of interest because we are interested in characterizing the interaction between what is internal to a system and the external world via [[sensorimotor_coordination|sensory receptors]] and effectors (e.g., muscles or glands). As discussed in chapter 3, a formal way to characterize the distinction between internal states of a system and external variables (and intermediate variables that mediate their interactions) is in terms of a [[Markov blanket]] (Pearl 1988).

The [[Markov blanket]] may be subdivided into two sorts of variables (Friston 2013):
1. Those that mediate the influence of the external world on internal states ([[sensorimotor_coordination|sensory states]])
2. Those that mediate the influence of internal states on the external world ([[action_selection|active states]])

## 6.4 What Is the Most Appropriate Form for the [[model_architecture|Generative Model]]?

🔢 Variable Categories:
1. Discrete Variables:
   - [[object_based_attention|Object Identities]]
   - [[decision_making|Action Plans]]
   - [[pattern_recognition|Pattern Categories]]
   - [[semantic_memory|Semantic Concepts]]
   - [[working_memory|Memory States]]

2. Continuous Variables:
   - [[motion_perception|Position/Velocity]]
   - [[biological_motion|Biological Motion]]
   - [[sensorimotor_coordination|Muscle Length]]
   - [[brightness_perception|Luminance]]
   - [[continuous_time_active_inference|Continuous Dynamics]]

3. Processing Considerations:
   - [[temporal_models|Temporal Processing]]
   - [[spatial_attention|Spatial Processing]]
   - [[pattern_recognition|Pattern Processing]]
   - [[information_processing|Information Flow]]
   - [[cognitive_control|Control Mechanisms]]

### 6.4.1 Discrete or Continuous Variables (or Both)?

The first design choice is to consider whether [[model_architecture|generative models]] that use discrete or continuous variables are more appropriate. The distinction between these approaches has important implications for:

- [[temporal_models|Temporal Processing]]
- [[neural_architectures|Neural Implementation]]
- [[predictive_coding|Message Passing]]
- [[hierarchical_inference|Hierarchical Organization]]
- [[active_inference_loop|Processing Cycles]]

### 6.4.2 Timescales of Inference: Shallow versus [[hierarchical_inference|Hierarchical Models]]

⏱️ Hierarchical Organization:
1. Temporal Scales:
   - [[temporal_binding|Fast Binding]]
   - [[temporal_models|Medium-term Dynamics]]
   - [[working_memory|Sustained Processing]]

2. Processing Levels:
   - [[hierarchical_inference|Hierarchical Processing]]
   - [[predictive_coding|Predictive Hierarchies]]
   - [[neural_architectures|Neural Hierarchies]]

3. Integration Mechanisms:
   - [[multisensory_integration|Cross-modal Integration]]
   - [[perceptual_binding|Feature Binding]]
   - [[temporal_binding|Temporal Integration]]

The second design choice concerns the timescales of [[Active Inference]]. One can select either (shallow) [[generative model|generative models]], in which all the variables evolve at the same timescale, or ([[hierarchical_inference|hierarchical]] or deep) models, which include variables that evolve at different timescales: slower for higher levels and faster for lower levels.

While many simple cognitive models only require shallow models, these are not sufficient when there is a clear separation of timescales between different aspects of a cognitive process of interest. One example of this is in language processing, in which short sequences of phonemes are contextualized by the word that is spoken and short sequences of words are contextualized by the current sentence. Crucially, the duration of the word transcends that of any one phoneme in the sequence and the duration of the sentence transcends that of any one word in the sequence. Hence, to model language processing, one can consider a hierarchical model in which sentences, words, and phonemes appear at different (higher to lower) hierarchical levels and evolve over (slower to faster) timescales that are approximately independent of one another. This is only an approximate separation, as levels must influence each other (e.g., the sentence influences the next words in the sequence; the word influences the next phonemes in the sequence). However, this does not mean we need to attempt to model the entire brain to develop meaningful simulations of a single level. For example, if we wanted to focus on word processing, we could address some aspects without having to deal with phoneme processing. This means we can treat input from parts of the brain drawing inferences about phonemes as providing observations from the perspective of word-processing areas. Phrasing this in terms of a Markov blanket, this typically means we treat the inferences performed by lower levels of a model as part of the sensory states of the blanket. This means we can summarize the inferences performed at the timescale of interest without having to specify the details of lower-level (faster) inferential processes—and this hierarchical factorization entails great computational benefits.

Another example is in the domain of intentional action selection, where the same goal (enter your apartment) can be active for an extended period of time and contextualizes a series of subgoals and actions (find keys, open door, enter) that are resolved at a much faster timescale. This separation of timescales, whether in the continuous or discrete domain, demands a hierarchical (deep) generative model. In neuroscience, one can assume that cortical hierarchies embed this sort of temporal separation of timescales, with slowly evolving states at higher levels and rapidly evolving states at lower levels, and that this recapitulates environmental dynamics, which also evolve at multiple timescales (e.g., during perceptual tasks like speech recognition or reading). In psychology, this sort of model is useful in reproducing hierarchical goal processing (Pezzulo, Rigoli, and Friston 2018) and working memory tasks (Parr and Friston 2017c) of the sort that rely on delay-period activity (Funahashi et al. 1989).

### 6.4.3 Temporal Depth of Inference and Planning

🕒 Planning Framework:
1. Temporal Aspects:
   - [[temporal_models|Temporal Depth]]
   - [[predictive_processing|Future Prediction]]
   - [[decision_making|Action Selection]]

2. Planning Components:
   - [[model_architecture|Model Structure]]
   - [[evidence_accumulation|Evidence Integration]]
   - [[uncertainty_estimation|Uncertainty Handling]]

3. Implementation Considerations:
   - [[performance_optimization|Computational Efficiency]]
   - [[resource_management|Resource Usage]]
   - [[model_complexity|Model Scaling]]

The third design choice concerns the temporal depth of inference. It is important to draw a distinction between two kinds of [[model_architecture|generative model]]:

1. Models with temporal depth that explicitly represent:
   - [[action_selection|Action Consequences]]
   - [[decision_making|Policy Selection]]
   - [[predictive_processing|Future States]]

2. Models without temporal depth that consider:
   - [[sensorimotor_coordination|Present Observations]]
   - [[information_processing|Current State]]
   - [[uncertainty_estimation|Immediate Uncertainty]]

## 6.5 How to Set Up the [[model_architecture|Generative Model]]?

📈 Implementation Framework:
1. Model Components:
   - [[model_architecture|Architecture Design]]
   - [[inference_patterns|Inference Methods]]
   - [[optimization_patterns|Optimization Strategies]]
   - [[active_inference_agent|Agent Implementation]]
   - [[active_inference_loop|Processing Cycles]]

2. Learning Elements:
   - [[learning_mechanisms|Learning Processes]]
   - [[adaptation_mechanisms|Adaptation]]
   - [[stability_plasticity|Stability-Plasticity]]
   - [[belief_updating|Belief Updates]]
   - [[evidence_accumulation|Evidence Integration]]

3. Processing Systems:
   - [[cognitive_modeling_concepts|Cognitive Models]]
   - [[predictive_processing|Predictive Processing]]
   - [[bayesian_inference|Bayesian Methods]]
   - [[probabilistic_inference|Probabilistic Computation]]
   - [[uncertainty_estimation|Uncertainty Handling]]

### 6.5.1 Setting Up the Variables of the [[model_architecture|Generative Model]]

📊 Variable Framework:
1. State Variables:
   - [[information_processing|Hidden States]]
   - [[sensorimotor_coordination|Observations]]
   - [[action_selection|Actions]]
   - [[perceptual_inference|Perceptual States]]
   - [[cognitive_control|Control States]]

2. Prior Beliefs:
   - [[bayesian_inference|Initial States]]
   - [[model_architecture|Model Structure]]
   - [[uncertainty_estimation|Uncertainty Levels]]
   - [[predictive_processing|Predictions]]
   - [[temporal_models|Temporal Dependencies]]

3. Learning Parameters:
   - [[learning_mechanisms|Learning Rates]]
   - [[adaptation_mechanisms|Adaptation Rules]]
   - [[stability_plasticity|Stability Controls]]
   - [[evidence_accumulation|Evidence Weights]]
   - [[belief_updating|Update Rules]]

### 6.5.2 Fixed versus Learned Components

🎓 Learning Framework:
1. Fixed Elements:
   - [[model_architecture|Core Structure]]
   - [[inference_patterns|Basic Rules]]
   - [[optimization_patterns|Key Constraints]]

2. Learned Components:
   - [[learning_mechanisms|Parameter Values]]
   - [[adaptation_mechanisms|Adaptive Features]]
   - [[model_complexity|Structure Updates]]

3. Learning Process:
   - [[evidence_accumulation|Evidence Collection]]
   - [[belief_updating|Belief Revision]]
   - [[performance_optimization|Performance Tuning]]

## 6.6 Setting Up the Generative Process

🛠️ Implementation Aspects:
1. Process Design:
   - [[model_architecture|Model Structure]]
   - [[sensorimotor_coordination|Sensorimotor Interface]]
   - [[environmental_interaction|Environmental Coupling]]

2. Learning Integration:
   - [[learning_mechanisms|Learning Methods]]
   - [[adaptation_mechanisms|Adaptation Strategies]]
   - [[performance_optimization|Performance Tuning]]

3. System Evaluation:
   - [[model_selection|Model Assessment]]
   - [[performance_optimization|Optimization]]
   - [[resource_management|Resource Management]]

## 6.7 Simulating, Visualizing, Analyzing, and Fitting Data Using [[active_inference|Active Inference]]

🔬 Implementation Framework:
1. Simulation Components:
   - [[model_architecture|Model Implementation]]
   - [[performance_optimization|Runtime Optimization]]
   - [[resource_management|Resource Allocation]]
   - [[active_inference_loop|Processing Pipeline]]
   - [[active_inference_agent|Agent Execution]]

2. Analysis Tools:
   - [[evidence_accumulation|Data Analysis]]
   - [[model_selection|Model Comparison]]
   - [[performance_optimization|Performance Metrics]]
   - [[uncertainty_estimation|Uncertainty Analysis]]
   - [[belief_updating|Learning Assessment]]

3. Visualization Methods:
   - [[information_processing|Data Visualization]]
   - [[model_complexity|Structure Visualization]]
   - [[temporal_models|Time Series Analysis]]
   - [[predictive_processing|Prediction Visualization]]
   - [[hierarchical_inference|Hierarchy Display]]

## 6.8 Summary

📚 Implementation Framework:
1. Model Components:
   - [[model_architecture|Architecture Design]]
   - [[inference_patterns|Inference Methods]]
   - [[optimization_patterns|Optimization Strategies]]
   - [[active_inference_agent|Agent Design]]
   - [[active_inference_loop|Process Flow]]

2. Processing Elements:
   - [[cognitive_modeling_concepts|Cognitive Models]]
   - [[predictive_processing|Predictive Processing]]
   - [[bayesian_inference|Bayesian Methods]]
   - [[probabilistic_inference|Probabilistic Methods]]
   - [[perceptual_inference|Perceptual Processing]]

3. Learning Mechanisms:
   - [[learning_mechanisms|Learning Processes]]
   - [[adaptation_mechanisms|Adaptation]]
   - [[stability_plasticity|Stability-Plasticity]]
   - [[evidence_accumulation|Evidence Integration]]
   - [[belief_updating|Belief Revision]]

4. Integration Aspects:
   - [[model_complexity|Complexity Management]]
   - [[performance_optimization|Performance]]
   - [[resource_management|Resource Usage]]
   - [[uncertainty_estimation|Uncertainty Handling]]
   - [[cognitive_control|Control Systems]]

🎯 Key Components:
- [[model_architecture|Model Architecture]]
- [[inference_patterns|Inference Patterns]]
- [[optimization_patterns|Optimization Patterns]]
- [[cognitive_modeling_concepts|Cognitive Modeling]]
- [[active_inference_agent|Agent Implementation]]
- [[active_inference_loop|Processing Pipeline]]

In this chapter, we have outlined the most important design choices that must be made in setting up an [[active_inference|Active Inference]] model. We provided a recipe in four steps and some guidelines to address the usual challenges that model designers face. Of course, it is not necessary to follow the recipe in a rigid manner. Some steps can be inverted (e.g., design the generative process before the [[model_architecture|generative model]]) or combined. But in general, these steps are all required.