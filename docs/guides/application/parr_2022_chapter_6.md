# Chapter 6: A Recipe for Designing Active Inference Models

> "Give me six hours to chop down a tree and I will spend the first four sharpening the axe."
> — Abraham Lincoln

## 6.1 Introduction

This chapter provides a four-step recipe to construct an Active Inference model, discussing the most important design choices one has to make to realize a model and providing some guidelines for those choices. It serves as an introduction to the second part of the book, which will illustrate several specific computational models using Active Inference and their applications in a variety of cognitive domains.

🎯 Core Concepts:

- Active Inference as a normative approach

- Behavioral processes

- Cognitive processes

- Neural processes

- Free Energy minimization

- Generative Models

- Probabilistic Inference

- Belief Updating

- Uncertainty Estimation

The generative modeling approach is used in several disciplines for:

- Cognitive Models

- Statistical Modeling

- Experimental Data Analysis

- Machine Learning

- Perceptual Models

- Knowledge Representation

📋 Design Methodology Examples:

1. Predictive Coding: Perception as inference about sensations

1. Discrete Time Models: Planning as inference about actions

1. Spatial Navigation: Planning with spatial variables

1. Visual Search: Planning with saccades

1. Cross-Modal Integration: Multisensory inference

1. Temporal Integration: Sequential processing

🔍 Key Design Questions:

1. Which system are we modeling?

1. What is the most appropriate form for the [[knowledge_base/cognitive/generative_model]]?

1. How to set up the [[knowledge_base/cognitive/generative_model]]?

1. How to set up the generative process?

As [[knowledge_base/cognitive/active_inference]] is a normative approach, it tries to explain as much as possible about behavior, cognitive, and neural processes from first principles. Consistently, the design philosophy of [[knowledge_base/cognitive/active_inference]] is top-down. Unlike many other approaches to computational neuroscience, the challenge is not to emulate a brain, piece by piece, but to find the generative model that describes the problem the brain is trying to solve. Once the problem is appropriately formalized in terms of a generative model, the solution to the problem emerges under [[knowledge_base/cognitive/active_inference]]—with accompanying predictions about brains and minds. In other words, the generative model provides a complete description of a system of interest. The resulting behavior, inference, and neural dynamics can all be derived from a model by minimizing [[docs/implementation/rxinfer/free_energy]].

The generative modeling approach is used in several disciplines for the realization of cognitive models, statistical modeling, experimental data analysis, and machine learning (Hinton 2007b; Lee and Wagenmakers 2014; Pezzulo, Rigoli, and Friston 2015; Allen et al. 2019; Foster 2019). Here, we are primarily interested in designing generative models that engender cognitive processes of interest. We have seen this design methodology in previous chapters. For example, using a generative model for predictive coding, perception was cast as an inference about the most likely cause of sensations; using a generative model that evolves in discrete time, planning was cast as an inference about the most likely course of action. Depending on the problem of interest (e.g., planning during spatial navigation or planning saccades during visual search), one can adapt the form of these generative models to equip them with different structures (e.g., shallow or hierarchical) and variables (e.g., beliefs about allocentric or egocentric spatial locations). Importantly, Active Inference may take on many different guises under different assumptions about the form of the generative model being optimized. For example, assumptions about models that evolve in discrete or continuous time influence the form of the message passing (see chapter 4). This implies that the choice of a generative model corresponds to specific predictions about both behavior and neurobiology.

This flexibility is useful as it allows us to use the same language to describe processes in multiple domains. However, it can also be confusing from a practical perspective, as there are a number of choices that must be made to find the appropriate level of description for the system of interest. In the second part of this book, we will try to resolve this confusion through a series of illustrative examples of Active Inference in silico. This chapter introduces a general recipe for the design of Active Inference models, highlighting some of the key design choices, distinctions, and dichotomies that will appear in the numerical analysis of computational models described in subsequent chapters.

## 6.2 Designing an Active Inference Model: A Recipe in Four Steps

Designing an Active Inference model requires four foundational steps, each resolving a specific design question:

1. Which system are we modeling? The first choice to make is always the system of interest. This may not be as simple as it seems; it rests on the identification of the boundaries (i.e., Markov blanket) of that system. What counts as an Active Inference agent (generative model), what counts as the external environment (generative process), and what is the interface (sensory data and actions) between them?

1. What is the most appropriate form for the [[knowledge_base/cognitive/generative_model]]? The first of the next three practical challenges is deciding whether it is appropriate to think of a process more in terms of categorical (discrete) inferences or continuous inferences, motivating the choice between discrete or continuous-time implementations (or a hybrid) of Active Inference. Then we need to select the most appropriate hierarchical depth, motivating the choice between shallow versus deep models. Finally, we need to consider whether it is necessary to endow generative models with temporal depth and the ability to predict action-contingent observations to support planning.

1. How to set up the [[knowledge_base/cognitive/generative_model]]? What are the generative model's most appropriate variables and priors? Which parts are fixed and what must be learned? We emphasize the importance of choosing the right sort of variables and prior beliefs; furthermore, we emphasize a separation in timescales between the (faster) update of state variables that occurs during inference and the (slower) update of model parameters that occurs during learning.

1. How to set up the generative process? What are the elements of the generative process (and how do they differ from the generative model)? These four steps (in most cases) suffice to design an Active Inference model. Once completed, the behavior of the system is determined by the standard schemes of Active Inference: the descent of the active and internal states on the free energy functional associated with the model. From a more practical perspective, once one has specified the generative model and generative process, one can use standard Active Inference software routines to obtain numerical results, as well as to perform data visualization, analysis, and fitting (e.g., model-based data analysis). In what follows, we will review the four design choices in order.

## 6.3 What System Are We Modeling?

🌐 System Boundaries:

- Neural Systems

- Cellular Systems

- Biological Systems

- Emergent Systems

🔄 Interaction Components:

- Sensory Receptors

- Motor Effectors

- System Boundaries

- Internal States

📊 Variable Types:

- Sensory States

- Active States

- Information States

- Uncertainty States

A useful first step in applying the formalism of Active Inference is to identify the boundaries of the system of interest because we are interested in characterizing the interaction between what is internal to a system and the external world via sensory receptors and effectors (e.g., muscles or glands). As discussed in chapter 3, a formal way to characterize the distinction between internal states of a system and external variables (and intermediate variables that mediate their interactions) is in terms of a [[knowledge_base/mathematics/markov_blanket]] (Pearl 1988).

The [[knowledge_base/mathematics/markov_blanket]] may be subdivided into two sorts of variables (Friston 2013):

1. Those that mediate the influence of the external world on internal states (sensory states)

1. Those that mediate the influence of internal states on the external world (active states)

## 6.4 What Is the Most Appropriate Form for the Generative Model?

🔢 Variable Categories:

1. Discrete Variables:

   - Object Identities

   - Action Plans

   - Pattern Categories

   - Semantic Concepts

   - Memory States

1. Continuous Variables:

   - Position/Velocity

   - Biological Motion

   - Muscle Length

   - Luminance

   - Continuous Dynamics

1. Processing Considerations:

   - Temporal Processing

   - Spatial Processing

   - Pattern Processing

   - Information Flow

   - Control Mechanisms

### 6.4.1 Discrete or Continuous Variables (or Both)?

The first design choice is to consider whether generative models that use discrete or continuous variables are more appropriate. The distinction between these approaches has important implications for:

- Temporal Processing

- Neural Implementation

- Message Passing

- Hierarchical Organization

- Processing Cycles

### 6.4.2 Timescales of Inference: Shallow versus Hierarchical Models

⏱️ Hierarchical Organization:

1. Temporal Scales:

   - Fast Binding

   - Medium-term Dynamics

   - Sustained Processing

1. Processing Levels:

   - Hierarchical Processing

   - Predictive Hierarchies

   - Neural Hierarchies

1. Integration Mechanisms:

   - Cross-modal Integration

   - Feature Binding

   - Temporal Integration

The second design choice concerns the timescales of [[knowledge_base/cognitive/active_inference]]. One can select either (shallow) [[knowledge_base/cognitive/generative_model|generative models]], in which all the variables evolve at the same timescale, or (hierarchical or deep) models, which include variables that evolve at different timescales: slower for higher levels and faster for lower levels.

While many simple cognitive models only require shallow models, these are not sufficient when there is a clear separation of timescales between different aspects of a cognitive process of interest. One example of this is in language processing, in which short sequences of phonemes are contextualized by the word that is spoken and short sequences of words are contextualized by the current sentence. Crucially, the duration of the word transcends that of any one phoneme in the sequence and the duration of the sentence transcends that of any one word in the sequence. Hence, to model language processing, one can consider a hierarchical model in which sentences, words, and phonemes appear at different (higher to lower) hierarchical levels and evolve over (slower to faster) timescales that are approximately independent of one another. This is only an approximate separation, as levels must influence each other (e.g., the sentence influences the next words in the sequence; the word influences the next phonemes in the sequence). However, this does not mean we need to attempt to model the entire brain to develop meaningful simulations of a single level. For example, if we wanted to focus on word processing, we could address some aspects without having to deal with phoneme processing. This means we can treat input from parts of the brain drawing inferences about phonemes as providing observations from the perspective of word-processing areas. Phrasing this in terms of a Markov blanket, this typically means we treat the inferences performed by lower levels of a model as part of the sensory states of the blanket. This means we can summarize the inferences performed at the timescale of interest without having to specify the details of lower-level (faster) inferential processes—and this hierarchical factorization entails great computational benefits.

Another example is in the domain of intentional action selection, where the same goal (enter your apartment) can be active for an extended period of time and contextualizes a series of subgoals and actions (find keys, open door, enter) that are resolved at a much faster timescale. This separation of timescales, whether in the continuous or discrete domain, demands a hierarchical (deep) generative model. In neuroscience, one can assume that cortical hierarchies embed this sort of temporal separation of timescales, with slowly evolving states at higher levels and rapidly evolving states at lower levels, and that this recapitulates environmental dynamics, which also evolve at multiple timescales (e.g., during perceptual tasks like speech recognition or reading). In psychology, this sort of model is useful in reproducing hierarchical goal processing (Pezzulo, Rigoli, and Friston 2018) and working memory tasks (Parr and Friston 2017c) of the sort that rely on delay-period activity (Funahashi et al. 1989).

### 6.4.3 Temporal Depth of Inference and Planning

🕒 Planning Framework:

1. Temporal Aspects:

   - Temporal Depth

   - Future Prediction

   - Action Selection

1. Planning Components:

   - Model Structure

   - Evidence Integration

   - Uncertainty Handling

1. Implementation Considerations:

   - Computational Efficiency

   - Resource Usage

   - Model Scaling

The third design choice concerns the temporal depth of inference. It is important to draw a distinction between two kinds of generative model:

1. Models with temporal depth that explicitly represent:

   - Action Consequences

   - Policy Selection

   - Future States

1. Models without temporal depth that consider:

   - Present Observations

   - Current State

   - Immediate Uncertainty

## 6.5 How to Set Up the Generative Model?

📈 Implementation Framework:

1. Model Components:

   - Architecture Design

   - Inference Methods

   - Optimization Strategies

   - Agent Implementation

   - Processing Cycles

1. Learning Elements:

   - Learning Processes

   - Adaptation

   - Stability-Plasticity

   - Belief Updates

   - Evidence Integration

1. Processing Systems:

   - Cognitive Models

   - Predictive Processing

   - Bayesian Methods

   - Probabilistic Computation

   - Uncertainty Handling

### 6.5.1 Setting Up the Variables of the Generative Model

📊 Variable Framework:

1. State Variables:

   - Hidden States

   - Observations

   - Actions

   - Perceptual States

   - Control States

1. Prior Beliefs:

   - Initial States

   - Model Structure

   - Uncertainty Levels

   - Predictions

   - Temporal Dependencies

1. Learning Parameters:

   - Learning Rates

   - Adaptation Rules

   - Stability Controls

   - Evidence Weights

   - Update Rules

### 6.5.2 Fixed versus Learned Components

🎓 Learning Framework:

1. Fixed Elements:

   - Core Structure

   - Basic Rules

   - Key Constraints

1. Learned Components:

   - Parameter Values

   - Adaptive Features

   - Structure Updates

1. Learning Process:

   - Evidence Collection

   - Belief Revision

   - Performance Tuning

## 6.6 Setting Up the Generative Process

🛠️ Implementation Aspects:

1. Process Design:

   - Model Structure

   - Sensorimotor Interface

   - Environmental Coupling

1. Learning Integration:

   - Learning Methods

   - Adaptation Strategies

   - Performance Tuning

1. System Evaluation:

   - Model Assessment

   - Optimization

   - Resource Management

## 6.7 Simulating, Visualizing, Analyzing, and Fitting Data Using Active Inference

🔬 Implementation Framework:

1. Simulation Components:

   - Model Implementation

   - Runtime Optimization

   - Resource Allocation

   - Processing Pipeline

   - Agent Execution

1. Analysis Tools:

   - Data Analysis

   - Model Comparison

   - Performance Metrics

   - Uncertainty Analysis

   - Learning Assessment

1. Visualization Methods:

   - Data Visualization

   - Structure Visualization

   - Time Series Analysis

   - Prediction Visualization

   - Hierarchy Display

## 6.8 Summary

📚 Implementation Framework:

1. Model Components:

   - Architecture Design

   - Inference Methods

   - Optimization Strategies

   - Agent Design

   - Process Flow

1. Processing Elements:

   - Cognitive Models

   - Predictive Processing

   - Bayesian Methods

   - Probabilistic Methods

   - Perceptual Processing

1. Learning Mechanisms:

   - Learning Processes

   - Adaptation

   - Stability-Plasticity

   - Evidence Integration

   - Belief Revision

1. Integration Aspects:

   - Complexity Management

   - Performance

   - Resource Usage

   - Uncertainty Handling

   - Control Systems

🎯 Key Components:

- Model Architecture

- Inference Patterns

- Optimization Patterns

- Cognitive Modeling

- Agent Implementation

- Processing Pipeline

In this chapter, we have outlined the most important design choices that must be made in setting up an Active Inference model. We provided a recipe in four steps and some guidelines to address the usual challenges that model designers face. Of course, it is not necessary to follow the recipe in a rigid manner. Some steps can be inverted (e.g., design the generative process before the generative model) or combined. But in general, these steps are all required.

