---
title: Getting Started with RxInfer
type: documentation
status: stable
created: 2024-03-20
tags:
  - rxinfer
  - tutorial
  - onboarding
semantic_relations:
  - type: implements
    links: 
      - [[probabilistic_models]]
      - [[bayesian_inference]]
      - [[factor_graphs]]
      - [[message_passing]]
  - type: related
    links:
      - [[julia_programming]]
      - [[reactive_programming]]
      - [[variational_inference]]
---

# Getting Started with RxInfer

## Overview

RxInfer is a Julia package for [[bayesian_inference|Bayesian Inference]] that combines the power of [[factor_graphs|factor graphs]] with [[reactive_programming|reactive programming]] principles. This guide will help you get started with RxInfer and understand its core concepts.

```mermaid
graph TD
    A[Model Definition] --> B[Factor Graph]
    B --> C[Message Passing]
    C --> D[Inference Results]
    style A fill:#f9f,stroke:#333
    style B fill:#bbf,stroke:#333
    style C fill:#bfb,stroke:#333
    style D fill:#fbb,stroke:#333
```

### Architecture Overview

```mermaid
graph TB
    subgraph Model Layer
        M1[Model Definition]
        M2[Constraints]
        M3[Initialization]
    end
    subgraph Inference Layer
        I1[Factor Graph]
        I2[Message Passing]
        I3[Variational Updates]
    end
    subgraph Data Layer
        D1[Static Data]
        D2[Streaming Data]
        D3[Reactive Updates]
    end
    M1 --> I1
    M2 --> I2
    M3 --> I3
    D1 --> I2
    D2 --> I3
    D3 --> I3
    style M1 fill:#f9f
    style M2 fill:#f9f
    style M3 fill:#f9f
    style I1 fill:#bbf
    style I2 fill:#bbf
    style I3 fill:#bbf
    style D1 fill:#bfb
    style D2 fill:#bfb
    style D3 fill:#bfb
```

## Prerequisites

Before diving into RxInfer, you should have:

1. [[julia_installation|Julia]] installed (version 1.6 or higher)
2. Basic understanding of:
   - [[probability_theory|Probability Theory]]
   - [[bayesian_statistics|Bayesian Statistics]]
   - [[julia_programming|Julia Programming]]

### Conceptual Prerequisites

```mermaid
mindmap
  root((RxInfer))
    Probability
      Distributions
      Random Variables
      Conditional Probability
    Julia
      Types
      Functions
      Broadcasting
    Bayesian Stats
      Prior
      Likelihood
      Posterior
    Factor Graphs
      Nodes
      Edges
      Messages
```

## Installation

Install RxInfer through Julia's package manager:

```julia
using Pkg
Pkg.add("RxInfer")
```

Or in package mode (`]`):

```julia
] add RxInfer
```

## Basic Concepts

RxInfer is built around three main concepts:

1. **[[model_specification|Model Specification]]**: Using the `@model` macro to define probabilistic models
2. **[[inference_execution|Inference Execution]]**: Running inference on your models with static or streaming data
3. **[[results_analysis|Results Analysis]]**: Analyzing posterior distributions and predictions

### Core Components Interaction

```mermaid
graph LR
    subgraph Model
        A1[Variables] --> A2[Distributions]
        A2 --> A3[Dependencies]
    end
    subgraph Inference
        B1[Factor Graph] --> B2[Message Passing]
        B2 --> B3[Convergence]
    end
    subgraph Results
        C1[Posteriors] --> C2[Predictions]
        C2 --> C3[Diagnostics]
    end
    Model --> Inference
    Inference --> Results
    style A1 fill:#f9f
    style A2 fill:#f9f
    style A3 fill:#f9f
    style B1 fill:#bbf
    style B2 fill:#bbf
    style B3 fill:#bbf
    style C1 fill:#bfb
    style C2 fill:#bfb
    style C3 fill:#bfb
```

## Your First Model

Let's create a simple coin flipping model to demonstrate RxInfer's basics:

```julia
using RxInfer, Distributions

# Define a model for coin flips
@model function coin_model(y)
    # Prior belief about coin fairness
    θ ~ Beta(1, 1)  # Uniform prior
    
    # Likelihood of observations
    y .~ Bernoulli(θ)
end

# Generate some synthetic data
true_bias = 0.7
data = rand(Bernoulli(true_bias), 100)

# Run inference
result = infer(
    model = coin_model(),
    data = (y = data,)
)

# Access the posterior distribution
posterior_θ = result.posteriors[:θ]
```

### Visualizing Results

```julia
using Plots

# Plot prior and posterior distributions
plot(x -> pdf(Beta(1, 1), x), 0, 1, label="Prior")
plot!(x -> pdf(posterior_θ, x), 0, 1, label="Posterior")
vline!([true_bias], label="True Value")
```

## Core Features

### 1. Model Specification

RxInfer uses a declarative syntax for model specification through the [[model_macro|@model]] macro:

```julia
@model function linear_regression(x, y)
    # Priors
    α ~ Normal(0, 10)    # Intercept
    β ~ Normal(0, 10)    # Slope
    σ ~ Gamma(1, 1)      # Noise
    
    # Likelihood
    y .~ Normal(α .+ β .* x, σ)
end
```

### 2. Data Conditioning

You can condition models on data in several ways:

```julia
# Using named tuple
model | (x = [1,2,3], y = [2,4,6])

# Using dictionary
model | Dict(:x => [1,2,3], :y => [2,4,6])

# Using deferred data for streaming
model | (x = RxInfer.DeferredDataHandler(),)
```

### 3. Inference Types

RxInfer supports multiple [[inference_paradigms|inference paradigms]]:

```julia
# Static inference
result = infer(
    model = my_model(),
    data = my_data
)

# Streaming inference
result = infer(
    model = my_model(),
    datastream = my_stream,
    autoupdates = my_updates
)
```

## Advanced Features

### Component Relationships

```mermaid
graph TB
    subgraph Models
        M1[Basic Models]
        M2[Composite Models]
        M3[Streaming Models]
    end
    subgraph Features
        F1[Constraints]
        F2[Custom Distributions]
        F3[Message Rules]
    end
    subgraph Integration
        I1[Data Sources]
        I2[External Systems]
        I3[Visualization]
    end
    M1 --> F1
    M2 --> F2
    M3 --> F3
    F1 --> I1
    F2 --> I2
    F3 --> I3
    style M1 fill:#f9f
    style M2 fill:#f9f
    style M3 fill:#f9f
    style F1 fill:#bbf
    style F2 fill:#bbf
    style F3 fill:#bbf
    style I1 fill:#bfb
    style I2 fill:#bfb
    style I3 fill:#bfb
```

### 1. Working with Constraints

For more complex models, you might need to specify [[variational_constraints|variational constraints]]:

```julia
# Define constraints
@constraints function model_constraints()
    # Mean-field factorization
    q(x, y) = q(x)q(y)
    
    # Distribution families
    q(x) :: NormalMeanPrecision
    q(y) :: GammaShapeRate
end

# Use constraints in inference
result = infer(
    model = my_model(),
    data = my_data,
    constraints = model_constraints()
)
```

### 2. Model Initialization

Some models require initialization of [[messages_and_marginals|messages or marginals]]:

```julia
# Define initialization
@initialization begin
    # Initialize marginals
    q(x) = vague(NormalMeanPrecision)
    
    # Initialize messages
    μ(y) = vague(NormalMeanPrecision)
end

# Use initialization in inference
result = infer(
    model = my_model(),
    data = my_data,
    initialization = my_init
)
```

### 3. Custom Distributions

Create [[custom_distributions|custom distributions]] for specialized models:

```julia
struct MyDistribution <: Distribution
    params::Vector{Float64}
end

# Define required methods
Distributions.pdf(d::MyDistribution, x::Real) = # ...
Distributions.logpdf(d::MyDistribution, x::Real) = # ...
```

## Common Patterns and Best Practices

### Design Pattern Relationships

```mermaid
graph LR
    subgraph Patterns
        P1[Broadcasting]
        P2[Composition]
        P3[Data Handling]
    end
    subgraph Benefits
        B1[Performance]
        B2[Modularity]
        B3[Flexibility]
    end
    P1 --> B1
    P2 --> B2
    P3 --> B3
    style P1 fill:#f9f
    style P2 fill:#f9f
    style P3 fill:#f9f
    style B1 fill:#bbf
    style B2 fill:#bbf
    style B3 fill:#bfb
```

## Interactive Examples

### Example Categories

```mermaid
mindmap
  root((Examples))
    Basic
      Coin Flip
      Linear Regression
      Classification
    Intermediate
      Time Series
      Missing Data
      Hierarchical
    Advanced
      Custom Rules
      Streaming
      Neural Nets
```

Visit our [[rxinfer_examples|interactive examples]] for hands-on tutorials:

1. [[basic_inference|Basic Inference Tutorial]]
2. [[streaming_data|Streaming Data Tutorial]]
3. [[custom_models|Custom Models Tutorial]]

## Troubleshooting

Common issues and their solutions:

1. **Installation Problems**
   - Check Julia version compatibility
   - Verify package dependencies
   - [[installation_guide|Installation Guide]]

2. **Runtime Errors**
   - [[type_errors|Type Errors]]
   - [[dimension_mismatch|Dimension Mismatch]]
   - [[convergence_issues|Convergence Issues]]

3. **Performance Issues**
   - [[performance_guide|Performance Guide]]
   - [[optimization_tips|Optimization Tips]]
   - [[profiling_guide|Profiling Guide]]

## Next Steps

1. Explore the [[model_macro_paradigm|@model Macro Paradigm]]
2. Learn about [[streaming_inference|Streaming Inference]]
3. Understand [[variational_inference|Variational Inference]]
4. Study [[factor_graphs|Factor Graphs]]
5. Practice with [[example_gallery|Example Gallery]]

## Community and Support

- [[github_repository|GitHub Repository]]
- [[documentation|Documentation]]
- [[discourse_forum|Discourse Forum]]
- [[slack_channel|Slack Channel]]
- [[stack_overflow|Stack Overflow Tag]]

## References

- [[rxinfer_documentation|RxInfer Documentation]]
- [[graphppl_documentation|GraphPPL Documentation]]
- [[message_passing_algorithms|Message Passing Algorithms]]
- [[variational_inference_theory|Variational Inference Theory]]
- [[reactive_programming_concepts|Reactive Programming Concepts]]
- [[julia_documentation|Julia Documentation]]

## Quick Reference

### Common Operations

```mermaid
graph TD
    subgraph Setup
        S1[Install] --> S2[Import]
        S2 --> S3[Configure]
    end
    subgraph Usage
        U1[Define Model] --> U2[Prepare Data]
        U2 --> U3[Run Inference]
    end
    subgraph Analysis
        A1[Get Results] --> A2[Visualize]
        A2 --> A3[Validate]
    end
    Setup --> Usage
    Usage --> Analysis
    style S1 fill:#f9f
    style S2 fill:#f9f
    style S3 fill:#f9f
    style U1 fill:#bbf
    style U2 fill:#bbf
    style U3 fill:#bbf
    style A1 fill:#bfb
    style A2 fill:#bfb
    style A3 fill:#bfb
``` 