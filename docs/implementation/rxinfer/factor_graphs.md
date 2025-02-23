---
title: Factor Graphs in RxInfer
type: documentation
status: stable
created: 2024-03-20
tags:
  - rxinfer
  - factor-graphs
  - probabilistic-models
semantic_relations:
  - type: implements
    links: 
      - [[probabilistic_models]]
      - [[graphical_models]]
  - type: related
    links:
      - [[message_passing]]
      - [[model_specification]]
---

# Factor Graphs in RxInfer

## Overview

Factor graphs in RxInfer provide a powerful graphical representation of probabilistic models. They decompose complex probability distributions into simpler factors, enabling efficient inference through [[message_passing|message passing]].

```mermaid
graph TD
    subgraph Factor Graph Structure
        V1[Variable Node] --- F1[Factor Node]
        F1 --- V2[Variable Node]
        V2 --- F2[Factor Node]
        F2 --- V3[Variable Node]
    end
    style V1 fill:#f9f,stroke:#333
    style V2 fill:#f9f,stroke:#333
    style V3 fill:#f9f,stroke:#333
    style F1 fill:#bbf,stroke:#333
    style F2 fill:#bbf,stroke:#333
```

## Core Components

### 1. Variable Nodes

Represent random variables in your model:

```julia
@model function example_model()
    # Variable nodes are created for:
    x ~ Normal(0, 1)     # Latent variable
    y ~ Normal(x, 1)     # Observable variable
end
```

### 2. Factor Nodes

Represent probability distributions or constraints:

```mermaid
graph LR
    subgraph Factor Types
        F1[Prior Factors]
        F2[Likelihood Factors]
        F3[Constraint Factors]
    end
    subgraph Examples
        E1[Normal(0,1)]
        E2[y|x ~ Normal(x,1)]
        E3[x > 0]
    end
    F1 --> E1
    F2 --> E2
    F3 --> E3
    style F1 fill:#f9f
    style F2 fill:#f9f
    style F3 fill:#f9f
    style E1 fill:#bbf
    style E2 fill:#bbf
    style E3 fill:#bbf
```

### 3. Edges

Connect variables and factors:

```mermaid
graph LR
    subgraph Edge Types
        E1[Variable-Factor]
        E2[Message Forward]
        E3[Message Backward]
    end
    V1[x] --- F1[p(x)]
    F1 --- V2[y]
    style V1 fill:#f9f
    style V2 fill:#f9f
    style F1 fill:#bbf
    style E1 fill:#bfb
    style E2 fill:#bfb
    style E3 fill:#bfb
```

## Graph Construction

### 1. Automatic Construction

RxInfer automatically constructs factor graphs from model definitions:

```julia
@model function linear_model(x, y)
    # Prior on parameters
    α ~ Normal(0, 10)    # Creates variable node α
    β ~ Normal(0, 10)    # Creates variable node β
    
    # Likelihood factor
    y .~ Normal(α .+ β .* x, 1)  # Creates factor connecting α, β, and y
end
```

### Graph Construction Process

```mermaid
graph TD
    subgraph Model Definition
        M1[Variables]
        M2[Distributions]
        M3[Dependencies]
    end
    subgraph Graph Construction
        G1[Create Nodes]
        G2[Add Factors]
        G3[Connect Edges]
    end
    subgraph Result
        R1[Factor Graph]
        R2[Message Rules]
    end
    M1 --> G1
    M2 --> G2
    M3 --> G3
    G1 --> R1
    G2 --> R1
    G3 --> R1
    G3 --> R2
    style M1 fill:#f9f
    style M2 fill:#f9f
    style M3 fill:#f9f
    style G1 fill:#bbf
    style G2 fill:#bbf
    style G3 fill:#bbf
    style R1 fill:#bfb
    style R2 fill:#bfb
```

## Factor Types

### 1. Distribution Factors

Represent probability distributions:

```julia
# Prior factors
x ~ Normal(0, 1)
θ ~ Beta(1, 1)

# Likelihood factors
y ~ Normal(x, 1)
z ~ Bernoulli(θ)
```

### 2. Deterministic Factors

Represent deterministic relationships:

```julia
@model function deterministic_example()
    x ~ Normal(0, 1)
    y = 2 * x           # Deterministic factor
    z ~ Normal(y, 1)    # Uses the deterministic relationship
end
```

### 3. Constraint Factors

Impose constraints on variables:

```julia
@constraints function model_constraints()
    # Factorization constraints
    q(x, y) = q(x)q(y)
    
    # Distribution family constraints
    q(x) :: NormalMeanPrecision
end
```

## Graph Patterns

### 1. Chain Structure

```mermaid
graph LR
    X1[x₁] --- F1[f₁] --- X2[x₂] --- F2[f₂] --- X3[x₃]
    style X1 fill:#f9f
    style X2 fill:#f9f
    style X3 fill:#f9f
    style F1 fill:#bbf
    style F2 fill:#bbf
```

```julia
@model function chain_model()
    x₁ ~ Normal(0, 1)
    x₂ ~ Normal(x₁, 1)
    x₃ ~ Normal(x₂, 1)
end
```

### 2. Star Structure

```mermaid
graph TD
    C[Center] --- F1[f₁] --- X1[x₁]
    C --- F2[f₂] --- X2[x₂]
    C --- F3[f₃] --- X3[x₃]
    style C fill:#f9f
    style X1 fill:#f9f
    style X2 fill:#f9f
    style X3 fill:#f9f
    style F1 fill:#bbf
    style F2 fill:#bbf
    style F3 fill:#bbf
```

```julia
@model function star_model()
    center ~ Normal(0, 1)
    x₁ ~ Normal(center, 1)
    x₂ ~ Normal(center, 1)
    x₃ ~ Normal(center, 1)
end
```

### 3. Grid Structure

```mermaid
graph TD
    X11[x₁₁] --- F12[f₁₂] --- X12[x₁₂]
    X11 --- F21[f₂₁] --- X21[x₂₁]
    X12 --- F22[f₂₂] --- X22[x₂₂]
    X21 --- F22
    style X11 fill:#f9f
    style X12 fill:#f9f
    style X21 fill:#f9f
    style X22 fill:#f9f
    style F12 fill:#bbf
    style F21 fill:#bbf
    style F22 fill:#bbf
```

## Advanced Topics

### 1. Graph Optimization

Techniques for efficient graph structure:

- Node elimination ordering
- Factor grouping
- Edge reduction

### 2. Custom Factor Types

Creating custom factors:

```julia
struct CustomFactor <: AbstractFactor
    variables::Vector{Variable}
    parameters::Vector{Float64}
end

# Define message computation rules
function compute_message(f::CustomFactor, msg_in)
    # Custom message computation logic
end
```

### 3. Graph Visualization

Visualizing factor graphs:

```julia
using GraphViz

# Visualize factor graph
function visualize_graph(model)
    graph = to_graphviz(model)
    draw(PNG("factor_graph.png"), graph)
end
```

## Best Practices

### 1. Graph Design

- Keep graph structure sparse when possible
- Group related factors
- Consider message passing efficiency

### 2. Performance Optimization

```mermaid
mindmap
  root((Optimization))
    Graph Structure
      Sparsity
      Node Ordering
      Factor Grouping
    Computation
      Message Scheduling
      Parallel Updates
      Caching
    Memory
      Variable Elimination
      Message Storage
      Graph Pruning
```

### 3. Debugging

- Visualize graph structure
- Check factor connections
- Monitor message convergence

## References

- [[graphical_models|Graphical Models]]
- [[message_passing|Message Passing]]
- [[variational_inference|Variational Inference]]
- [[model_specification|Model Specification]] 