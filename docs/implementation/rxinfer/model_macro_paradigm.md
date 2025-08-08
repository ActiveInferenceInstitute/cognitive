---

title: The @model Macro Paradigm in RxInfer

type: documentation

status: stable

created: 2024-03-20

tags:

  - rxinfer

  - julia

  - probabilistic-programming

  - model-specification

semantic_relations:

  - type: implements

    links:

      - [[probabilistic_models]]

      - [[factor_graphs]]

  - type: uses

    links:

      - [[message_passing]]

      - [[variational_inference]]

---

# The @model Macro Paradigm in RxInfer

## Overview

The `@model` macro in [[RxInfer]] represents a powerful [[domain_specific_language|domain-specific language]] for specifying [[probabilistic_models|probabilistic models]] in Julia. It seamlessly translates high-level model descriptions into efficient [[factor_graphs|factor graph]] representations, enabling sophisticated [[bayesian_inference|Bayesian inference]] through [[message_passing|message passing]] algorithms.

## Core Concepts

### Model Specification

The `@model` macro follows a declarative programming paradigm where models are specified through:

1. **[[random_variables|Random Variable]] Declaration**

```julia

@model function simple_model()

    # Prior distribution

    x ~ Normal(mean = 0.0, precision = 0.1)

    # Likelihood function

    y ~ Normal(mean = x, precision = 1.0)

end

```

1. **[[probabilistic_relationships|Probabilistic Relationships]]**

   - Direct dependencies using the `~` operator for stochastic relationships

   - Deterministic transformations using `:=` operator

   - Conditional dependencies through control flow

### Model Arguments and Data Handling

1. **Model Arguments**

```julia

@model function coin_model(y, a, b)

    # Prior with hyperparameters

    θ ~ Beta(a, b)

    # Observations

    y .~ Bernoulli(θ)

end

```

1. **Data Conditioning**

```julia

# Condition on data using | operator

model = coin_model(a = 1.0, b = 1.0) | (y = [true, false, true],)

# Or using dictionary

data = Dict(:y => [true, false, true])

model = coin_model(a = 1.0, b = 1.0) | data

```

1. **Deferred Data Handling**

```julia

# For reactive/streaming inference

model = coin_model() | (y = RxInfer.DeferredDataHandler(),)

```

### Factor Graph Translation

The macro automatically converts model specifications into [[factor_graphs|factor graphs]]:

1. **[[node_types|Node Types]]**

   - [[variable_nodes|Variable nodes]] for random variables

   - [[factor_nodes|Factor nodes]] for probability distributions

   - [[deterministic_nodes|Deterministic nodes]] for transformations

1. **[[graph_structure|Graph Structure]]**

   - [[edges|Edges]] representing probabilistic dependencies

   - [[message_passing_routes|Message passing routes]] for inference

   - [[clique_trees|Clique trees]] for efficient computation

## Advanced Features

### [[model_composition|Model Composition]]

Support for hierarchical model construction:

```julia

@model function component_model(x)

    y ~ Normal(mean = x, precision = 1.0)

    return y

end

@model function composite_model()

    x ~ Gamma(shape = 1.0, rate = 1.0)

    y = component_model(x) # Model composition

end

```

### [[streaming_models|Streaming Models]]

Support for dynamic models with temporal dependencies:

```julia

@model function streaming_model(T)

    # State evolution

    x = Vector{Random}(undef, T)

    x[1] ~ Normal(mean = 0.0, precision = 1.0)

    for t in 2:T

        x[t] ~ Normal(mean = x[t-1], precision = 0.1)

    end

end

```

### [[constraints_specification|Constraints Specification]]

Integration with variational constraints:

```julia

@constraints function model_constraints()

    # Factorized approximation

    q(x, y) = q(x)q(y)

    # Distribution constraints

    q(x) :: ExponentialFamily

    q(y) :: Gaussian

end

```

### [[initialization|Model Initialization]]

Support for initializing messages and marginals:

```julia

@initialization begin

    # Initialize marginal for x

    q(x) = vague(NormalMeanPrecision)

    # Initialize message for y

    μ(y) = vague(NormalMeanPrecision)

end

```

### [[automatic_differentiation|Automatic Differentiation Integration]]

Support for gradient-based inference:

```julia

@model function gradient_model(x)

    # Parameters with gradients

    θ = @param Normal(0.0, 1.0)

    # Forward pass with AD

    y = forward(θ, x)

    # Likelihood with gradient support

    return y ~ Normal(θ, exp(-θ))

end

```

### [[reactive_programming|Reactive Programming Features]]

Integration with reactive programming paradigms:

```julia

@model function reactive_model(data_stream)

    # Create reactive variable

    x = ReactiveVariable()

    # Subscribe to data stream

    subscribe!(data_stream) do value

        # Update model

        update!(x, value)

    end

    # Define reactive dependencies

    y = @reactive begin

        μ = mean(x)

        σ = std(x)

        Normal(μ, σ)

    end

end

```

## Implementation Details

### [[ast_transformation|AST Transformation]]

1. **Variable Detection**

```julia

function extract_variables(expr)

    variables = Set{Symbol}()

    # Walk AST

    MacroTools.postwalk(expr) do x

        if @capture(x, v_ ~ dist_)

            push!(variables, v)

        end

        return x

    end

    return variables

end

```

1. **Factor Graph Construction**

```julia

function construct_factor_graph(variables, dependencies)

    graph = FactorGraph()

    # Add variable nodes

    for var in variables

        add_variable!(graph, var)

    end

    # Add factor nodes

    for (var, deps) in dependencies

        add_factor!(graph, var, deps)

    end

    return graph

end

```

### [[type_system_integration|Type System Integration]]

1. **Distribution Types**

```julia

abstract type Distribution end

struct Normal <: Distribution

    μ::Float64

    σ::Float64

end

struct Factor{T<:Distribution}

    dist::T

    variables::Vector{Symbol}

end

```

1. **Message Types**

```julia

struct Message{T<:Distribution}

    source::Union{Variable,Factor}

    target::Union{Variable,Factor}

    content::T

end

```

## Best Practices

### [[model_design|Model Design]]

1. **[[modularity|Modularity]]**

   - Break complex models into components

   - Reuse common patterns

   - Maintain clear dependencies

1. **[[performance_optimization|Performance Optimization]]**

   - Use appropriate distribution types

   - Leverage sparsity in dependencies

   - Consider memory usage patterns

### [[debugging_strategies|Debugging Strategies]]

1. **[[model_validation|Model Validation]]**

   - Check variable dimensions

   - Verify distribution parameters

   - Test with synthetic data

1. **[[inference_diagnostics|Inference Diagnostics]]**

   - Monitor convergence

   - Check message validity

   - Validate posterior distributions

## Applications

### [[scientific_computing|Scientific Computing]]

1. **[[bayesian_modeling|Bayesian Modeling]]**

   - Parameter estimation

   - Uncertainty quantification

   - Model comparison

1. **[[machine_learning|Machine Learning]]**

   - [[variational_autoencoders|Variational autoencoders]]

   - [[bayesian_neural_networks|Bayesian neural networks]]

   - [[probabilistic_clustering|Probabilistic clustering]]

### [[active_inference|Active Inference]]

Integration with active inference frameworks:

```julia

@model function active_inference_model()

    # Prior preferences

    θ ~ Normal(mean = 0.0, precision = 1.0)

    # Policy selection

    π ~ Categorical(softmax(-θ))

    # Observation model

    y ~ Normal(mean = π, precision = exp(-θ))

end

```

## References

For detailed information, see:

- [[rxinfer_documentation|RxInfer Documentation]]

- [[graphppl_documentation|GraphPPL Documentation]]

- [[message_passing_algorithms|Message Passing Algorithms]]

- [[variational_inference|Variational Inference]]

