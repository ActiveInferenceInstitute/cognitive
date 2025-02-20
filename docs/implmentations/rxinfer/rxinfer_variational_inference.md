---
title: Variational Inference in RxInfer
type: documentation
status: stable
created: 2024-03-20
tags:
  - rxinfer
  - variational-inference
  - probabilistic-programming
semantic_relations:
  - type: implements
    links: 
      - [[variational_inference]]
      - [[message_passing]]
---

# Variational Inference in RxInfer

## Overview

RxInfer implements [[variational_inference|Variational Inference]] (VI) through [[message_passing|message passing]] on [[factor_graphs|factor graphs]]. This guide explains how to use VI in RxInfer and understand its key concepts.

## Core Concepts

### 1. Variational Approximation

VI approximates complex posterior distributions with simpler ones:

```julia
@model function complex_model(y)
    # True posterior would be intractable
    μ ~ Normal(0, 10)
    τ ~ Gamma(1, 1)
    y .~ Normal(μ, τ)
end

# Approximate with factorized distributions
@constraints function factorized()
    q(μ, τ) = q(μ)q(τ)  # Mean-field approximation
end
```

### 2. Distribution Constraints

Specify the form of approximate posteriors:

```julia
@constraints function distribution_families()
    # Specify distribution families
    q(μ) :: NormalMeanPrecision
    q(τ) :: GammaShapeRate
    
    # Factorization structure
    q(μ, τ) = q(μ)q(τ)
end
```

### 3. Initialization

Initialize variational parameters:

```julia
@initialization function init_values()
    # Initialize with vague distributions
    q(μ) = vague(NormalMeanPrecision)
    q(τ) = vague(GammaShapeRate)
end
```

## Practical Examples

### 1. Gaussian Mixture Model

```julia
using RxInfer, Distributions

# Model specification
@model function mixture_model(y)
    # Mixture weights
    w ~ Dirichlet([1.0, 1.0])
    
    # Component parameters
    μ₁ ~ Normal(0, 10)
    μ₂ ~ Normal(0, 10)
    
    # Component assignments
    z .~ Categorical(w)
    
    # Observations
    y .~ Normal.(z .== 1 ? μ₁ : μ₂, 1.0)
end

# Variational constraints
@constraints function mixture_constraints()
    # Mean-field factorization
    q(w, μ₁, μ₂, z) = q(w)q(μ₁)q(μ₂)q(z)
    
    # Distribution families
    q(w) :: Dirichlet
    q(μ₁) :: NormalMeanPrecision
    q(μ₂) :: NormalMeanPrecision
    q(z) :: Categorical
end

# Initialization
@initialization function mixture_init()
    q(w) = Dirichlet([1.0, 1.0])
    q(μ₁) = NormalMeanPrecision(0.0, 0.1)
    q(μ₂) = NormalMeanPrecision(0.0, 0.1)
end

# Run inference
result = infer(
    model = mixture_model(),
    data = (y = data,),
    constraints = mixture_constraints(),
    initialization = mixture_init(),
    iterations = 100
)
```

### 2. Hierarchical Model

```julia
@model function hierarchical(y, groups)
    # Hyperparameters
    α ~ Gamma(1, 1)
    β ~ Gamma(1, 1)
    
    # Group-level parameters
    θ = Vector{Random}(undef, length(groups))
    for g in 1:length(groups)
        θ[g] ~ Normal(0, 1/sqrt(α))
    end
    
    # Observations
    y .~ Normal.(θ[groups], 1/sqrt(β))
end

@constraints function hierarchical_constraints()
    # Structured mean-field
    q(α, β, θ) = q(α)q(β)q(θ)
    
    # Distribution families
    q(α) :: GammaShapeRate
    q(β) :: GammaShapeRate
    q(θ) :: MultivariateNormalMeanPrecision
end
```

## Advanced Features

### 1. Non-conjugate Inference

Handle non-conjugate models using projection:

```julia
using ExponentialFamilyProjection

@model function nonconjugate_model(y)
    p ~ Beta(1, 1)
    y .~ Normal(p, 1)  # Non-conjugate likelihood
end

@constraints function projection_constraints()
    q(p) :: ProjectedTo(Beta)
end
```

### 2. Structured Approximations

Create structured variational approximations:

```julia
@model function time_series(y)
    x = Vector{Random}(undef, length(y))
    x[1] ~ Normal(0, 1)
    
    for t in 2:length(y)
        x[t] ~ Normal(x[t-1], 1)
    end
    
    y .~ Normal.(x, 1)
end

@constraints function structured_constraints()
    # Preserve temporal structure
    q(x) :: MultivariateNormalMeanPrecision
end
```

### 3. Custom Distribution Families

Define custom variational families:

```julia
struct MyDistribution <: ExponentialFamily
    params::Vector{Float64}
end

@constraints function custom_constraints()
    q(x) :: MyDistribution
end
```

## Best Practices

### 1. Initialization Strategies

Choose appropriate initialization:

```julia
# Conservative initialization
@initialization function safe_init()
    q(x) = vague(NormalMeanPrecision)
end

# Informed initialization
@initialization function informed_init()
    q(x) = NormalMeanPrecision(data_mean, 1.0)
end
```

### 2. Convergence Monitoring

Monitor convergence with free energy:

```julia
result = infer(
    model = my_model(),
    data = my_data,
    constraints = my_constraints(),
    initialization = my_init(),
    iterations = 100,
    free_energy = true
)

# Plot convergence
using Plots
plot(result.free_energy, 
    xlabel = "Iteration",
    ylabel = "Free Energy",
    title = "Convergence Plot"
)
```

### 3. Model Validation

Validate variational approximations:

```julia
# Compare with ground truth (if available)
using StatsPlots
p1 = plot(result.posteriors[:x], label="VI Approximation")
p2 = plot!(true_posterior, label="True Posterior")

# Check moments
@assert isapprox(mean(result.posteriors[:x]), true_mean, rtol=0.1)
```

## Common Patterns

### 1. Incremental Complexity

Build models incrementally:

```julia
# Start simple
@model function simple_model(y)
    θ ~ Normal(0, 1)
    y .~ Normal(θ, 1)
end

# Add complexity
@model function complex_model(y)
    θ ~ Normal(0, 1)
    σ ~ Gamma(1, 1)
    y .~ Normal(θ, σ)
end
```

### 2. Modular Constraints

Create reusable constraint patterns:

```julia
# Base constraints
@constraints function base_constraints()
    q(θ) :: NormalMeanPrecision
end

# Extended constraints
@constraints function extended_constraints()
    include(base_constraints())
    q(σ) :: GammaShapeRate
end
```

### 3. Diagnostic Tools

Implement diagnostic functions:

```julia
function check_approximation(result)
    # Check moments
    println("Mean: ", mean(result.posteriors[:x]))
    println("Variance: ", var(result.posteriors[:x]))
    
    # Check convergence
    Δfe = diff(result.free_energy)
    println("Converged: ", all(abs.(Δfe) .< 1e-6))
end
```

## References

- [[variational_inference|Variational Inference Theory]]
- [[message_passing|Message Passing Algorithms]]
- [[exponential_families|Exponential Families]]
- [[factor_graphs|Factor Graphs]] 