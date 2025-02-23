---
title: Streaming Inference in RxInfer
type: documentation
status: stable
created: 2024-03-20
tags:
  - rxinfer
  - streaming
  - online-learning
semantic_relations:
  - type: implements
    links: 
      - [[online_learning]]
      - [[reactive_programming]]
---

# Streaming Inference in RxInfer

## Overview

RxInfer provides powerful support for [[online_learning|online learning]] and [[streaming_data|streaming data]] processing through its [[reactive_programming|reactive programming]] architecture. This guide explains how to perform inference on streaming data sources.

## Core Concepts

### 1. Reactive Data Streams

RxInfer integrates with [[rocket_jl|Rocket.jl]] for reactive stream processing:

```julia
using RxInfer, Rocket

# Create a data stream
stream = Subject(Float64)

# Create labeled stream for RxInfer
observations = labeled(Val((:y,)), stream)
```

### 2. Model Specification

Models for streaming inference are similar to static models:

```julia
@model function streaming_gaussian(y)
    # Prior beliefs
    μ ~ Normal(0, 10)    # Mean
    τ ~ Gamma(1, 1)      # Precision
    
    # Streaming observation
    y ~ Normal(μ, τ)     # Single observation
end
```

### 3. Automatic Updates

Use `@autoupdates` to specify how priors should update:

```julia
updates = @autoupdates begin
    # Update prior parameters based on current posterior
    μ_prior, τ_prior = params(q(μ))
end
```

## Streaming Inference Examples

### 1. Online Mean Estimation

```julia
using RxInfer, Rocket, Distributions

# Model definition
@model function online_mean(y)
    μ ~ Normal(mean = 0, precision = 0.1)
    y ~ Normal(mean = μ, precision = 1.0)
end

# Auto-updates for online learning
updates = @autoupdates begin
    μ_mean, μ_prec = params(q(μ))
end

# Create data stream
stream = Subject(Float64)
observations = labeled(Val((:y,)), stream)

# Run streaming inference
result = infer(
    model = online_mean(),
    datastream = observations,
    autoupdates = updates
)

# Subscribe to posterior updates
subscribe!(result.posteriors[:μ]) do posterior
    println("Updated mean: ", mean(posterior))
end

# Feed data
next!(stream, 1.0)
next!(stream, 2.0)
next!(stream, 3.0)
```

### 2. State Space Model

```julia
@model function state_space(y)
    # Initial state
    x₁ ~ Normal(0, 1)
    x = [x₁]
    
    # State transition
    x_next ~ Normal(mean = last(x), precision = 1.0)
    push!(x, x_next)
    
    # Observation
    y ~ Normal(mean = last(x), precision = 1.0)
end

# Auto-updates for state tracking
updates = @autoupdates begin
    x_mean, x_prec = params(q(x_next))
end
```

## Advanced Features

### 1. Deferred Data Handling

For complex streaming scenarios:

```julia
@model function complex_stream(y, n)
    # State vector
    x = Vector{Random}(undef, n)
    x[1] ~ Normal(0, 1)
    
    # Streaming updates
    for t in 2:n
        x[t] ~ Normal(x[t-1], 1)
    end
    
    # Observations
    y .~ Normal(x, 1)
end

# Defer data handling
model | (y = RxInfer.DeferredDataHandler(),)
```

### 2. Message Passing Control

Fine-tune the streaming inference:

```julia
result = infer(
    model = my_model(),
    datastream = my_stream,
    autoupdates = my_updates,
    scheduler = CustomScheduler(),
    messagepassingiterations = 5
)
```

### 3. Reactive Transformations

Transform streams before inference:

```julia
# Create transformed stream
filtered = observations |> 
    filter(x -> !isnan(x)) |>
    map(x -> (y = float(x),))

# Use in inference
result = infer(
    model = my_model(),
    datastream = filtered
)
```

## Best Practices

### 1. Stream Management

- Handle stream completion properly
- Consider backpressure for fast streams
- Clean up resources with unsubscribe

```julia
# Proper stream cleanup
subscription = subscribe!(stream, handler)
# ... use stream ...
unsubscribe!(subscription)
```

### 2. Performance Optimization

- Use appropriate buffer sizes
- Consider message passing iterations
- Monitor memory usage

```julia
result = infer(
    model = my_model(),
    datastream = my_stream,
    buffersizehint = 1000,
    messagepassingiterations = 3
)
```

### 3. Error Handling

Implement proper error handling:

```julia
subscription = subscribe!(stream,
    # OnNext handler
    data -> try_update(model, data),
    # OnError handler
    err -> handle_error(err),
    # OnCompleted handler
    () -> cleanup_resources()
)
```

## Common Patterns

### 1. Windowed Processing

Process data in windows:

```julia
@model function window_model(ys)
    μ ~ Normal(0, 1)
    ys .~ Normal(μ, 1)
end

# Create windowed stream
windowed = observations |>
    buffer(size = 10, stride = 5) |>
    map(window -> (ys = collect(window),))
```

### 2. Adaptive Learning

Implement adaptive learning rates:

```julia
@model function adaptive_model(y)
    # Learning rate
    α ~ Gamma(1, 1)
    
    # State
    θ ~ Normal(0, 1/α)
    
    # Observation
    y ~ Normal(θ, 1)
end

updates = @autoupdates begin
    # Update learning rate based on uncertainty
    α = 1/var(q(θ))
end
```

### 3. Online Model Selection

Perform model selection online:

```julia
@model function model_selection(y)
    # Model index
    k ~ Categorical([0.5, 0.5])
    
    # Model parameters
    θ₁ ~ Normal(0, 1)
    θ₂ ~ Normal(0, 1)
    
    # Model averaging
    y ~ Normal(k == 1 ? θ₁ : θ₂, 1)
end
```

## Debugging and Monitoring

### 1. Stream Debugging

Add debug points in your stream:

```julia
observations |>
    tap(x -> println("Raw: ", x)) |>
    map(process_data) |>
    tap(x -> println("Processed: ", x))
```

### 2. Performance Monitoring

Monitor inference performance:

```julia
result = infer(
    model = my_model(),
    datastream = my_stream,
    free_energy = true
)

subscribe!(result.free_energy) do fe
    println("Free Energy: ", fe)
end
```

## References

- [[reactive_programming|Reactive Programming]]
- [[online_learning|Online Learning]]
- [[message_passing|Message Passing]]
- [[rocket_jl|Rocket.jl Documentation]] 