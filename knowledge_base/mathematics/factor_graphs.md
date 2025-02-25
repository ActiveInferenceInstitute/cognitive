---
title: Factor Graphs
type: concept
status: stable
created: 2024-03-20
tags:
  - mathematics
  - graphical-models
  - inference
semantic_relations:
  - type: foundation
    links: 
      - [[graphical_models]]
      - [[probabilistic_models]]
      - [[bayesian_networks]]
  - type: implements
    links:
      - [[message_passing]]
      - [[belief_propagation]]
      - [[variational_inference]]
  - type: related
    links:
      - [[markov_random_fields]]
      - [[conditional_random_fields]]
      - [[probabilistic_programming]]
---

# Factor Graphs

## Overview

Factor graphs are a type of [[graphical_models|graphical model]] that represents factorizations of functions, particularly probability distributions. They provide a unified framework for [[inference_algorithms|inference algorithms]] and bridge the gap between [[directed_graphical_models|directed]] and [[undirected_graphical_models|undirected]] graphical models.

## Mathematical Foundation

### 1. Basic Structure
A factor graph G = (V, F, E) consists of:
- V: Variable nodes representing [[random_variables|random variables]]
- F: Factor nodes encoding [[probability_distributions|probability distributions]]
- E: Edges representing [[probabilistic_dependencies|probabilistic dependencies]]

### 2. Factorization
For a probability distribution p(x):
```math
p(x_1, ..., x_n) = \prod_{a \in F} f_a(x_{\partial a})
```
where:
- f_a are [[factor_functions|factor functions]]
- ∂a denotes variables connected to factor a

### 3. [[bayesian_factorization|Bayesian Factorization]]
In Bayesian terms:
```math
p(x, θ | y) ∝ p(y | x, θ)p(x | θ)p(θ)
```
where:
- p(y | x, θ) is the [[likelihood_function|likelihood]]
- p(x | θ) is the [[prior_distribution|prior]]
- p(θ) is the [[hyperprior|hyperprior]]

## Components and Structure

### 1. [[variable_nodes|Variable Nodes]]

#### Types and Properties
```julia
struct VariableNode{T}
    id::Symbol
    domain::Domain{T}
    neighbors::Set{FactorNode}
    messages::Dict{FactorNode, Message}
    belief::Distribution{T}
end
```

#### Categories
- **Observable Variables**
  - Represent data points
  - Fixed during inference
  - Drive belief updates

- **Latent Variables**
  - Hidden states
  - Model parameters
  - Inferred quantities

- **Parameter Nodes**
  - [[hyperparameters|Hyperparameters]]
  - [[model_parameters|Model parameters]]
  - [[sufficient_statistics|Sufficient statistics]]

### 2. [[factor_nodes|Factor Nodes]]

#### Base Implementation
```julia
abstract type FactorNode end

struct ProbabilisticFactor <: FactorNode
    distribution::Distribution
    variables::Vector{VariableNode}
    parameters::Dict{Symbol, Any}
end

struct DeterministicFactor <: FactorNode
    function::Function
    inputs::Vector{VariableNode}
    outputs::Vector{VariableNode}
end

struct ConstraintFactor <: FactorNode
    constraint::Function
    variables::Vector{VariableNode}
    tolerance::Float64
end
```

#### Message Computation
```julia
function compute_message(factor::FactorNode, to::VariableNode)
    # Collect incoming messages
    messages = [msg for (node, msg) in factor.messages if node != to]
    
    # Compute outgoing message
    if isa(factor, ProbabilisticFactor)
        return compute_probabilistic_message(factor, messages, to)
    elseif isa(factor, DeterministicFactor)
        return compute_deterministic_message(factor, messages, to)
    else
        return compute_constraint_message(factor, messages, to)
    end
end
```

### 3. [[edges|Edges]]

#### Properties
```julia
struct Edge
    source::Union{VariableNode, FactorNode}
    target::Union{VariableNode, FactorNode}
    message_type::Type{<:Message}
    parameters::Dict{Symbol, Any}
end
```

#### Message Types
- Forward messages (variable to factor)
- Backward messages (factor to variable)
- Parameter messages
- Constraint messages

## Message Passing and Inference

### 1. [[belief_propagation|Belief Propagation]]

#### Forward Messages
```math
μ_{x→f}(x) = \prod_{g \in N(x) \backslash f} μ_{g→x}(x)
```

#### Backward Messages
```math
μ_{f→x}(x) = \sum_{x_{\partial f \backslash x}} f(x_{\partial f}) \prod_{y \in \partial f \backslash x} μ_{y→f}(y)
```

#### Implementation
```julia
function belief_propagation!(graph::FactorGraph; max_iters=100)
    for iter in 1:max_iters
        # Update variable to factor messages
        for var in graph.variables
            for factor in var.neighbors
                message = compute_var_to_factor_message(var, factor)
                update_message!(var, factor, message)
            end
        end
        
        # Update factor to variable messages
        for factor in graph.factors
            for var in factor.variables
                message = compute_factor_to_var_message(factor, var)
                update_message!(factor, var, message)
            end
        end
        
        # Check convergence
        if check_convergence(graph)
            break
        end
    end
end
```

### 2. [[variational_message_passing|Variational Message Passing]]

#### ELBO Optimization
```math
\mathcal{L}(q) = \mathbb{E}_q[\log p(x,z)] - \mathbb{E}_q[\log q(z)]
```

#### Natural Gradient Updates
```math
θ_t = θ_{t-1} + η\nabla_{\text{nat}}\mathcal{L}(q)
```

#### Implementation
```julia
function variational_message_passing!(graph::FactorGraph; 
                                    learning_rate=0.01, 
                                    max_iters=100)
    for iter in 1:max_iters
        # Compute natural gradients
        gradients = compute_natural_gradients(graph)
        
        # Update variational parameters
        for (node, grad) in gradients
            update_parameters!(node, grad, learning_rate)
        end
        
        # Update messages
        update_messages!(graph)
        
        # Check ELBO convergence
        if check_elbo_convergence(graph)
            break
        end
    end
end
```

### 3. [[expectation_propagation|Expectation Propagation]]

#### Moment Matching
```math
\text{minimize}_{q_i} \text{KL}(p||q_1...q_i...q_n)
```

#### Implementation
```julia
function expectation_propagation!(graph::FactorGraph; max_iters=100)
    for iter in 1:max_iters
        # Update approximate factors
        for factor in graph.factors
            # Compute cavity distribution
            cavity = compute_cavity_distribution(factor)
            
            # Moment matching
            new_approx = moment_match(cavity, factor)
            
            # Update approximation
            update_approximation!(factor, new_approx)
        end
        
        # Check convergence
        if check_convergence(graph)
            break
        end
    end
end
```

## Advanced Topics

### 1. [[structured_factor_graphs|Structured Factor Graphs]]

#### Temporal Structure
```julia
@model function temporal_factor_graph(T)
    # State variables
    x = Vector{VariableNode}(undef, T)
    
    # Temporal factors
    for t in 2:T
        @factor f[t] begin
            x[t] ~ transition(x[t-1])
        end
    end
    
    return x
end
```

#### Hierarchical Structure
```julia
@model function hierarchical_factor_graph()
    # Global parameters
    θ_global ~ prior()
    
    # Local parameters
    θ_local = Vector{VariableNode}(undef, N)
    for i in 1:N
        θ_local[i] ~ conditional(θ_global)
    end
    
    return θ_local
end
```

### 2. [[continuous_variables|Continuous Variables]]

#### Gaussian Messages
```julia
struct GaussianMessage <: Message
    mean::Vector{Float64}
    precision::Matrix{Float64}
    
    function GaussianMessage(μ, Λ)
        @assert size(μ, 1) == size(Λ, 1) == size(Λ, 2)
        new(μ, Λ)
    end
end

function multiply_messages(m1::GaussianMessage, m2::GaussianMessage)
    Λ = m1.precision + m2.precision
    μ = Λ \ (m1.precision * m1.mean + m2.precision * m2.mean)
    return GaussianMessage(μ, Λ)
end
```

### 3. [[convergence_properties|Convergence Properties]]

#### Fixed Point Conditions
```math
b^*(x) = \frac{1}{Z} \prod_{f \in N(x)} μ^*_{f→x}(x)
```

#### Bethe Free Energy
```math
F_{\text{Bethe}} = \sum_i F_i + \sum_a F_a
```

## Implementation

### 1. Graph Construction
```julia
struct FactorGraph
    variables::Set{VariableNode}
    factors::Set{FactorNode}
    edges::Set{Edge}
    
    function FactorGraph()
        new(Set{VariableNode}(), Set{FactorNode}(), Set{Edge}())
    end
end

function add_variable!(graph::FactorGraph, var::VariableNode)
    push!(graph.variables, var)
end

function add_factor!(graph::FactorGraph, factor::FactorNode)
    push!(graph.factors, factor)
    for var in factor.variables
        add_edge!(graph, Edge(var, factor))
    end
end
```

### 2. Message Scheduling
```julia
struct MessageSchedule
    order::Vector{Tuple{Union{VariableNode,FactorNode}, 
                       Union{VariableNode,FactorNode}}}
    priorities::Vector{Float64}
end

function schedule_messages(graph::FactorGraph)
    schedule = MessageSchedule()
    
    # Forward pass
    for level in graph.levels
        for node in level
            schedule_forward_messages!(schedule, node)
        end
    end
    
    # Backward pass
    for level in reverse(graph.levels)
        for node in level
            schedule_backward_messages!(schedule, node)
        end
    end
    
    return schedule
end
```

### 3. Inference Execution
```julia
function run_inference(graph::FactorGraph; 
                      method=:belief_propagation,
                      max_iters=100)
    if method == :belief_propagation
        belief_propagation!(graph, max_iters=max_iters)
    elseif method == :variational
        variational_message_passing!(graph, max_iters=max_iters)
    elseif method == :expectation_propagation
        expectation_propagation!(graph, max_iters=max_iters)
    else
        error("Unknown inference method: $method")
    end
    
    return compute_beliefs(graph)
end
```

## Applications

### 1. [[bayesian_inference|Bayesian Inference]]
- Parameter estimation
- Model selection
- Uncertainty quantification

### 2. [[probabilistic_programming|Probabilistic Programming]]
- Model specification
- Automatic inference
- Compositional modeling

### 3. [[active_inference|Active Inference]]
- Policy selection
- Perception-action loops
- Free energy minimization

## Best Practices

### 1. Design Patterns
- Modular factor construction
- Reusable message computations
- Efficient graph structures

### 2. Numerical Considerations
- Message normalization
- Numerical stability
- Convergence monitoring

### 3. Testing and Validation
- Unit tests for factors
- Message validity checks
- End-to-end inference tests

## Integration with Bayesian Networks

### 1. [[bayesian_network_conversion|Bayesian Network Conversion]]
```julia
function from_bayesian_network(bn::BayesianNetwork)
    # Create factor graph
    fg = FactorGraph()
    
    # Add variables
    for node in bn.nodes
        add_variable!(fg, VariableNode(node))
    end
    
    # Add CPT factors
    for (var, cpt) in bn.parameters
        add_cpt_factor!(fg, var, cpt)
    end
    
    return fg
end
```

### 2. [[inference_equivalence|Inference Equivalence]]
- Message passing in trees
- Loopy belief propagation
- Variational approximations

### 3. [[model_comparison|Model Comparison]]
- Structure comparison
- Parameter learning
- Performance metrics

## RxInfer Integration

### 1. Reactive Message Passing

```julia
using RxInfer

@model function rxinfer_factor_graph(data)
    # Prior distributions
    θ ~ GaussianMeanPrecision(0.0, 1.0)
    σ ~ GammaShape(1.0, 1.0)
    
    # Likelihood factors
    for i in 1:length(data)
        y[i] ~ GaussianMeanPrecision(θ, σ)
    end
end

# Create inference algorithm
algorithm = ReactiveMP.messagePassingAlgorithm(
    model = rxinfer_factor_graph,
    data = observed_data
)

# Execute inference
results = ReactiveMP.infer(algorithm)
```

### 2. Streaming Factor Graphs

```julia
struct StreamingFactorGraph
    base_graph::FactorGraph
    stream_nodes::Vector{StreamNode}
    buffer_size::Int
    
    function StreamingFactorGraph(model, buffer_size=1000)
        graph = create_base_graph(model)
        stream_nodes = initialize_stream_nodes(model)
        new(graph, stream_nodes, buffer_size)
    end
end

function process_stream!(graph::StreamingFactorGraph, data_stream)
    for batch in data_stream
        # Update streaming nodes
        update_stream_nodes!(graph, batch)
        
        # Perform message passing
        propagate_messages!(graph)
        
        # Update beliefs
        update_beliefs!(graph)
        
        # Prune old messages if needed
        prune_old_messages!(graph)
    end
end
```

### 3. Reactive Message Types

```julia
abstract type ReactiveMessage end

struct GaussianReactiveMessage <: ReactiveMessage
    mean::Observable{Float64}
    precision::Observable{Float64}
    
    function GaussianReactiveMessage(μ::Observable, τ::Observable)
        new(μ, τ)
    end
end

struct StreamingMessage <: ReactiveMessage
    distribution::Observable{Distribution}
    timestamp::Observable{Float64}
    
    function StreamingMessage(dist::Observable, ts::Observable)
        new(dist, ts)
    end
end
```

## Advanced Bayesian Integration

### 1. Hierarchical Bayesian Models

```mermaid
graph TD
    subgraph "Hierarchical Factor Structure"
        A[Global Parameters] --> B[Group Parameters]
        B --> C[Individual Parameters]
        C --> D[Observations]
        E[Hyperpriors] --> A
    end
    style A fill:#f9f,stroke:#333
    style C fill:#bbf,stroke:#333
    style E fill:#bfb,stroke:#333
```

### 2. Conjugate Factor Pairs

```julia
struct ConjugateFactor{T<:Distribution} <: FactorNode
    prior::T
    likelihood::Function
    posterior::T
    sufficient_stats::Dict{Symbol, Any}
    
    function ConjugateFactor(prior::T, likelihood::Function) where T
        posterior = deepcopy(prior)
        stats = initialize_sufficient_stats(prior)
        new{T}(prior, likelihood, posterior, stats)
    end
end

function update_conjugate_factor!(factor::ConjugateFactor, data)
    # Update sufficient statistics
    update_stats!(factor.sufficient_stats, data)
    
    # Compute posterior parameters
    posterior_params = compute_posterior_params(
        factor.prior, factor.sufficient_stats)
    
    # Update posterior
    update_posterior!(factor.posterior, posterior_params)
end
```

### 3. Non-parametric Extensions

```julia
struct DirichletProcessFactor <: FactorNode
    base_measure::Distribution
    concentration::Float64
    clusters::Vector{Cluster}
    assignments::Dict{Int, Int}
    
    function DirichletProcessFactor(base::Distribution, α::Float64)
        new(base, α, Cluster[], Dict())
    end
end

function sample_assignment(factor::DirichletProcessFactor, data_point)
    # Compute cluster probabilities
    probs = compute_cluster_probabilities(factor, data_point)
    
    # Sample new assignment
    assignment = sample_categorical(probs)
    
    # Update clusters if necessary
    if assignment > length(factor.clusters)
        create_new_cluster!(factor, data_point)
    end
    
    return assignment
end
```

## Advanced Message Passing Schemes

### 1. Stochastic Message Passing

```julia
struct StochasticMessagePassing
    n_particles::Int
    resampling_threshold::Float64
    
    function StochasticMessagePassing(n_particles=1000, threshold=0.5)
        new(n_particles, threshold)
    end
end

function propagate_particles!(smp::StochasticMessagePassing, 
                            factor::FactorNode,
                            particles::Vector{Particle})
    # Propagate particles through factor
    weights = zeros(smp.n_particles)
    new_particles = similar(particles)
    
    for i in 1:smp.n_particles
        # Sample
        new_particles[i] = propose_particle(factor, particles[i])
        
        # Weight
        weights[i] = compute_importance_weight(
            factor, particles[i], new_particles[i])
    end
    
    # Resample if needed
    if effective_sample_size(weights) < smp.resampling_threshold
        new_particles = resample_particles(new_particles, weights)
    end
    
    return new_particles
end
```

### 2. Distributed Message Passing

```julia
struct DistributedFactorGraph
    subgraphs::Vector{FactorGraph}
    interfaces::Dict{Tuple{Int,Int}, Interface}
    
    function DistributedFactorGraph(graph::FactorGraph, n_partitions)
        # Partition graph
        subgraphs = partition_graph(graph, n_partitions)
        
        # Create interfaces
        interfaces = create_interfaces(subgraphs)
        
        new(subgraphs, interfaces)
    end
end

function distributed_inference!(graph::DistributedFactorGraph)
    # Initialize workers
    workers = [Worker(subgraph) for subgraph in graph.subgraphs]
    
    while !converged(workers)
        # Local inference
        @sync for worker in workers
            @async local_inference!(worker)
        end
        
        # Exchange messages
        exchange_interface_messages!(graph)
        
        # Update beliefs
        update_worker_beliefs!(workers)
    end
end
```

### 3. Adaptive Message Scheduling

```julia
struct AdaptiveScheduler
    priority_queue::PriorityQueue{Message, Float64}
    residual_threshold::Float64
    
    function AdaptiveScheduler(threshold=1e-6)
        new(PriorityQueue{Message, Float64}(), threshold)
    end
end

function schedule_message!(scheduler::AdaptiveScheduler, 
                         message::Message,
                         residual::Float64)
    if residual > scheduler.residual_threshold
        enqueue!(scheduler.priority_queue, message => residual)
    end
end

function process_messages!(scheduler::AdaptiveScheduler)
    while !isempty(scheduler.priority_queue)
        # Get highest priority message
        message = dequeue!(scheduler.priority_queue)
        
        # Update message
        new_message = update_message!(message)
        
        # Compute residual and reschedule if needed
        residual = compute_residual(message, new_message)
        schedule_message!(scheduler, new_message, residual)
    end
end
```

## Integration with Active Inference

### 1. Free Energy Minimization

```julia
struct FreeEnergyFactor <: FactorNode
    internal_states::Vector{VariableNode}
    external_states::Vector{VariableNode}
    precision::Matrix{Float64}
    
    function FreeEnergyFactor(internal, external, precision)
        new(internal, external, precision)
    end
end

function compute_free_energy(factor::FreeEnergyFactor)
    # Compute prediction error
    error = compute_prediction_error(
        factor.internal_states,
        factor.external_states
    )
    
    # Weight by precision
    weighted_error = error' * factor.precision * error
    
    # Add complexity penalty
    complexity = compute_complexity_term(factor.internal_states)
    
    return 0.5 * weighted_error + complexity
end
```

### 2. Policy Selection

```julia
struct PolicyFactor <: FactorNode
    policies::Vector{Policy}
    expected_free_energy::Vector{Float64}
    temperature::Float64
    
    function PolicyFactor(policies, temperature=1.0)
        n_policies = length(policies)
        new(policies, zeros(n_policies), temperature)
    end
end

function select_policy(factor::PolicyFactor)
    # Compute softmax probabilities
    probs = softmax(-factor.temperature * factor.expected_free_energy)
    
    # Sample policy
    policy_idx = sample_categorical(probs)
    
    return factor.policies[policy_idx]
end
```

## Performance Optimization

### 1. Message Caching

```julia
struct MessageCache
    storage::Dict{Tuple{FactorNode, VariableNode}, Message}
    max_size::Int
    eviction_policy::Symbol
    
    function MessageCache(max_size=10000, policy=:lru)
        new(Dict(), max_size, policy)
    end
end

function cache_message!(cache::MessageCache,
                       factor::FactorNode,
                       variable::VariableNode,
                       message::Message)
    key = (factor, variable)
    
    # Evict if needed
    if length(cache.storage) >= cache.max_size
        evict_message!(cache)
    end
    
    # Store message
    cache.storage[key] = message
end
```

### 2. Parallel Message Updates

```julia
function parallel_message_passing!(graph::FactorGraph)
    # Group independent messages
    message_groups = group_independent_messages(graph)
    
    # Update messages in parallel
    @sync for group in message_groups
        @async begin
            for message in group
                update_message!(message)
            end
        end
    end
end
```

### 3. GPU Acceleration

```julia
struct GPUFactorGraph
    variables::CuArray{VariableNode}
    factors::CuArray{FactorNode}
    messages::CuArray{Message}
    
    function GPUFactorGraph(graph::FactorGraph)
        # Transfer to GPU
        variables = cu(collect(graph.variables))
        factors = cu(collect(graph.factors))
        messages = cu(collect_messages(graph))
        
        new(variables, factors, messages)
    end
end

function gpu_message_passing!(graph::GPUFactorGraph)
    # Kernel for parallel message updates
    @cuda threads=256 blocks=div(length(graph.messages), 256) do
        update_messages_kernel(graph.messages)
    end
    
    synchronize()
end
```

## References

1. Kschischang, F. R., et al. (2001). Factor Graphs and the Sum-Product Algorithm
2. Wainwright, M. J., & Jordan, M. I. (2008). Graphical Models, Exponential Families, and Variational Inference
3. Loeliger, H. A. (2004). An Introduction to Factor Graphs
4. Bishop, C. M. (2006). Pattern Recognition and Machine Learning
5. Koller, D., & Friedman, N. (2009). Probabilistic Graphical Models 