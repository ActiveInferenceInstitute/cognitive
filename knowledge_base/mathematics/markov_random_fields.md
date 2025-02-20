---
title: Markov Random Fields
type: concept
status: stable
created: 2024-03-20
tags:
  - mathematics
  - probability
  - graphical-models
  - undirected-graphs
semantic_relations:
  - type: foundation
    links: 
      - [[probabilistic_graphical_models]]
      - [[graphical_models]]
      - [[probability_theory]]
      - [[bayesian_graph_theory]]
  - type: related
    links:
      - [[bayesian_networks]]
      - [[factor_graphs]]
      - [[conditional_random_fields]]
  - type: implements
    links:
      - [[message_passing]]
      - [[probabilistic_inference]]
---

# Markov Random Fields

## Overview

Markov Random Fields (MRFs), also known as [[undirected_graphical_models|undirected graphical models]] or [[markov_networks|Markov networks]], are a type of [[probabilistic_graphical_models|probabilistic graphical model]] that represents joint probability distributions through local interactions in an undirected graph structure. They are particularly useful for modeling systems with symmetric dependencies and spatial relationships.

## Mathematical Foundation

### 1. Factorization
```math
p(x) = \frac{1}{Z} \prod_{c \in C} ψ_c(x_c)
```
where:
- Z is the [[partition_function|partition function]]
- C is the set of maximal cliques
- ψ_c are [[potential_functions|potential functions]]

### 2. [[markov_property|Markov Property]]
```math
p(x_i | x_{-i}) = p(x_i | ne(x_i))
```
where ne(x_i) denotes the neighbors of x_i

### 3. [[hammersley_clifford|Hammersley-Clifford Theorem]]
```math
p(x) > 0 \implies p(x) = \frac{1}{Z} \exp\left(-\sum_{c \in C} E_c(x_c)\right)
```
where E_c are energy functions

## Core Components

### 1. [[graph_structure|Graph Structure]]

```julia
struct MarkovRandomField
    nodes::Set{Node}
    edges::Set{UndirectedEdge}
    potentials::Dict{Clique, PotentialFunction}
    
    function MarkovRandomField()
        new(Set{Node}(), Set{UndirectedEdge}(), Dict{Clique, PotentialFunction}())
    end
end

struct Clique
    nodes::Set{Node}
    
    function Clique(nodes::Set{Node})
        # Verify clique property
        @assert all(n1 ≠ n2 ? has_edge(n1, n2) : true 
                    for n1 in nodes, n2 in nodes)
        new(nodes)
    end
end

struct PotentialFunction
    function_type::Symbol  # :tabular, :exponential, or :neural
    parameters::Dict{Symbol, Any}
    compute::Function
end
```

### 2. [[potential_functions|Potential Functions]]

#### Tabular Potentials
```julia
function create_tabular_potential(clique::Clique, 
                                values::Array{Float64})
    dims = [length(domain(node)) for node in clique.nodes]
    @assert size(values) == Tuple(dims)
    
    return PotentialFunction(
        :tabular,
        Dict(:values => values),
        x -> values[CartesianIndex(x...)]
    )
end
```

#### Exponential Family
```julia
function create_exponential_potential(clique::Clique,
                                    features::Function,
                                    weights::Vector{Float64})
    return PotentialFunction(
        :exponential,
        Dict(:weights => weights),
        x -> exp(dot(weights, features(x)))
    )
end
```

#### Neural Potentials
```julia
function create_neural_potential(clique::Clique,
                               network::NeuralNetwork)
    return PotentialFunction(
        :neural,
        Dict(:network => network),
        x -> exp(network(x))
    )
end
```

## Inference Methods

### 1. [[gibbs_sampling|Gibbs Sampling]]

```julia
function gibbs_sampling(mrf::MarkovRandomField,
                       n_samples::Int;
                       burn_in::Int=1000)
    # Initialize state
    state = initialize_state(mrf)
    samples = Vector{Dict{Node, Any}}(undef, n_samples)
    
    # Burn-in period
    for _ in 1:burn_in
        for node in mrf.nodes
            # Sample from conditional
            state[node] = sample_conditional(mrf, node, state)
        end
    end
    
    # Collect samples
    for i in 1:n_samples
        for node in mrf.nodes
            state[node] = sample_conditional(mrf, node, state)
        end
        samples[i] = copy(state)
    end
    
    return samples
end
```

### 2. [[belief_propagation|Belief Propagation]]

```julia
function loopy_belief_propagation!(mrf::MarkovRandomField;
                                 max_iters::Int=100,
                                 tolerance::Float64=1e-6)
    # Initialize messages
    messages = initialize_messages(mrf)
    
    for iter in 1:max_iters
        old_messages = copy(messages)
        
        # Update all messages
        for edge in mrf.edges
            messages[edge] = compute_message(mrf, edge, old_messages)
        end
        
        # Check convergence
        if maximum(abs.(messages - old_messages)) < tolerance
            break
        end
    end
    
    return compute_beliefs(mrf, messages)
end
```

### 3. [[mean_field|Mean Field Inference]]

```julia
function mean_field_inference!(mrf::MarkovRandomField;
                             max_iters::Int=100,
                             tolerance::Float64=1e-6)
    # Initialize variational parameters
    q = initialize_mean_field(mrf)
    
    for iter in 1:max_iters
        q_old = copy(q)
        
        # Update each node
        for node in mrf.nodes
            q[node] = optimize_local_parameters(mrf, node, q)
        end
        
        # Check convergence
        if maximum(abs.(q - q_old)) < tolerance
            break
        end
    end
    
    return q
end
```

## Learning

### 1. [[parameter_learning|Parameter Learning]]

#### Maximum Likelihood
```julia
function learn_parameters!(mrf::MarkovRandomField,
                         data::Vector{Dict{Node, Any}})
    # For each potential function
    for (clique, potential) in mrf.potentials
        if potential.function_type == :exponential
            # Compute empirical expectations
            E_empirical = compute_empirical_expectations(data, clique)
            
            # Optimize weights
            weights = optimize_weights(E_empirical, potential)
            
            # Update potential
            update_potential!(mrf, clique, weights)
        end
    end
end
```

#### Pseudolikelihood
```julia
function maximize_pseudolikelihood!(mrf::MarkovRandomField,
                                  data::Vector{Dict{Node, Any}})
    # For each node
    for node in mrf.nodes
        # Collect local configurations
        local_data = extract_local_configurations(data, node)
        
        # Optimize local parameters
        params = optimize_local_parameters(local_data)
        
        # Update local potentials
        update_local_potentials!(mrf, node, params)
    end
end
```

### 2. [[structure_learning|Structure Learning]]

```julia
function learn_structure!(mrf::MarkovRandomField,
                         data::Vector{Dict{Node, Any}},
                         regularizer::Float64)
    # Initialize empty graph
    edges = Set{UndirectedEdge}()
    
    # For each pair of nodes
    for n1 in mrf.nodes, n2 in mrf.nodes
        if n1 < n2  # Avoid duplicates
            # Compute edge score
            score = compute_edge_score(data, n1, n2)
            
            # Add edge if significant
            if score > regularizer
                push!(edges, UndirectedEdge(n1, n2))
            end
        end
    end
    
    # Update graph structure
    mrf.edges = edges
    
    # Initialize potential functions
    initialize_potentials!(mrf)
end
```

## Applications

### 1. [[image_processing|Image Processing]]

#### Image Segmentation
```julia
function segment_image(image::Matrix{Float64},
                      n_segments::Int)
    # Create MRF
    mrf = create_image_mrf(size(image))
    
    # Define potentials
    add_intensity_potentials!(mrf, image)
    add_smoothness_potentials!(mrf)
    
    # Run inference
    labels = graph_cut_inference(mrf, n_segments)
    
    return labels
end
```

### 2. [[spatial_statistics|Spatial Statistics]]

```julia
function spatial_modeling(locations::Vector{Point2D},
                        observations::Vector{Float64})
    # Create spatial MRF
    mrf = create_spatial_mrf(locations)
    
    # Add distance-based potentials
    add_spatial_potentials!(mrf, locations)
    
    # Add observation model
    add_observation_potentials!(mrf, observations)
    
    # Perform inference
    return spatial_inference(mrf)
end
```

### 3. [[computer_vision|Computer Vision]]

```julia
function scene_understanding(image::Matrix{Float64})
    # Create hierarchical MRF
    mrf = create_hierarchical_mrf()
    
    # Add appearance potentials
    add_appearance_potentials!(mrf, image)
    
    # Add spatial relationships
    add_spatial_relations!(mrf)
    
    # Perform inference
    return hierarchical_inference(mrf)
end
```

## Integration with Other Models

### 1. [[conversion_methods|Conversion Methods]]

#### From Bayesian Network
```julia
function from_bayesian_network(bn::BayesianNetwork)
    # Create MRF
    mrf = MarkovRandomField()
    
    # Moralize graph
    moral_graph = moralize(bn)
    
    # Convert CPDs to potentials
    potentials = convert_cpds_to_potentials(bn.parameters)
    
    # Add to MRF
    add_structure!(mrf, moral_graph)
    add_potentials!(mrf, potentials)
    
    return mrf
end
```

#### To Factor Graph
```julia
function to_factor_graph(mrf::MarkovRandomField)
    # Create factor graph
    fg = FactorGraph()
    
    # Add variable nodes
    for node in mrf.nodes
        add_variable!(fg, node)
    end
    
    # Add factor nodes for potentials
    for (clique, potential) in mrf.potentials
        add_factor!(fg, create_factor(clique, potential))
    end
    
    return fg
end
```

### 2. [[hybrid_models|Hybrid Models]]

```julia
struct HybridGraphicalModel
    mrf_component::MarkovRandomField
    bn_component::BayesianNetwork
    connections::Dict{Node, Set{Node}}
    
    function HybridGraphicalModel(mrf, bn)
        connections = identify_connections(mrf, bn)
        new(mrf, bn, connections)
    end
end

function hybrid_inference(model::HybridGraphicalModel,
                        query::Set{Node},
                        evidence::Dict{Node, Any})
    # Initialize messages
    messages = initialize_hybrid_messages(model)
    
    # Iterative inference
    for _ in 1:max_iters
        # Update MRF messages
        update_mrf_messages!(model.mrf_component, messages)
        
        # Update BN messages
        update_bn_messages!(model.bn_component, messages)
        
        # Update connection messages
        update_connection_messages!(model, messages)
    end
    
    return compute_hybrid_marginals(model, messages, query)
end
```

## Best Practices

### 1. Model Design
- Choose appropriate potential functions
- Consider computational tractability
- Design efficient clique structure
- Balance model complexity

### 2. Implementation
- Use sparse representations
- Implement efficient message passing
- Cache intermediate computations
- Handle numerical stability

### 3. Validation
- Test with synthetic data
- Validate independence assumptions
- Monitor convergence
- Analyze sensitivity

## References

1. Kindermann, R., & Snell, J. L. (1980). Markov Random Fields and Their Applications
2. Li, S. Z. (2009). Markov Random Field Modeling in Image Analysis
3. Koller, D., & Friedman, N. (2009). Probabilistic Graphical Models
4. Wainwright, M. J., & Jordan, M. I. (2008). Graphical Models, Exponential Families, and Variational Inference
5. Blake, A., Kohli, P., & Rother, C. (2011). Markov Random Fields for Vision and Image Processing 