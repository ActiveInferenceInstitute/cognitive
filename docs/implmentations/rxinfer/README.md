# RxInfer.jl

RxInfer.jl is a powerful Julia package for [[bayesian_inference|Bayesian Inference]] on [[factor_graphs|Factor Graphs]] using [[message_passing|Message Passing]]. It provides a flexible and efficient framework for both [[exact_inference|exact]] and [[variational_inference|variational inference]] algorithms through [[reactive_programming|reactive message passing]].

## Key Features

- **Flexible Model Specification**: Easy-to-use domain-specific language for defining [[probabilistic_models|probabilistic models]] using the [`@model`](https://reactivebayes.github.io/GraphPPL.jl/stable/) macro
- **Multiple Inference Methods**: Support for both [[exact_inference|exact]] and [[variational_inference|variational inference]] algorithms
- **[[streaming_inference|Streaming Inference]]**: Capability to handle both [[static_inference|static datasets]] and [[dynamic_inference|dynamic/streaming data]]
- **[[reactive_programming|Reactive Architecture]]**: Built on reactive programming principles for efficient asynchronous processing
- **[[extensible_framework|Extensible Framework]]**: Easy integration of [[custom_nodes|custom nodes]] and message passing rules
- **[[non_conjugate_inference|Non-conjugate Inference]]**: Support for models with non-conjugate structures
- **[[deterministic_transformations|Deterministic Transformations]]**: Comprehensive support for deterministic nodes

## Installation

Install RxInfer through the Julia package manager:

```julia
] add RxInfer
```

Or using Pkg:

```julia
using Pkg
Pkg.add("RxInfer")
```

## Quick Start

Here's a simple example of using RxInfer to infer the bias of a coin using [[beta_bernoulli|Beta-Bernoulli inference]]:

```julia
using RxInfer

# Define the model
@model function coin_model(y, a, b)
    θ ~ Beta(a, b)  # Prior on coin bias
    y .~ Bernoulli(θ) # Likelihood for coin flips
end

# Generate some synthetic data
using Distributions, Random
rng = MersenneTwister(42)
true_bias = 0.75
data = rand(rng, Bernoulli(true_bias), 1000)

# Run inference
result = infer(
    model = coin_model(a = 1.0, b = 1.0),
    data = (y = data,)
)

# Access the posterior distribution
posterior = result.posteriors[:θ]
```

For more examples, check out the [examples repository](https://reactivebayes.github.io/RxInferExamples.jl/).

## Core Components

RxInfer is built around three main packages in the [[reactive_bayes|ReactiveBayes]] ecosystem:

1. **[[reactive_mp|ReactiveMP.jl]]**: The underlying message passing-based inference engine
2. **[[graph_ppl|GraphPPL.jl]]**: Model and constraints specification package
3. **[[rocket_jl|Rocket.jl]]**: Reactive extensions package for Julia

## Key Concepts

### [[model_specification|Model Specification]]

Models are specified using the `@model` macro, which translates a textual description into a [[factor_graphs|factor graph]] representation. See the [Model Specification guide](https://reactivebayes.github.io/RxInfer.jl/stable/manuals/model-specification/) for details.

```julia
@model function my_model(y)
    μ ~ Normal(0.0, 1.0)
    τ ~ Gamma(1.0, 1.0)
    y .~ Normal(μ, τ)
end
```

### Inference Types

- **[[static_inference|Static Inference]]**: For fixed datasets
- **[[streaming_inference|Streaming Inference]]**: For continuous data streams with automatic updates
- **[[non_conjugate_inference|Non-conjugate Inference]]**: Support for models with non-conjugate structures
- **[[deterministic_nodes|Deterministic Transformations]]**: Various approximation methods for deterministic nodes

### [[constraints_initialization|Constraints and Initialization]]

- Use [[variational_constraints|@constraints]] for specifying variational constraints
- Use [[initialization|@initialization]] for setting initial values in iterative inference
- Support for [[auto_updates|automatic updates]] with `@autoupdates`

## Advanced Features

### Non-conjugate Inference

RxInfer supports non-conjugate inference through the `ExponentialFamilyProjection` package. See the [Non-conjugate Inference guide](https://reactivebayes.github.io/RxInfer.jl/stable/manuals/inference/nonconjugate/) for details.

```julia
@constraints function model_constraints()
    q(x) :: ProjectedTo(Beta)
    q(x, y) = q(x)q(y)
end
```

### Deterministic Nodes

Support for various approximation methods:
- [Linearization](https://reactivebayes.github.io/RxInfer.jl/stable/manuals/inference/delta-node/#Gaussian-Case)
- [Unscented Transform](https://reactivebayes.github.io/RxInfer.jl/stable/manuals/inference/delta-node/#Gaussian-Case)
- [CVI Projection](https://reactivebayes.github.io/RxInfer.jl/stable/manuals/inference/delta-node/#Features-and-Supported-Inference-Scenarios)
- [Inverse Function Support](https://reactivebayes.github.io/RxInfer.jl/stable/manuals/inference/delta-node/#Features-and-Supported-Inference-Scenarios)

## Best Practices

1. Use appropriate [initialization](https://reactivebayes.github.io/RxInfer.jl/stable/manuals/inference/initialization/) for models with loops
2. Monitor convergence using [Bethe Free Energy](https://reactivebayes.github.io/RxInfer.jl/stable/library/bethe-free-energy/) when available
3. Consider [factorization constraints](https://reactivebayes.github.io/RxInfer.jl/stable/manuals/constraints-specification/) for complex models
4. Use appropriate [approximation methods](https://reactivebayes.github.io/RxInfer.jl/stable/manuals/inference/delta-node/) for deterministic transformations
5. Leverage [streaming inference](https://reactivebayes.github.io/RxInfer.jl/stable/manuals/inference/streamlined/) for online data processing

## Debugging

When encountering issues:

1. Check the [Rule Not Found Error](https://reactivebayes.github.io/RxInfer.jl/stable/manuals/sharpbits/rule-not-found/) guide for common message passing errors
2. Review model constraints and initialization if inference isn't converging
3. Use visualization tools from GraphPPL.jl to inspect your factor graph
4. Enable Bethe Free Energy diagnostics to monitor convergence

## Contributing

Contributions are welcome! Please feel free to:
- Submit issues and pull requests
- Add new examples to [RxInferExamples.jl](https://reactivebayes.github.io/RxInferExamples.jl/)
- Improve documentation
- Share your use cases and experiences

## References

If you use RxInfer in your research, please cite:

- [RxInfer: A Julia package for reactive real-time Bayesian inference](https://doi.org/10.21105/joss.05161) - The reference paper for the RxInfer.jl framework
- [Reactive Probabilistic Programming for Scalable Bayesian Inference](https://pure.tue.nl/ws/portalfiles/portal/313860204/20231219_Bagaev_hf.pdf) - PhD dissertation outlining core ideas and principles
- [Variational Message Passing and Local Constraint Manipulation in Factor Graphs](https://doi.org/10.3390/e23070807) - Theoretical aspects of the underlying Bayesian inference method
- [Reactive Message Passing for Scalable Bayesian Inference](https://doi.org/10.48550/arXiv.2112.13251) - Implementation aspects and benchmarks
- [A Julia package for reactive variational Bayesian inference](https://doi.org/10.1016/j.simpa.2022.100299) - Reference paper for ReactiveMP.jl

## Ecosystem

RxInfer is part of the [ReactiveBayes](https://github.com/ReactiveBayes) ecosystem which unites three core packages:

- [ReactiveMP.jl](https://github.com/reactivebayes/ReactiveMP.jl) - Core package for efficient and scalable reactive message passing
- [GraphPPL.jl](https://github.com/reactivebayes/GraphPPL.jl) - Package for model and constraints specification  
- [Rocket.jl](https://github.com/reactivebayes/Rocket.jl) - Reactive programming tools

## Documentation

Full documentation is available at:
- [Latest Release](https://reactivebayes.github.io/RxInfer.jl/stable/)
- [Development Version](https://reactivebayes.github.io/RxInfer.jl/dev/)

## Theory and Foundations

### [[active_inference|Active Inference]] Connection

RxInfer provides powerful tools for implementing [[active_inference]] models, which combine [[perception|perception]] and [[action|action]] in a unified framework. Key capabilities include:

- **[[free_energy_principle|Free Energy Minimization]]**: Direct implementation of variational free energy minimization
- **[[policy_selection|Policy Selection]]**: Support for sophisticated policy selection through message passing
- **[[state_estimation|State Estimation]]**: Efficient inference of hidden states using variational methods
- **[[precision_weighting|Precision Estimation]]**: Automatic handling of precision parameters in hierarchical models

For mathematical details of active inference implementation, see the [[active_inference]] documentation.

### [[mathematical_framework|Mathematical Framework]]

RxInfer implements several key mathematical concepts from [[variational_inference|variational inference]] and [[active_inference|active inference]]:

- **[[message_passing|Variational Message Passing]]**:
```math
m_{f→x}(x) ∝ \exp(\mathbb{E}_{q_{∖x}}[\log f(x, θ)])
```

- **[[expected_free_energy|Expected Free Energy]]**:
```math
G(π) = \sum_τ G(π,τ)
```

- **[[policy_selection|Policy Selection]]**:
```math
P(π) = σ(-γG(π))
```

## Applications

### [[active_inference_models|Active Inference Models]]

RxInfer excels at implementing active inference models for:

1. **[[perception_learning|Perception and Learning]]**
   - [[hierarchical_models|Hierarchical state estimation]]
   - [[parameter_learning|Parameter learning]] in dynamic environments
   - [[prediction_error|Precision-weighted prediction error minimization]]

2. **[[decision_making|Decision Making]]**
   - [[policy_selection|Policy selection]] through free energy minimization
   - [[risk_sensitive_behavior|Risk-sensitive behavior modeling]]
   - [[exploration_exploitation|Exploration-exploitation trade-offs]]

3. **[[control_systems|Control Systems]]**
   - [[adaptive_control|Adaptive control]]
   - [[optimal_control|Optimal control with uncertainty]]
   - [[real_time_state_estimation|Real-time state estimation and control]]

### Example: Active Inference Agent

```julia
using RxInfer

@model function active_inference_model(observations, actions)
    # Prior preferences
    θ ~ Normal(0.0, 1.0)
    
    # Hidden states
    x = Vector{Random}(undef, length(observations))
    x[1] ~ Normal(0.0, 1.0)
    
    # State transitions and observations
    for t in 2:length(observations)
        x[t] ~ Normal(x[t-1] + actions[t-1], exp(-θ))
        observations[t] ~ Normal(x[t], 1.0)
    end
end

# Define policy selection
function select_policy(model, policies)
    G = zeros(length(policies))
    for (i, π) in enumerate(policies)
        G[i] = expected_free_energy(model, π)
    end
    return softmax(-G)
end
```

## Integration with Other Frameworks

### [[cognitive_science|Cognitive Science Tools]]

RxInfer can be integrated with:

1. **[[computational_psychiatry|Computational Psychiatry Models]]**
   - [[aberrant_precision|Aberrant precision models]]
   - [[belief_updating|Belief updating]] in psychiatric conditions
   - [[treatment_response|Treatment response prediction]]

2. **[[neuroscience|Neuroscience Applications]]**
   - [[neural_mass_models|Neural mass models]]
   - [[dynamic_causal_modeling|Dynamic causal modeling]]
   - [[brain_connectivity|Brain connectivity inference]]

3. **[[behavioral_modeling|Behavioral Modeling]]**
   - [[decision_making|Decision-making under uncertainty]]
   - [[learning_adaptation|Learning and adaptation]]
   - [[social_interaction|Social interaction models]]

### Performance Considerations

When implementing active inference models:

1. **[[computational_efficiency|Computational Efficiency]]**
   - Use streaming inference for real-time applications
   - Leverage parallel message passing when possible
   - Consider approximate inference for large-scale models

2. **[[memory_management|Memory Management]]**
   - Use appropriate buffer sizes for streaming data
   - Clear message caches when needed
   - Monitor memory usage in long-running simulations

3. **[[numerical_stability|Numerical Stability]]**
   - Use stable parameterizations
   - Monitor convergence of message passing
   - Implement appropriate numerical safeguards

## Technical Details

### [[message_passing_algorithms|Message Passing Algorithms]]

RxInfer implements several message passing algorithms:

1. **[[sum_product|Sum-Product Algorithm]]**
   - [[belief_propagation|Belief Propagation]] on trees
   - [[loopy_belief_propagation|Loopy Belief Propagation]] for cyclic graphs
   - [[structured_variational|Structured Variational]] approximations

2. **[[expectation_propagation|Expectation Propagation]]**
   - [[moment_matching|Moment Matching]] for approximate inference
   - [[natural_parameters|Natural Parameter]] updates
   - [[divergence_minimization|Divergence Minimization]]

3. **[[variational_message_passing|Variational Message Passing]]**
   ```math
   m_{x→f}(x) = \exp(\mathbb{E}_{q_{\backslash x}}[\ln p(x|pa(x))])
   ```
   where:
   - \( q_{\backslash x} \) is the approximate posterior excluding x
   - \( pa(x) \) represents parent nodes of x

### [[probabilistic_programming|Probabilistic Programming]]

#### [[model_composition|Model Composition]]

1. **[[hierarchical_models|Hierarchical Models]]**
```julia
@model function hierarchical_model(y, N, M)
    # Hyperparameters
    α ~ [[gamma_distribution|Gamma]](1.0, 1.0)
    β ~ [[gamma_distribution|Gamma]](1.0, 1.0)
    
    # Group-level parameters
    μ = Vector{Random}(undef, M)
    for m in 1:M
        μ[m] ~ [[normal_distribution|Normal]](0.0, 1/sqrt(α))
    end
    
    # Observation model
    y = Matrix{Random}(undef, N, M)
    for m in 1:M, n in 1:N
        y[n,m] ~ Normal(μ[m], 1/sqrt(β))
    end
end
```

2. **[[conjugate_models|Conjugate Models]]**
   - [[exponential_family|Exponential Family]] distributions
   - [[sufficient_statistics|Sufficient Statistics]]
   - [[natural_gradients|Natural Gradients]]

### [[inference_algorithms|Advanced Inference]]

#### [[streaming_inference|Streaming Inference Algorithms]]

1. **[[particle_filtering|Particle Filtering]]**
```julia
@model function particle_filter(observations)
    # State transition model
    x = [[state_space_model|StateSpace]](dim=3)
    x[1] ~ [[multivariate_normal|MultivariateNormal]](zeros(3), I)
    
    for t in 2:length(observations)
        x[t] ~ [[transition_model|TransitionModel]](x[t-1])
        observations[t] ~ [[measurement_model|MeasurementModel]](x[t])
    end
end
```

2. **[[kalman_filtering|Kalman Filtering]]**
   - [[linear_gaussian|Linear Gaussian]] models
   - [[extended_kalman|Extended Kalman Filter]]
   - [[unscented_kalman|Unscented Kalman Filter]]

#### [[optimization_methods|Optimization Methods]]

1. **[[natural_gradient_descent|Natural Gradient Descent]]**
```math
θ_{t+1} = θ_t - η F^{-1}(θ_t)∇L(θ_t)
```
where:
- \( F(θ) \) is the [[fisher_information|Fisher Information]] matrix
- \( L(θ) \) is the loss function
- \( η \) is the learning rate

2. **[[stochastic_optimization|Stochastic Optimization]]**
   - [[adam_optimizer|ADAM]] optimization
   - [[rmsprop|RMSprop]] variants
   - [[momentum_methods|Momentum Methods]]

### [[implementation_details|Implementation Details]]

#### [[computational_graphs|Computational Graphs]]

1. **[[graph_construction|Graph Construction]]**
```julia
@model function factor_graph_example()
    # Define variables
    x = [[variable_node|VariableNode]]("x")
    y = [[variable_node|VariableNode]]("y")
    
    # Define factors
    f1 = [[factor_node|FactorNode]]("f1", x)
    f2 = [[factor_node|FactorNode]]("f2", x, y)
    
    # Connect nodes
    connect!(f1, x)
    connect!(f2, [x, y])
end
```

2. **[[message_scheduling|Message Scheduling]]**
   - [[parallel_updates|Parallel Updates]]
   - [[sequential_updates|Sequential Updates]]
   - [[residual_scheduling|Residual-based Scheduling]]

#### [[performance_optimization|Performance Optimization]]

1. **[[memory_efficiency|Memory Efficiency]]**
   - [[message_caching|Message Caching]] strategies
   - [[buffer_management|Buffer Management]]
   - [[garbage_collection|Garbage Collection]] optimization

2. **[[parallel_computation|Parallel Computation]]**
   - [[distributed_inference|Distributed Inference]]
   - [[gpu_acceleration|GPU Acceleration]]
   - [[multi_threading|Multi-threading]] support

### [[error_handling|Error Handling and Debugging]]

1. **[[numerical_issues|Numerical Issues]]**
   - [[underflow_prevention|Underflow Prevention]]
   - [[overflow_handling|Overflow Handling]]
   - [[stability_techniques|Numerical Stability Techniques]]

2. **[[convergence_diagnostics|Convergence Diagnostics]]**
   - [[bethe_energy|Bethe Free Energy]] monitoring
   - [[kl_divergence|KL Divergence]] tracking
   - [[elbo|ELBO]] computation

The developers express deep appreciation to the entire open-source community for their tremendous efforts in developing the numerous packages that RxInfer relies upon. 