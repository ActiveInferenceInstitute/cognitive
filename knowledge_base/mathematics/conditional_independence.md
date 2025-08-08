---

title: Conditional Independence

type: concept

status: stable

created: 2024-03-20

tags:

  - mathematics

  - probability

  - statistics

  - graphical-models

semantic_relations:

  - type: foundation

    links:

      - [[probability_theory]]

      - [[information_theory]]

      - [[graphical_models]]

  - type: used_by

    links:

      - [[bayesian_networks]]

      - [[markov_random_fields]]

      - [[probabilistic_graphical_models]]

  - type: related

    links:

      - [[markov_properties]]

      - [[d_separation]]

      - [[independence_testing]]

---

# Conditional Independence

## Overview

Conditional Independence is a fundamental concept in probability theory and graphical models that describes when knowledge about one variable provides no additional information about another variable, given that we already know the value of a third variable.

## Mathematical Definition

### 1. Basic Definition

For random variables X, Y, and Z:

```math

X \perp Y | Z \iff P(X|Y,Z) = P(X|Z)

```

where:

- X ⊥ Y | Z denotes X is conditionally independent of Y given Z

- P(X|Y,Z) is the conditional probability of X given Y and Z

- P(X|Z) is the conditional probability of X given Z

### 2. Alternative Formulations

#### Product Form

```math

P(X,Y|Z) = P(X|Z)P(Y|Z)

```

#### Ratio Form

```math

\frac{P(X|Y,Z)}{P(X|Z)} = 1

```

### 3. [[information_theoretic|Information Theoretic]] Perspective

```math

I(X;Y|Z) = 0

```

where I(X;Y|Z) is the conditional mutual information.

## Properties

### 1. [[symmetry|Symmetry]]

```math

X \perp Y | Z \iff Y \perp X | Z

```

### 2. [[decomposition|Decomposition]]

```math

X \perp (Y,W) | Z \implies X \perp Y | Z \text{ and } X \perp W | Z

```

### 3. [[weak_union|Weak Union]]

```math

X \perp (Y,W) | Z \implies X \perp Y | (Z,W)

```

### 4. [[contraction|Contraction]]

```math

(X \perp Y | Z) \text{ and } (X \perp W | (Y,Z)) \implies X \perp (Y,W) | Z

```

## Implementation

### 1. Testing for Conditional Independence

```julia

function test_conditional_independence(data::Matrix{Float64},

                                    X::Vector{Int},

                                    Y::Vector{Int},

                                    Z::Vector{Int};

                                    method::Symbol=:partial_correlation)

    if method == :partial_correlation

        return test_gaussian_ci(data, X, Y, Z)

    elseif method == :chi_square

        return test_discrete_ci(data, X, Y, Z)

    elseif method == :kernel

        return test_nonparametric_ci(data, X, Y, Z)

    else

        error("Unknown testing method")

    end

end

```

### 2. Gaussian Case

```julia

function test_gaussian_ci(data::Matrix{Float64},

                         X::Vector{Int},

                         Y::Vector{Int},

                         Z::Vector{Int})

    # Compute partial correlation

    Σ = cov(data)

    ρ_xy_z = partial_correlation(Σ, X, Y, Z)

    # Compute test statistic

    n = size(data, 1)

    t_stat = ρ_xy_z * sqrt((n - length(Z) - 2)/(1 - ρ_xy_z^2))

    # Compute p-value

    p_value = 2 * (1 - cdf(TDist(n - length(Z) - 2), abs(t_stat)))

    return p_value < 0.05

end

function partial_correlation(Σ::Matrix{Float64},

                          X::Vector{Int},

                          Y::Vector{Int},

                          Z::Vector{Int})

    # Compute precision matrix

    Λ = inv(Σ)

    # Extract relevant elements

    ρ = -Λ[X,Y] / sqrt(Λ[X,X] * Λ[Y,Y])

    return ρ

end

```

### 3. Discrete Case

```julia

function test_discrete_ci(data::Matrix{Float64},

                         X::Vector{Int},

                         Y::Vector{Int},

                         Z::Vector{Int})

    # Compute contingency tables

    joint_counts = compute_contingency(data, [X; Y; Z])

    marg_counts = compute_contingency(data, [X; Z])

    # Compute chi-square statistic

    χ2 = compute_chi_square(joint_counts, marg_counts)

    # Compute degrees of freedom

    df = (prod(size(joint_counts)) - 1) - 

         (prod(size(marg_counts)) - 1)

    # Compute p-value

    p_value = 1 - cdf(Chisq(df), χ2)

    return p_value < 0.05

end

```

## Applications in Graphical Models

### 1. [[bayesian_networks|Bayesian Networks]]

#### D-separation

```julia

function d_separated(graph::BayesianNetwork,

                    X::Node,

                    Y::Node,

                    Z::Set{Node})

    # Find all paths between X and Y

    paths = find_paths(graph, X, Y)

    # Check if all paths are blocked by Z

    return all(is_blocked(path, Z) for path in paths)

end

```

### 2. [[markov_random_fields|Markov Random Fields]]

#### Global Markov Property

```julia

function global_markov(graph::MarkovRandomField,

                      A::Set{Node},

                      B::Set{Node},

                      S::Set{Node})

    # Check if S separates A and B

    return is_separated(graph, A, B, S)

end

```

### 3. [[factor_graphs|Factor Graphs]]

#### Message Passing

```julia

function factor_to_variable_message(factor::Factor,

                                  variable::Variable,

                                  messages::Dict{Tuple{Node,Node}, Message})

    # If variables are conditionally independent

    if are_conditionally_independent(factor, variable)

        # Simplify message computation

        return simplified_message(factor, variable, messages)

    else

        # Full message computation

        return compute_full_message(factor, variable, messages)

    end

end

```

## Testing and Validation

### 1. [[independence_testing|Statistical Tests]]

#### Correlation-based Tests

```julia

function correlation_test(X::Vector{Float64},

                        Y::Vector{Float64},

                        Z::Vector{Float64})

    # Compute partial correlation

    ρ_xy_z = partial_correlation_coefficient(X, Y, Z)

    # Test significance

    return test_significance(ρ_xy_z, length(X))

end

```

#### Information-based Tests

```julia

function mutual_information_test(X::Vector{Float64},

                               Y::Vector{Float64},

                               Z::Vector{Float64})

    # Estimate conditional mutual information

    I_xy_z = estimate_conditional_mi(X, Y, Z)

    # Test significance

    return test_significance(I_xy_z, length(X))

end

```

### 2. [[robustness|Robustness Considerations]]

```julia

function robust_ci_test(X::Vector{Float64},

                       Y::Vector{Float64},

                       Z::Vector{Float64};

                       n_bootstrap::Int=1000,

                       α::Float64=0.05)

    # Bootstrap samples

    results = zeros(Bool, n_bootstrap)

    for i in 1:n_bootstrap

        # Resample data

        indices = rand(1:length(X), length(X))

        # Test CI on bootstrap sample

        results[i] = test_conditional_independence(

            X[indices], Y[indices], Z[indices]

        )

    end

    # Compute confidence

    confidence = mean(results)

    return confidence > (1 - α)

end

```

## Best Practices

### 1. Testing Strategy

- Choose appropriate test for data type

- Consider sample size requirements

- Account for multiple testing

- Validate assumptions

### 2. Implementation

- Use numerically stable methods

- Handle missing data appropriately

- Implement efficient algorithms

- Cache intermediate results

### 3. Validation

- Cross-validate results

- Use multiple test statistics

- Consider effect sizes

- Report confidence intervals

## References

1. Pearl, J. (1988). Probabilistic Reasoning in Intelligent Systems

1. Dawid, A. P. (1979). Conditional Independence in Statistical Theory

1. Koller, D., & Friedman, N. (2009). Probabilistic Graphical Models

1. Spirtes, P., Glymour, C., & Scheines, R. (2000). Causation, Prediction, and Search

1. Edwards, D. (2000). Introduction to Graphical Modelling

