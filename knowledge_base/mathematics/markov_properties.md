---

title: Markov Properties

type: concept

status: stable

created: 2024-03-20

tags:

  - mathematics

  - probability

  - graphical-models

  - markov-theory

semantic_relations:

  - type: foundation

    links:

      - [[probability_theory]]

      - [[graph_theory]]

      - [[conditional_independence]]

  - type: used_by

    links:

      - [[markov_random_fields]]

      - [[bayesian_networks]]

      - [[probabilistic_graphical_models]]

  - type: related

    links:

      - [[hammersley_clifford_theorem]]

      - [[gibbs_distributions]]

      - [[graphical_separation]]

---

# Markov Properties

## Overview

Markov Properties are fundamental principles in probabilistic graphical models that characterize conditional independence relationships through graph structure. They provide the connection between the topology of a graph and the factorization of the associated probability distribution.

## Core Properties

### 1. Local Markov Property

```math

X_i \perp X_V\backslash cl(i) | X_{ne(i)}

```

where:

- X_i is a random variable

- V is the set of all variables

- cl(i) is the closure of i (i and its neighbors)

- ne(i) is the set of neighbors of i

### 2. Global Markov Property

```math

X_A \perp X_B | X_S

```

where:

- A, B are sets of nodes

- S separates A and B in the graph

### 3. Pairwise Markov Property

```math

(i,j) \notin E \implies X_i \perp X_j | X_{V\backslash\{i,j\}}

```

where:

- E is the set of edges

- V is the set of all variables

## Mathematical Framework

### 1. [[graph_separation|Graph Separation]]

```julia

struct GraphSeparation

    graph::Graph

    function is_separated(A::Set{Node},

                         B::Set{Node},

                         S::Set{Node})

        # Remove separator nodes

        subgraph = remove_nodes(graph, S)

        # Check connectivity

        return !is_connected(subgraph, A, B)

    end

end

```

### 2. [[markov_blanket|Markov Blanket]]

```julia

function markov_blanket(graph::Graph, node::Node)

    blanket = Set{Node}()

    # Add neighbors

    union!(blanket, neighbors(graph, node))

    if isa(graph, DirectedGraph)

        # Add children's parents for directed graphs

        for child in children(graph, node)

            union!(blanket, parents(graph, child))

        end

    end

    return blanket

end

```

### 3. [[factorization|Factorization Properties]]

```julia

function check_factorization(distribution::Distribution,

                           graph::Graph)

    # For undirected graphs

    if isa(graph, UndirectedGraph)

        return check_clique_factorization(distribution, graph)

    else

        # For directed graphs

        return check_parent_factorization(distribution, graph)

    end

end

```

## Implementations

### 1. [[local_markov|Local Markov]] Verification

```julia

function verify_local_markov(graph::Graph,

                           distribution::Distribution,

                           node::Node)

    # Get Markov blanket

    blanket = markov_blanket(graph, node)

    # Get remaining variables

    rest = setdiff(Set(nodes(graph)), union(Set([node]), blanket))

    # Test conditional independence

    return test_conditional_independence(

        distribution,

        node,

        collect(rest),

        collect(blanket)

    )

end

```

### 2. [[global_markov|Global Markov]] Verification

```julia

function verify_global_markov(graph::Graph,

                            distribution::Distribution)

    # Test all possible separations

    for (A, B, S) in generate_separations(graph)

        if is_separated(graph, A, B, S)

            # Test conditional independence

            if !test_conditional_independence(

                distribution, A, B, S)

                return false

            end

        end

    end

    return true

end

```

### 3. [[pairwise_markov|Pairwise Markov]] Verification

```julia

function verify_pairwise_markov(graph::Graph,

                              distribution::Distribution)

    for i in nodes(graph)

        for j in nodes(graph)

            if i != j && !has_edge(graph, i, j)

                # Get all other variables

                others = setdiff(Set(nodes(graph)), Set([i, j]))

                # Test conditional independence

                if !test_conditional_independence(

                    distribution, i, j, collect(others))

                    return false

                end

            end

        end

    end

    return true

end

```

## Applications

### 1. [[undirected_models|Undirected Models]]

#### Markov Random Fields

```julia

function create_mrf(graph::UndirectedGraph)

    # Create potential functions for cliques

    potentials = Dict{Set{Node}, Function}()

    for clique in find_maximal_cliques(graph)

        potentials[clique] = create_clique_potential(clique)

    end

    return MarkovRandomField(graph, potentials)

end

```

### 2. [[directed_models|Directed Models]]

#### Bayesian Networks

```julia

function create_bayesian_network(graph::DirectedGraph)

    # Create conditional probability tables

    cpts = Dict{Node, CPT}()

    for node in topological_sort(graph)

        parents = get_parents(graph, node)

        cpts[node] = create_cpt(node, parents)

    end

    return BayesianNetwork(graph, cpts)

end

```

### 3. [[inference_algorithms|Inference Algorithms]]

```julia

function belief_propagation(graph::Graph,

                          query::Node,

                          evidence::Dict{Node, Any})

    # Initialize messages

    messages = initialize_messages(graph)

    # Message passing schedule based on Markov properties

    schedule = create_schedule(graph, query)

    # Run belief propagation

    for (source, target) in schedule

        messages[(source, target)] = compute_message(

            graph, source, target, messages

        )

    end

    return compute_belief(query, messages)

end

```

## Theoretical Results

### 1. [[equivalence_theorems|Equivalence Theorems]]

```julia

function verify_markov_equivalence(graph::Graph)

    # Check equivalence of Markov properties

    local_global = local_implies_global(graph)

    global_pairwise = global_implies_pairwise(graph)

    pairwise_local = pairwise_implies_local(graph)

    return local_global && global_pairwise && pairwise_local

end

```

### 2. [[separation_criteria|Separation Criteria]]

```julia

function verify_separation_criterion(graph::Graph,

                                  A::Set{Node},

                                  B::Set{Node},

                                  S::Set{Node})

    if isa(graph, DirectedGraph)

        return verify_d_separation(graph, A, B, S)

    else

        return verify_graph_separation(graph, A, B, S)

    end

end

```

### 3. [[hammersley_clifford|Hammersley-Clifford Theorem]]

```julia

function verify_hammersley_clifford(distribution::Distribution,

                                  graph::UndirectedGraph)

    # Check positivity

    if !is_positive(distribution)

        return false

    end

    # Check factorization

    return can_factorize_over_cliques(distribution, graph)

end

```

## Best Practices

### 1. Model Design

- Choose appropriate Markov property

- Consider computational implications

- Validate independence assumptions

- Design efficient graph structure

### 2. Implementation

- Use efficient graph algorithms

- Implement numerical safeguards

- Cache separation results

- Optimize independence tests

### 3. Validation

- Test all Markov properties

- Verify factorization

- Check separation criteria

- Validate probability computations

## References

1. Lauritzen, S. L. (1996). Graphical Models

1. Pearl, J. (1988). Probabilistic Reasoning in Intelligent Systems

1. Koller, D., & Friedman, N. (2009). Probabilistic Graphical Models

1. Whittaker, J. (1990). Graphical Models in Applied Multivariate Statistics

1. Edwards, D. (2000). Introduction to Graphical Modelling

