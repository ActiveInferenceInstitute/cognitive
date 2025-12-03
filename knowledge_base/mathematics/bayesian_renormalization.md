---

title: Bayesian Renormalization

type: mathematical_concept

status: stable

created: 2024-03-15

tags:

  - mathematics

  - probability

  - renormalization

  - bayesian-inference

  - statistical-physics

semantic_relations:

  - type: foundation

    links:

      - [[renormalization_group]]

      - [[probability_theory]]

      - [[bayesian_inference]]

      - [[bayes_theorem]]

  - type: implements

    links:

      - [[bayesian_networks]]

      - [[bayesian_graph_theory]]

      - [[hierarchical_models]]

  - type: related

    links:

      - [[path_integral_bayesian_inference]]

      - [[scale_free_networks]]

      - [[continuous_discrete_inference]]

      - [[bayesian_generative_models]]

---

# Bayesian Renormalization

## Overview

Bayesian Renormalization applies techniques from the [[renormalization_group|renormalization group theory]] in physics to [[bayesian_inference|Bayesian inference]], providing a powerful framework for handling multi-scale phenomena, model complexity reduction, and efficient inference in complex systems. This approach enables the systematic coarse-graining of probabilistic models while preserving their essential statistical properties.

```mermaid

graph TD

    A[Bayesian Renormalization] --> B[Theoretical Framework]

    A --> C[Renormalization Operations]

    A --> D[Hierarchical Bayesian Models]

    A --> E[Computational Methods]

    A --> F[Applications]

    A --> G[Connections to Physics]

    style A fill:#ff9999,stroke:#05386b

    style B fill:#d4f1f9,stroke:#05386b

    style C fill:#dcedc1,stroke:#05386b

    style D fill:#ffcccb,stroke:#05386b

    style E fill:#ffd580,stroke:#05386b

    style F fill:#d8bfd8,stroke:#05386b

    style G fill:#ffb6c1,stroke:#05386b

```

## Theoretical Framework

### Renormalization in Bayesian Context

The core idea of Bayesian renormalization is to transform complex, high-dimensional probability distributions into simpler, lower-dimensional ones while preserving their essential statistical properties:

```math

P^{(k+1)}(\theta^{(k+1)}) = \mathcal{R}[P^{(k)}(\theta^{(k)})]

```

where:

- $P^{(k)}$ is the probability distribution at scale $k$

- $\theta^{(k)}$ are the parameters at scale $k$

- $\mathcal{R}$ is the renormalization operator

```mermaid

flowchart TB

    A[High-Dimensional Model] -->|"Marginalization"| B[Intermediate Model]

    B -->|"Marginalization"| C[Coarse-Grained Model]

    A -->|"Direct Renormalization"| C

    D[Microscopic Parameters θ⁽⁰⁾] -->|"Renormalization Flow"| E[Mesoscopic Parameters θ⁽¹⁾]

    E -->|"Renormalization Flow"| F[Macroscopic Parameters θ⁽²⁾]

    style A fill:#d4f1f9,stroke:#05386b

    style B fill:#dcedc1,stroke:#05386b

    style C fill:#ffcccb,stroke:#05386b

    style D fill:#ffd580,stroke:#05386b

    style E fill:#d8bfd8,stroke:#05386b

    style F fill:#ffb6c1,stroke:#05386b

```

### Scale Transformation and Fixed Points

The Bayesian model undergoes scale transformations governed by renormalization flow equations:

```math

\frac{d\theta}{dl} = \beta(\theta)

```

where:

- $l$ is the logarithmic scale parameter

- $\beta(\theta)$ is the beta function determining the flow

Fixed points $\theta^*$ where $\beta(\theta^*) = 0$ represent scale-invariant models with universal properties.

## Renormalization Operations

```mermaid

graph TB

    A[Renormalization Operations] --> B[Decimation]

    A --> C[Block-Spin Transformation]

    A --> D[Hierarchical Coarse-Graining]

    A --> E[Variational Renormalization]

    B --> B1[Parameter Space Reduction]

    B --> B2[Degrees of Freedom Elimination]

    C --> C1[Local Variable Grouping]

    C --> C2[Effective Interaction Creation]

    D --> D1[Multi-level Representation]

    D --> D2[Scale Hierarchy]

    E --> E1[Variational Approximation]

    E --> E2[KL Minimization]

    style A fill:#d4f1f9,stroke:#05386b

    style B fill:#dcedc1,stroke:#05386b

    style C fill:#ffcccb,stroke:#05386b

    style D fill:#ffd580,stroke:#05386b

    style E fill:#d8bfd8,stroke:#05386b

```

### 1. Decimation in Bayesian Networks

In a [[bayesian_networks|Bayesian network]], decimation involves marginalizing over selected variables to create a simpler network:

```python

def decimate_bayesian_network(network, variables_to_eliminate):

    """

    Perform decimation on a Bayesian network by marginalizing over variables

    """

    reduced_network = network.copy()

    for variable in variables_to_eliminate:

        # Identify the Markov blanket of the variable

        markov_blanket = get_markov_blanket(reduced_network, variable)

        # Marginalize over the variable

        marginalize_variable(reduced_network, variable, markov_blanket)

        # The network now contains effective interactions between 

        # members of the Markov blanket

    return reduced_network

```

### 2. Block-Spin Transformation

The block-spin transformation groups variables together to form new "effective" variables:

```math

X^{(k+1)}_i = \mathcal{B}(\{X^{(k)}_j | j \in \mathcal{N}(i)\})

```

where $\mathcal{B}$ is a blocking function and $\mathcal{N}(i)$ is a neighborhood in the original model.

### 3. Multi-scale Representation

```mermaid

graph TB

    subgraph "Scale 0 (Fine)"

        A1[X₁] --- A2[X₂]

        A2 --- A3[X₃]

        A3 --- A4[X₄]

        A4 --- A1

        A1 --- A3

        A2 --- A4

    end

    subgraph "Scale 1 (Intermediate)"

        B1[Y₁] --- B2[Y₂]

        B1 --- B2

    end

    subgraph "Scale 2 (Coarse)"

        C1[Z₁]

    end

    A1 & A2 -->|Block| B1

    A3 & A4 -->|Block| B2

    B1 & B2 -->|Block| C1

    style A1 fill:#d4f1f9,stroke:#05386b

    style A2 fill:#d4f1f9,stroke:#05386b

    style A3 fill:#d4f1f9,stroke:#05386b

    style A4 fill:#d4f1f9,stroke:#05386b

    style B1 fill:#dcedc1,stroke:#05386b

    style B2 fill:#dcedc1,stroke:#05386b

    style C1 fill:#ffcccb,stroke:#05386b

```

## Hierarchical Bayesian Models

Hierarchical Bayesian models naturally embody renormalization group ideas through their multi-level structure:

```mermaid

flowchart TB

    A[Global Parameters θ] --> B[Group-Level Parameters φ₁]

    A --> C[Group-Level Parameters φ₂]

    A --> D[Group-Level Parameters φ₃]

    B --> B1[Local Parameters ψ₁₁]

    B --> B2[Local Parameters ψ₁₂]

    C --> C1[Local Parameters ψ₂₁]

    C --> C2[Local Parameters ψ₂₂]

    D --> D1[Local Parameters ψ₃₁]

    D --> D2[Local Parameters ψ₃₂]

    B1 --> B1a[Data X₁₁]

    B2 --> B2a[Data X₁₂]

    C1 --> C1a[Data X₂₁]

    C2 --> C2a[Data X₂₂]

    D1 --> D1a[Data X₃₁]

    D2 --> D2a[Data X₃₂]

    style A fill:#d4f1f9,stroke:#05386b

    style B fill:#dcedc1,stroke:#05386b

    style C fill:#dcedc1,stroke:#05386b

    style D fill:#dcedc1,stroke:#05386b

```

### 1. Hierarchical Model Structure

In a hierarchical Bayesian model, parameters at higher levels govern the behavior of parameters at lower levels, forming a natural renormalization structure:

```math

P(\theta, \phi, \psi, X) = P(\theta)P(\phi|\theta)P(\psi|\phi)P(X|\psi)

```

### 2. Deep Belief Networks

Deep belief networks implement renormalization principles with their layered structure:

```julia

struct DeepBeliefNetwork

    layers::Vector{RestrictedBoltzmannMachine}

    function DeepBeliefNetwork(layer_sizes)

        # Create a stack of RBMs

        layers = []

        for i in 1:length(layer_sizes)-1

            push!(layers, RestrictedBoltzmannMachine(layer_sizes[i], layer_sizes[i+1]))

        end

        new(layers)

    end

end

function renormalization_flow(model::DeepBeliefNetwork, data)

    """

    Visualize the renormalization flow through network layers

    """

    activations = [data]

    current_data = data

    # Forward pass through each layer

    for layer in model.layers

        current_data = propagate_forward(layer, current_data)

        push!(activations, current_data)

    end

    return activations

end

```

### 3. Scale-Invariant Priors

Certain prior distributions exhibit scale-invariance properties that align with renormalization principles:

```math

P(\theta) \propto \frac{1}{|\theta|}

```

This scale-invariant (Jeffreys) prior remains unchanged under rescaling transformations.

## Computational Methods

```mermaid

graph TB

    A[Computational Methods] --> B[Monte Carlo Renormalization]

    A --> C[Variational Renormalization]

    A --> D[Real-Space Renormalization]

    B --> B1[Partial Gibbs Sampling]

    B --> B2[Block Sampling]

    C --> C1[Variational Mean-Field]

    C --> C2[Renormalized Variational Inference]

    D --> D1[Tree Tensor Networks]

    D --> D2[Wavelet Transformation]

    style A fill:#d4f1f9,stroke:#05386b

    style B fill:#dcedc1,stroke:#05386b

    style C fill:#ffcccb,stroke:#05386b

    style D fill:#ffd580,stroke:#05386b

```

### 1. Monte Carlo Renormalization Group

Monte Carlo methods can implement renormalization by first sampling from a fine-grained model and then aggregating samples to represent coarse-grained behavior:

```python

def monte_carlo_renormalization(fine_model, block_size, num_samples):

    """

    Perform Monte Carlo Renormalization Group

    """

    # Generate samples from the fine-grained model

    fine_samples = mcmc_sample(fine_model, num_samples)

    # Transform samples using blocking transformation

    coarse_samples = []

    for sample in fine_samples:

        # Reshape sample into blocks

        blocks = reshape_into_blocks(sample, block_size)

        # Apply block transformation (e.g., majority rule)

        coarse_sample = apply_block_transformation(blocks)

        coarse_samples.append(coarse_sample)

    # Fit a new model to the coarse-grained samples

    coarse_model = fit_model_to_samples(coarse_samples)

    return coarse_model

```

### 2. Variational Renormalization Group

Variational methods implement renormalization by optimizing a simpler model to approximate a more complex one:

```math

\mathcal{F}[P_{\text{simple}}] = \text{KL}(P_{\text{simple}} || P_{\text{complex}}) - \text{entropy}[P_{\text{simple}}]

```

### 3. Real-Space Renormalization

Real-space renormalization directly transforms the model structure in physical or latent space:

```mermaid

sequenceDiagram

    participant F as Fine-grained Model

    participant I as Intermediate Model

    participant C as Coarse-grained Model

    F->>F: Define blocking transformation

    F->>I: Apply first level of coarse-graining

    I->>I: Compute effective parameters

    I->>C: Apply second level of coarse-graining

    C->>C: Compute final effective parameters

    Note over F,C: Analyze flow of parameters

```

## Applications

```mermaid

mindmap

  root((Bayesian<br>Renormalization))

    Complex Systems

      Phase Transitions

      Critical Phenomena

      Universality Classes

    Machine Learning

      Deep Hierarchical Models

      Representation Learning

      Feature Extraction

    Statistical Inference

      Model Selection

      Complexity Reduction

      Efficient Sampling

    Network Analysis

      Scale-Free Networks

      Community Detection

      Network Coarse-Graining

    Computational Physics

      Material Properties

      Quantum Many-Body Systems

      Effective Field Theories

```

### 1. Image Processing and Computer Vision

Bayesian renormalization provides a framework for multi-scale image analysis, object recognition, and scene understanding:

```python

def image_renormalization(image, levels=3):

    """

    Perform multi-scale Bayesian analysis of an image

    """

    # Initialize pyramid

    image_pyramid = [image]

    feature_hierarchy = []

    # Generate image pyramid

    current_image = image

    for _ in range(levels):

        # Apply coarse-graining (e.g., 2×2 block averaging)

        coarse_image = block_average(current_image, block_size=2)

        image_pyramid.append(coarse_image)

        current_image = coarse_image

    # Bottom-up feature extraction

    for level_idx, level_image in enumerate(image_pyramid):

        # Extract features at this level

        level_features = extract_features(level_image)

        # Infer latent variables using Bayesian methods

        if level_idx > 0:

            # Use information from lower level

            level_features = combine_with_lower_level(

                level_features, feature_hierarchy[-1])

        feature_hierarchy.append(level_features)

    # Top-down refinement (optional)

    refined_features = top_down_refine(feature_hierarchy)

    return refined_features

```

### 2. Biological Network Analysis

Bayesian renormalization helps identify functional modules in complex biological networks:

```math

P(z|A) \propto P(A|z)P(z)

```

where:

- $A$ is the network adjacency matrix

- $z$ is the node community assignment

- The renormalization process identifies communities at different scales

### 3. Physics-Inspired Machine Learning

Integrating renormalization group ideas into Bayesian neural networks allows for effective model compression and transfer learning.

## Connections to Physics

```mermaid

graph TB

    A[Bayesian Renormalization] --- B[Statistical Mechanics]

    A --- C[Quantum Field Theory]

    A --- D[Condensed Matter Physics]

    B --- B1[Critical Phenomena]

    B --- B2[Phase Transitions]

    C --- C1[Effective Field Theories]

    C --- C2[Wilsonian Renormalization]

    D --- D1[Many-Body Problems]

    D --- D2[Universality Classes]

    style A fill:#d4f1f9,stroke:#05386b

    style B fill:#dcedc1,stroke:#05386b

    style C fill:#ffcccb,stroke:#05386b

    style D fill:#ffd580,stroke:#05386b

```

### 1. Universal Behavior Near Critical Points

Just as physical systems exhibit universal behavior near critical points, complex Bayesian models can display universal properties when their parameters approach critical values:

```math

P(x) \sim |x - x_c|^{-\beta}

```

where $\beta$ is a universal critical exponent determined by the system's universality class.

### 2. Fixed Points of Renormalization Flow

The renormalization flow in parameter space reveals fixed points corresponding to distinct phases of the model's behavior:

```math

\frac{d\theta}{dl} = \beta(\theta) \approx \sum_i c_i (\theta - \theta^*)^i

```

The linearization around fixed points yields critical exponents that characterize universal behavior.

### 3. Information-Theoretic Perspective

```mermaid

flowchart LR

    A[Information Content] --> B{Renormalization}

    B -->|"Relevant Information"| C[Preserved]

    B -->|"Irrelevant Information"| D[Discarded]

    style A fill:#d4f1f9,stroke:#05386b

    style B fill:#ffcccb,stroke:#05386b

    style C fill:#dcedc1,stroke:#05386b

    style D fill:#ffd580,stroke:#05386b

```

Bayesian renormalization can be viewed as information compression, preserving relevant statistical information while discarding irrelevant details.

## Advanced Topics

### 1. Non-equilibrium Renormalization

Extends renormalization to time-dependent Bayesian inference problems:

```math

\frac{\partial P(x,t)}{\partial t} = \mathcal{L}[P(x,t)]

```

where $\mathcal{L}$ is a time-evolution operator.

### 2. Functional Renormalization Group

Applies renormalization to entire probability functionals:

```math

\frac{d\Gamma_k[P]}{dk} = \frac{1}{2}\text{Tr}\left[\left(\frac{\delta^2\Gamma_k}{\delta P \delta P} + R_k\right)^{-1}\frac{dR_k}{dk}\right]

```

where $\Gamma_k$ is the effective action and $R_k$ is a scale-dependent regulator.

### 3. Quantum Bayesian Renormalization

```mermaid

graph LR

    A[Quantum Measurement] --> B[Continuous Monitoring]

    B --> C[Recursive Filtering]

    C --> D[Quantum Renormalization]

    style A fill:#d4f1f9,stroke:#05386b

    style B fill:#dcedc1,stroke:#05386b

    style C fill:#ffcccb,stroke:#05386b

    style D fill:#ffd580,stroke:#05386b

```

Extends Bayesian renormalization to quantum systems, connecting quantum measurement theory with renormalization group concepts.

## References

1. Beny, C., & Osborne, T. J. (2015). The renormalization group via statistical inference. New Journal of Physics, 17(8), 083005.

1. Mehta, P., & Schwab, D. J. (2014). An exact mapping between the variational renormalization group and deep learning. arXiv preprint arXiv:1410.3831.

1. Koch-Janusz, M., & Ringel, Z. (2018). Mutual information, neural networks and the renormalization group. Nature Physics, 14(6), 578-582.

1. Lin, H. W., Tegmark, M., & Rolnick, D. (2017). Why does deep and cheap learning work so well? Journal of Statistical Physics, 168(6), 1223-1247.

1. Williams, M. J., Graves, T., Reeves, B., Hauert, S., & Yakovenko, V. M. (2022). Hierarchical Bayesian Renormalization Group: Application to financial market crashes. arXiv preprint arXiv:2201.01259.

