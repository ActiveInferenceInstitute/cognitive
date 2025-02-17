---
type: concept
id: population_genetics_001
created: 2024-03-15
modified: 2024-03-15
tags: [genetics, evolution, mathematical-biology, complex-systems]
aliases: [pop-gen, population-genetics-theory]
complexity: advanced
processing_priority: 1
semantic_relations:
  - type: foundation
    links:
      - [[evolutionary_dynamics]]
      - [[molecular_evolution]]
      - [[quantitative_genetics]]
  - type: implements
    links:
      - [[hardy_weinberg_equilibrium]]
      - [[selection_theory]]
      - [[genetic_drift]]
  - type: relates
    links:
      - [[ecological_genetics]]
      - [[molecular_biology]]
      - [[statistical_genetics]]
---

# Population Genetics

## Overview

Population genetics provides the mathematical foundation for understanding how genetic variation changes in populations over time. It integrates principles from genetics, evolution, and statistical physics to model the dynamics of allele frequencies and genetic diversity.

## Mathematical Framework

### 1. Hardy-Weinberg Equilibrium

Basic equations of genetic equilibrium:

```math
\begin{aligned}
& \text{Genotype Frequencies:} \\
& P(AA) = p^2, P(Aa) = 2pq, P(aa) = q^2 \\
& \text{Allele Frequencies:} \\
& p + q = 1 \\
& \text{Equilibrium Test:} \\
& \chi^2 = \sum_i \frac{(O_i - E_i)^2}{E_i}
\end{aligned}
```

### 2. Selection Dynamics

Selection equations:

```math
\begin{aligned}
& \text{Directional Selection:} \\
& \Delta p = \frac{sp(1-p)}{1 + s(2p-1)} \\
& \text{Balancing Selection:} \\
& \dot{p} = sp(1-p)(h - p) \\
& \text{Frequency-Dependent:} \\
& w_i = \alpha_i + \sum_j \beta_{ij}p_j
\end{aligned}
```

### 3. Genetic Drift

Stochastic processes:

```math
\begin{aligned}
& \text{Wright-Fisher Model:} \\
& P(i \to j) = \binom{2N}{j}\left(\frac{i}{2N}\right)^j\left(1-\frac{i}{2N}\right)^{2N-j} \\
& \text{Coalescent Time:} \\
& \mathbb{E}[T_{MRCA}] = 4N_e \\
& \text{Fixation Probability:} \\
& u(p) = \frac{1-e^{-4N_esp}}{1-e^{-4N_es}}
\end{aligned}
```

## Implementation Framework

### 1. Population Simulator

```python
class PopulationGenetics:
    """Simulates population genetic processes"""
    def __init__(self,
                 population_size: int,
                 n_loci: int,
                 mutation_rate: float,
                 selection_coefficients: np.ndarray):
        self.N = population_size
        self.L = n_loci
        self.mu = mutation_rate
        self.s = selection_coefficients
        self.population = self.initialize_population()
        
    def simulate_generation(self) -> np.ndarray:
        """Simulate one generation"""
        # Selection
        fitness = self.compute_fitness()
        selected = self.selection_step(fitness)
        
        # Reproduction
        offspring = self.reproduction_step(selected)
        
        # Mutation
        mutated = self.mutation_step(offspring)
        
        # Genetic drift
        drifted = self.drift_step(mutated)
        
        self.population = drifted
        return self.compute_allele_frequencies()
        
    def compute_fitness(self) -> np.ndarray:
        """Compute fitness of individuals"""
        return np.exp(self.population @ self.s)
        
    def selection_step(self,
                      fitness: np.ndarray) -> np.ndarray:
        """Apply natural selection"""
        probabilities = fitness / np.sum(fitness)
        selected_indices = np.random.choice(
            self.N,
            size=self.N,
            p=probabilities
        )
        return self.population[selected_indices]
        
    def reproduction_step(self,
                         selected: np.ndarray) -> np.ndarray:
        """Simulate sexual reproduction"""
        offspring = []
        for i in range(0, self.N, 2):
            parent1 = selected[i]
            parent2 = selected[i+1]
            child1, child2 = self.recombine(parent1, parent2)
            offspring.extend([child1, child2])
        return np.array(offspring)
        
    def mutation_step(self,
                     offspring: np.ndarray) -> np.ndarray:
        """Apply mutations"""
        mutation_mask = np.random.random(offspring.shape) < self.mu
        return np.where(mutation_mask, 1-offspring, offspring)
        
    def drift_step(self,
                  population: np.ndarray) -> np.ndarray:
        """Simulate genetic drift"""
        if self.N < len(population):
            indices = np.random.choice(
                len(population),
                size=self.N,
                replace=False
            )
            return population[indices]
        return population
```

### 2. Coalescent Simulator

```python
class CoalescentSimulator:
    """Simulates genealogical processes"""
    def __init__(self,
                 sample_size: int,
                 effective_size: int,
                 mutation_rate: float):
        self.n = sample_size
        self.Ne = effective_size
        self.mu = mutation_rate
        
    def simulate_genealogy(self) -> Tree:
        """Simulate coalescent genealogy"""
        # Initialize lineages
        lineages = [Node(i) for i in range(self.n)]
        time = 0
        
        while len(lineages) > 1:
            # Compute coalescence rate
            k = len(lineages)
            rate = k * (k-1) / (4 * self.Ne)
            
            # Time until next coalescence
            time += np.random.exponential(1/rate)
            
            # Choose lineages to coalesce
            i, j = np.random.choice(k, size=2, replace=False)
            
            # Create new node
            new_node = Node(time)
            new_node.add_child(lineages[i])
            new_node.add_child(lineages[j])
            
            # Update lineages
            lineages = [l for idx, l in enumerate(lineages)
                       if idx not in (i,j)]
            lineages.append(new_node)
            
        return Tree(lineages[0])
        
    def add_mutations(self,
                     tree: Tree) -> Tree:
        """Add mutations to genealogy"""
        for node in tree.traverse():
            # Branch length
            t = node.dist
            
            # Number of mutations
            n_mutations = np.random.poisson(self.mu * t)
            
            # Add mutations
            for _ in range(n_mutations):
                position = np.random.random()
                node.add_mutation(position)
                
        return tree
```

### 3. Selection Analyzer

```python
class SelectionAnalyzer:
    """Analyzes selection patterns"""
    def __init__(self):
        self.neutrality = NeutralityTests()
        self.divergence = DivergenceAnalysis()
        self.linkage = LinkageAnalysis()
        
    def analyze_selection(self,
                         sequences: List[str],
                         outgroup: str = None) -> Dict:
        """Analyze selection patterns"""
        # Neutrality tests
        neutrality_stats = self.neutrality.compute_statistics(
            sequences)
            
        # Divergence analysis
        if outgroup:
            divergence_stats = self.divergence.compute_statistics(
                sequences, outgroup)
        else:
            divergence_stats = None
            
        # Linkage analysis
        linkage_stats = self.linkage.compute_statistics(
            sequences)
            
        return {
            'neutrality': neutrality_stats,
            'divergence': divergence_stats,
            'linkage': linkage_stats
        }
```

## Advanced Concepts

### 1. Mutation-Selection Balance

```math
\begin{aligned}
& \text{Equilibrium Frequency:} \\
& \hat{q} = \frac{\mu}{hs} \text{ (dominant)} \\
& \hat{q} = \sqrt{\frac{\mu}{s}} \text{ (recessive)} \\
& \text{Genetic Load:} \\
& L = 1 - \bar{w} = \sum_i \mu_i
\end{aligned}
```

### 2. Linkage Disequilibrium

```math
\begin{aligned}
& \text{LD Measure:} \\
& D = p_{AB} - p_Ap_B \\
& \text{Standardized LD:} \\
& D' = \frac{D}{\min(p_Ap_b, p_aP_B)} \\
& \text{LD Decay:} \\
& D_t = D_0(1-r)^t
\end{aligned}
```

### 3. Population Structure

```math
\begin{aligned}
& \text{F-Statistics:} \\
& F_{ST} = \frac{\sigma^2_p}{\bar{p}(1-\bar{p})} \\
& \text{Migration Rate:} \\
& F_{ST} \approx \frac{1}{4N_em + 1} \\
& \text{Isolation by Distance:} \\
& F_{ST} \propto \frac{1}{1 + 4\pi D\sigma^2t}
\end{aligned}
```

## Applications

### 1. Conservation Genetics
- Effective population size
- Inbreeding depression
- Genetic rescue

### 2. Medical Genetics
- Disease allele frequencies
- Genetic risk prediction
- Drug resistance evolution

### 3. Molecular Evolution
- Neutral theory
- Adaptive evolution
- Molecular clocks

## Advanced Mathematical Extensions

### 1. Diffusion Approximation

```math
\begin{aligned}
& \text{Forward Equation:} \\
& \frac{\partial \phi}{\partial t} = -\frac{\partial}{\partial p}[M(p)\phi] + \frac{1}{2}\frac{\partial^2}{\partial p^2}[V(p)\phi] \\
& \text{Stationary Distribution:} \\
& \phi(p) = Ce^{4N_esp}\frac{1}{p(1-p)} \\
& \text{Mean First Passage:} \\
& \tau(p) = -2N_e[p\ln p + (1-p)\ln(1-p)]
\end{aligned}
```

### 2. Branching Processes

```math
\begin{aligned}
& \text{Generating Function:} \\
& f(s) = \sum_{k=0}^\infty p_ks^k \\
& \text{Extinction Probability:} \\
& q = \min\{x \geq 0: f(x) = x\} \\
& \text{Mean Offspring:} \\
& m = f'(1) = \sum_{k=0}^\infty kp_k
\end{aligned}
```

### 3. Ancestral Processes

```math
\begin{aligned}
& \text{Coalescent Rate:} \\
& \lambda_k = \binom{k}{2} \\
& \text{TMRCA Distribution:} \\
& f(t) = \sum_{k=2}^n a_k\lambda_ke^{-\lambda_kt} \\
& \text{Site Frequency Spectrum:} \\
& \xi_i = \frac{\theta}{i}, 1 \leq i < n
\end{aligned}
```

## Implementation Considerations

### 1. Numerical Methods
- Forward-time simulation
- Backward-time simulation
- Importance sampling

### 2. Statistical Analysis
- Maximum likelihood
- Bayesian inference
- ABC methods

### 3. Computational Efficiency
- Sparse matrix methods
- GPU acceleration
- Parallel simulation

## References
- [[kimura_1983]] - "The Neutral Theory of Molecular Evolution"
- [[ewens_2004]] - "Mathematical Population Genetics"
- [[wakeley_2008]] - "Coalescent Theory: An Introduction"
- [[nielsen_2005]] - "Statistical Methods in Molecular Evolution"

## See Also
- [[evolutionary_dynamics]]
- [[molecular_evolution]]
- [[quantitative_genetics]]
- [[statistical_genetics]]
- [[ecological_genetics]] 