---

title: Bethe Free Energy

type: mathematical_concept

status: stable

created: 2025-08-08

tags: [variational, graphical-models]

semantic_relations:

  - type: relates

    links: [message_passing, belief_propagation]

  - type: connects

    links: [variational_free_energy]

---

## Bethe Free Energy

### Definition (factor graphs)

Let \(b_i(x_i)\) and \(b_a(\mathbf{x}_a)\) be variable and factor beliefs. The Bethe functional:

```math

F_{\mathrm{Bethe}}(b) = \sum_a \sum_{\mathbf{x}_a} b_a(\mathbf{x}_a) \log \frac{b_a(\mathbf{x}_a)}{\psi_a(\mathbf{x}_a)} - \sum_i (d_i-1) \sum_{x_i} b_i(x_i) \log b_i(x_i)

```

where \(d_i\) is the degree of variable \(i\). Stationary points under consistency constraints correspond to (loopy) BP fixed points.

### Notes

- Exact on trees; approximate on loopy graphs

- Links variational stationary conditions to message passing updates

## Implementation

### Bethe Free Energy Computation

```python
def bethe_free_energy(factor_beliefs, variable_beliefs, factors, variables):
    """
    Compute Bethe free energy for factor graph.

    Args:
        factor_beliefs: Dictionary of factor beliefs b_a(x_a)
        variable_beliefs: Dictionary of variable beliefs b_i(x_i)
        factors: List of factor nodes with their connected variables
        variables: List of variable nodes with their degrees

    Returns:
        float: Bethe free energy value
    """

    # Energy term: sum over factors
    energy = 0.0
    for factor_id, belief in factor_beliefs.items():
        factor_potential = factors[factor_id]['potential']
        energy += np.sum(belief * np.log(belief / factor_potential))

    # Entropy term: sum over variables
    entropy = 0.0
    for var_id, belief in variable_beliefs.items():
        degree = variables[var_id]['degree']
        entropy -= (degree - 1) * np.sum(belief * np.log(belief))

    return energy - entropy
```

### Message Passing for Bethe Free Energy

```python
class BetheFreeEnergyOptimizer:
    """Optimize Bethe free energy using message passing."""

    def __init__(self, factor_graph):
        self.graph = factor_graph
        self.messages = self._initialize_messages()

    def _initialize_messages(self):
        """Initialize messages uniformly."""
        messages = {}
        for edge in self.graph.edges:
            messages[edge] = np.ones(self.graph.get_edge_domain(edge)) / self.graph.get_edge_domain_size(edge)
        return messages

    def update_messages(self, max_iterations=100, tolerance=1e-6):
        """Update messages using belief propagation."""
        for iteration in range(max_iterations):
            old_messages = self.messages.copy()

            # Update variable-to-factor messages
            for var_node in self.graph.variables:
                for factor_neighbor in self.graph.get_factor_neighbors(var_node):
                    self._update_variable_to_factor_message(var_node, factor_neighbor)

            # Update factor-to-variable messages
            for factor_node in self.graph.factors:
                for var_neighbor in self.graph.get_variable_neighbors(factor_node):
                    self._update_factor_to_variable_message(factor_node, var_neighbor)

            # Check convergence
            if self._check_convergence(old_messages, tolerance):
                break

    def compute_beliefs(self):
        """Compute variable and factor beliefs from messages."""
        variable_beliefs = {}
        factor_beliefs = {}

        # Compute variable beliefs
        for var_node in self.graph.variables:
            incoming_messages = [
                self.messages[(factor, var_node)]
                for factor in self.graph.get_factor_neighbors(var_node)
            ]
            variable_beliefs[var_node] = self._normalize(self._product(incoming_messages))

        # Compute factor beliefs
        for factor_node in self.graph.factors:
            factor_potential = self.graph.factors[factor_node]['potential']
            incoming_messages = [
                self.messages[(var, factor_node)]
                for var in self.graph.get_variable_neighbors(factor_node)
            ]
            factor_beliefs[factor_node] = self._normalize(
                factor_potential * self._product(incoming_messages)
            )

        return variable_beliefs, factor_beliefs

    def compute_free_energy(self):
        """Compute Bethe free energy."""
        variable_beliefs, factor_beliefs = self.compute_beliefs()
        return bethe_free_energy(
            factor_beliefs, variable_beliefs,
            self.graph.factors, self.graph.variables
        )
```

### Theoretical Properties

The Bethe free energy provides:

1. **Variational Bound**: Upper bound on the true log partition function
2. **Stationary Points**: Fixed points correspond to BP solutions
3. **Tree Exactness**: Exact on tree-structured graphs
4. **Loop Corrections**: Approximate corrections for loopy graphs

### Applications

- **Error-Correcting Codes**: LDPC and turbo codes decoding
- **Computer Vision**: Markov random field inference
- **Statistical Physics**: Approximate free energy computation
- **Bayesian Networks**: Approximate inference in large networks

### See also

- [[message_passing]] · [[belief_propagation]] · [[variational_free_energy]] · [[factor_graphs]]

