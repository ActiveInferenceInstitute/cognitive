---
title: Graph Theory
type: concept
status: stable
created: 2024-03-15
complexity: intermediate
processing_priority: 1
tags:
  - mathematics
  - discrete_mathematics
  - networks
  - combinatorics
semantic_relations:
  - type: foundation_for
    links:
      - [[probabilistic_graphical_models]]
      - [[network_science]]
      - [[markov_random_fields]]
  - type: implements
    links:
      - [[discrete_mathematics]]
      - [[combinatorics]]
  - type: relates
    links:
      - [[optimization_theory]]
      - [[information_theory]]
      - [[complexity_theory]]

---

# Graph Theory

## Overview

Graph Theory is a branch of mathematics that studies the relationships between pairs of objects. It provides the mathematical foundation for analyzing networks, connections, and structural properties in various domains, from social networks to neural architectures.

## Mathematical Foundation

### Basic Definitions

#### Graph Structure
```math
G = (V,E)
```
where:
- $V$ is vertex set
- $E \subseteq V \times V$ is edge set

#### Graph Properties
```math
\begin{align*}
\text{degree}(v) &= |\{u \in V : (v,u) \in E\}| \\
\text{path}(u,v) &= (v_1,\ldots,v_k), v_1=u, v_k=v \\
\text{distance}(u,v) &= \min\{k : \text{path}(u,v) \text{ has length } k\}
\end{align*}
```

## Implementation

### Graph Data Structure

```python
class Graph:
    def __init__(self,
                 vertices: Set[Any],
                 edges: Set[Tuple[Any, Any]],
                 directed: bool = False,
                 weighted: bool = False):
        """Initialize graph.
        
        Args:
            vertices: Set of vertices
            edges: Set of edges
            directed: Whether graph is directed
            weighted: Whether graph is weighted
        """
        self.vertices = vertices
        self.edges = edges
        self.directed = directed
        self.weighted = weighted
        
        # Initialize adjacency structure
        self.adjacency = self._build_adjacency()
        
        if weighted:
            self.weights = {}
    
    def _build_adjacency(self) -> Dict[Any, Set[Any]]:
        """Build adjacency structure.
        
        Returns:
            adjacency: Adjacency dictionary
        """
        adj = {v: set() for v in self.vertices}
        
        for u, v in self.edges:
            adj[u].add(v)
            if not self.directed:
                adj[v].add(u)
        
        return adj
    
    def add_edge(self,
                u: Any,
                v: Any,
                weight: Optional[float] = None):
        """Add edge to graph.
        
        Args:
            u: First vertex
            v: Second vertex
            weight: Edge weight
        """
        self.edges.add((u, v))
        self.adjacency[u].add(v)
        
        if not self.directed:
            self.adjacency[v].add(u)
        
        if self.weighted and weight is not None:
            self.weights[(u, v)] = weight
            if not self.directed:
                self.weights[(v, u)] = weight
    
    def remove_edge(self,
                   u: Any,
                   v: Any):
        """Remove edge from graph.
        
        Args:
            u: First vertex
            v: Second vertex
        """
        self.edges.remove((u, v))
        self.adjacency[u].remove(v)
        
        if not self.directed:
            self.adjacency[v].remove(u)
        
        if self.weighted:
            del self.weights[(u, v)]
            if not self.directed:
                del self.weights[(v, u)]
```

### Graph Algorithms

```python
class GraphAlgorithms:
    def __init__(self,
                 graph: Graph):
        """Initialize graph algorithms.
        
        Args:
            graph: Input graph
        """
        self.graph = graph
    
    def shortest_path(self,
                     source: Any,
                     target: Any) -> Tuple[List[Any], float]:
        """Find shortest path using Dijkstra's algorithm.
        
        Args:
            source: Source vertex
            target: Target vertex
            
        Returns:
            path: Shortest path
            distance: Path length
        """
        # Initialize distances
        dist = {v: float('inf') for v in self.graph.vertices}
        dist[source] = 0
        
        # Initialize predecessors
        pred = {v: None for v in self.graph.vertices}
        
        # Priority queue
        pq = [(0, source)]
        visited = set()
        
        while pq:
            d, u = heapq.heappop(pq)
            
            if u == target:
                break
                
            if u in visited:
                continue
                
            visited.add(u)
            
            # Update neighbors
            for v in self.graph.adjacency[u]:
                weight = self.graph.weights.get((u, v), 1)
                alt = d + weight
                
                if alt < dist[v]:
                    dist[v] = alt
                    pred[v] = u
                    heapq.heappush(pq, (alt, v))
        
        # Reconstruct path
        path = []
        current = target
        
        while current is not None:
            path.append(current)
            current = pred[current]
        
        return path[::-1], dist[target]
    
    def minimum_spanning_tree(self) -> Set[Tuple[Any, Any]]:
        """Find minimum spanning tree using Kruskal's algorithm.
        
        Returns:
            mst: Minimum spanning tree edges
        """
        # Initialize disjoint sets
        parent = {v: v for v in self.graph.vertices}
        rank = {v: 0 for v in self.graph.vertices}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if rank[px] < rank[py]:
                parent[px] = py
            elif rank[px] > rank[py]:
                parent[py] = px
            else:
                parent[py] = px
                rank[px] += 1
        
        # Sort edges by weight
        edges = sorted(
            self.graph.edges,
            key=lambda e: self.graph.weights.get(e, 1)
        )
        
        # Build MST
        mst = set()
        for u, v in edges:
            if find(u) != find(v):
                union(u, v)
                mst.add((u, v))
        
        return mst
```

## Applications

### Network Analysis

#### Centrality Measures
- Degree centrality
- Betweenness centrality
- Eigenvector centrality
- PageRank

#### Community Detection
- Modularity optimization
- Spectral clustering
- Label propagation

### Path Finding

#### Shortest Paths
- Dijkstra's algorithm
- Bellman-Ford algorithm
- Floyd-Warshall algorithm

#### Network Flows
- Maximum flow
- Minimum cut
- Bipartite matching

## Best Practices

### Implementation
1. Choose appropriate representation
2. Optimize for operations
3. Handle special cases
4. Consider scalability

### Algorithm Design
1. Analyze complexity
2. Handle edge cases
3. Optimize memory
4. Consider parallelization

### Validation
1. Test connectivity
2. Verify properties
3. Check invariants
4. Benchmark performance

## Common Issues

### Technical Challenges
1. Large graph processing
2. Dynamic updates
3. Memory constraints
4. Algorithm complexity

### Solutions
1. Sparse representations
2. Incremental updates
3. Distributed processing
4. Approximation algorithms

### Spectral Graph Theory

#### Mathematical Foundations

**Definition** (Graph Laplacian): For a graph $G = (V,E)$ with adjacency matrix $A$, the graph Laplacian is defined as:
$$L = D - A$$
where $D$ is the degree matrix with $D_{ii} = \sum_j A_{ij}$.

**Definition** (Normalized Laplacian): The normalized Laplacian is:
$$\mathcal{L} = D^{-1/2}LD^{-1/2} = I - D^{-1/2}AD^{-1/2}$$

**Theorem** (Spectral Properties): The eigenvalues of the Laplacian satisfy:
1. $0 = \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n$
2. The multiplicity of eigenvalue 0 equals the number of connected components
3. The second smallest eigenvalue $\lambda_2$ (algebraic connectivity) characterizes connectivity

```python
class SpectralGraphAnalysis:
    """Advanced spectral analysis of graphs with rigorous mathematical foundation."""
    
    def __init__(self, graph: Graph):
        """Initialize spectral analyzer.
        
        Args:
            graph: Input graph structure
        """
        self.graph = graph
        self._adjacency_matrix = None
        self._laplacian_matrix = None
        self._spectrum_cache = {}
    
    def compute_graph_spectrum(self, 
                             normalized: bool = True,
                             return_eigenvectors: bool = True) -> Dict[str, np.ndarray]:
        """Compute eigenvalues and eigenvectors of graph Laplacian.
        
        Mathematical Background:
        The spectrum of the graph Laplacian encodes fundamental structural properties.
        For the normalized Laplacian L_norm = D^(-1/2) L D^(-1/2):
        - Eigenvalues lie in [0, 2]
        - 0 is always an eigenvalue with multiplicity = # connected components
        - λ₂ (algebraic connectivity) measures how well-connected the graph is
        
        Args:
            normalized: Use normalized Laplacian if True, combinatorial if False
            return_eigenvectors: Whether to compute and return eigenvectors
            
        Returns:
            spectrum: Dictionary containing eigenvalues, eigenvectors, and derived measures
        """
        cache_key = f"normalized_{normalized}_eigvecs_{return_eigenvectors}"
        if cache_key in self._spectrum_cache:
            return self._spectrum_cache[cache_key]
        
        # Construct adjacency matrix
        adjacency = self._get_adjacency_matrix()
        
        # Construct Laplacian
        degree = np.diag(np.sum(adjacency, axis=1))
        laplacian = degree - adjacency
        
        if normalized:
            # Normalized Laplacian: L_norm = D^(-1/2) L D^(-1/2)
            degree_inv_sqrt = np.diag(1.0 / (np.sqrt(np.diag(degree)) + 1e-12))
            laplacian = degree_inv_sqrt @ laplacian @ degree_inv_sqrt
        
        # Compute spectrum
        if return_eigenvectors:
            eigenvals, eigenvecs = np.linalg.eigh(laplacian)
        else:
            eigenvals = np.linalg.eigvals(laplacian)
            eigenvecs = None
        
        # Sort by eigenvalue magnitude
        sort_idx = np.argsort(eigenvals)
        eigenvals = eigenvals[sort_idx]
        if eigenvecs is not None:
            eigenvecs = eigenvecs[:, sort_idx]
        
        # Compute derived measures
        result = {
            'eigenvalues': eigenvals,
            'eigenvectors': eigenvecs,
            'algebraic_connectivity': eigenvals[1] if len(eigenvals) > 1 else 0.0,
            'spectral_gap': eigenvals[1] - eigenvals[0] if len(eigenvals) > 1 else 0.0,
            'spectral_radius': eigenvals[-1],
            'number_of_components': np.sum(eigenvals < 1e-10),
            'laplacian_matrix': laplacian
        }
        
        # Add spectral embedding coordinates
        if eigenvecs is not None and len(eigenvals) > 2:
            result['fiedler_vector'] = eigenvecs[:, 1]  # Second eigenvector
            result['spectral_embedding_2d'] = eigenvecs[:, 1:3]  # 2D embedding
        
        self._spectrum_cache[cache_key] = result
        return result
    
    def cheeger_bound(self) -> Dict[str, float]:
        """Compute Cheeger constant and Cheeger's inequality bounds.
        
        Cheeger's Inequality: For a graph with normalized Laplacian eigenvalues λ₁ ≤ λ₂ ≤ ...,
        the Cheeger constant h satisfies:
        λ₂/2 ≤ h ≤ √(2λ₂)
        
        Returns:
            bounds: Dictionary containing Cheeger bounds and related measures
        """
        spectrum = self.compute_graph_spectrum(normalized=True)
        lambda_2 = spectrum['algebraic_connectivity']
        
        # Theoretical bounds
        lower_bound = lambda_2 / 2
        upper_bound = np.sqrt(2 * lambda_2)
        
        # Empirical Cheeger constant (simplified approximation)
        empirical_cheeger = self._compute_empirical_cheeger()
        
        return {
            'algebraic_connectivity': lambda_2,
            'cheeger_lower_bound': lower_bound,
            'cheeger_upper_bound': upper_bound,
            'empirical_cheeger_constant': empirical_cheeger,
            'bound_tightness': (empirical_cheeger - lower_bound) / (upper_bound - lower_bound)
        }
    
    def spectral_clustering(self, 
                          k: int, 
                          method: str = 'normalized_cuts') -> Dict[str, np.ndarray]:
        """Perform spectral clustering using eigenvectors of the Laplacian.
        
        Args:
            k: Number of clusters
            method: Clustering method ('normalized_cuts', 'ratio_cuts')
            
        Returns:
            clustering: Dictionary containing cluster assignments and quality measures
        """
        from sklearn.cluster import KMeans
        
        spectrum = self.compute_graph_spectrum(normalized=(method == 'normalized_cuts'))
        
        # Use first k eigenvectors for embedding
        embedding = spectrum['eigenvectors'][:, :k]
        
        # Normalize rows to unit length (for normalized cuts)
        if method == 'normalized_cuts':
            row_norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            embedding = embedding / (row_norms + 1e-12)
        
        # K-means clustering in spectral space
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embedding)
        
        # Compute clustering quality measures
        modularity = self._compute_modularity(cluster_labels)
        conductance = self._compute_average_conductance(cluster_labels)
        
        return {
            'cluster_labels': cluster_labels,
            'spectral_embedding': embedding,
            'cluster_centers': kmeans.cluster_centers_,
            'modularity': modularity,
            'average_conductance': conductance,
            'inertia': kmeans.inertia_
        }
    
    def random_walk_analysis(self) -> Dict[str, float]:
        """Analyze random walk properties using spectral methods.
        
        Returns:
            rw_properties: Dictionary of random walk characteristics
        """
        spectrum = self.compute_graph_spectrum(normalized=True)
        eigenvals = spectrum['eigenvalues']
        
        # Mixing time (approximate)
        lambda_2 = spectrum['algebraic_connectivity']
        mixing_time = np.log(self.graph.vertices.__len__()) / lambda_2 if lambda_2 > 0 else np.inf
        
        # Relaxation time
        relaxation_time = 1 / lambda_2 if lambda_2 > 0 else np.inf
        
        # Spectral gap
        spectral_gap = eigenvals[1] - eigenvals[0] if len(eigenvals) > 1 else 0
        
        return {
            'mixing_time_bound': mixing_time,
            'relaxation_time': relaxation_time,
            'spectral_gap': spectral_gap,
            'is_bipartite': abs(eigenvals[-1] - 2.0) < 1e-10,  # λₙ = 2 iff bipartite
            'chromatic_number_bound': len(eigenvals)  # χ(G) ≤ λₙ + 1
        }
    
    def graph_signal_processing(self, 
                              signal: np.ndarray,
                              filter_type: str = 'low_pass',
                              cutoff: float = 0.5) -> Dict[str, np.ndarray]:
        """Apply graph signal processing using spectral methods.
        
        Args:
            signal: Signal on graph vertices
            filter_type: Type of filter ('low_pass', 'high_pass', 'band_pass')
            cutoff: Filter cutoff parameter
            
        Returns:
            filtered_result: Dictionary containing filtered signal and analysis
        """
        spectrum = self.compute_graph_spectrum(normalized=True)
        eigenvals = spectrum['eigenvalues']
        eigenvecs = spectrum['eigenvectors']
        
        # Transform signal to spectral domain
        spectral_signal = eigenvecs.T @ signal
        
        # Apply frequency filter
        if filter_type == 'low_pass':
            filter_response = (eigenvals <= cutoff).astype(float)
        elif filter_type == 'high_pass':
            filter_response = (eigenvals >= cutoff).astype(float)
        elif filter_type == 'band_pass':
            filter_response = ((eigenvals >= cutoff[0]) & 
                             (eigenvals <= cutoff[1])).astype(float)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
        
        # Apply filter and transform back
        filtered_spectral = spectral_signal * filter_response
        filtered_signal = eigenvecs @ filtered_spectral
        
        return {
            'filtered_signal': filtered_signal,
            'original_signal': signal,
            'spectral_coefficients': spectral_signal,
            'filter_response': filter_response,
            'energy_preservation': np.linalg.norm(filtered_signal) / np.linalg.norm(signal)
        }
    
    def _get_adjacency_matrix(self) -> np.ndarray:
        """Get adjacency matrix representation of graph."""
        if self._adjacency_matrix is not None:
            return self._adjacency_matrix
        
        vertices = list(self.graph.vertices)
        n = len(vertices)
        vertex_to_idx = {v: i for i, v in enumerate(vertices)}
        
        adjacency = np.zeros((n, n))
        for u, v in self.graph.edges:
            i, j = vertex_to_idx[u], vertex_to_idx[v]
            adjacency[i, j] = 1
            if not self.graph.directed:
                adjacency[j, i] = 1
        
        self._adjacency_matrix = adjacency
        return adjacency
    
    def _compute_empirical_cheeger(self) -> float:
        """Compute empirical Cheeger constant (simplified)."""
        # This would require finding the actual minimum cut
        # Simplified approximation using Fiedler vector
        spectrum = self.compute_graph_spectrum(normalized=True)
        fiedler_vector = spectrum.get('fiedler_vector', np.ones(len(self.graph.vertices)))
        
        # Threshold Fiedler vector to get bipartition
        threshold = np.median(fiedler_vector)
        partition = fiedler_vector >= threshold
        
        # Compute cut size and conductance
        cut_size = 0
        vol_s = np.sum(partition)
        vol_complement = len(partition) - vol_s
        
        adjacency = self._get_adjacency_matrix()
        for i in range(len(partition)):
            for j in range(len(partition)):
                if partition[i] != partition[j] and adjacency[i, j] > 0:
                    cut_size += 1
        
        conductance = cut_size / min(vol_s, vol_complement) if min(vol_s, vol_complement) > 0 else 0
        return conductance
    
    def _compute_modularity(self, cluster_labels: np.ndarray) -> float:
        """Compute modularity of clustering."""
        adjacency = self._get_adjacency_matrix()
        m = np.sum(adjacency) / 2  # Number of edges
        degrees = np.sum(adjacency, axis=1)
        
        modularity = 0.0
        n_clusters = len(np.unique(cluster_labels))
        
        for cluster in range(n_clusters):
            cluster_nodes = np.where(cluster_labels == cluster)[0]
            
            # Internal edges
            internal_edges = 0
            cluster_degree_sum = 0
            
            for i in cluster_nodes:
                cluster_degree_sum += degrees[i]
                for j in cluster_nodes:
                    internal_edges += adjacency[i, j]
            
            internal_edges /= 2  # Each edge counted twice
            expected_internal = (cluster_degree_sum / (2 * m))**2
            
            modularity += (internal_edges / m) - expected_internal
        
        return modularity
    
    def _compute_average_conductance(self, cluster_labels: np.ndarray) -> float:
        """Compute average conductance of clusters."""
        adjacency = self._get_adjacency_matrix()
        n_clusters = len(np.unique(cluster_labels))
        
        total_conductance = 0.0
        for cluster in range(n_clusters):
            cluster_nodes = np.where(cluster_labels == cluster)[0]
            
            # Cut size
            cut_size = 0
            cluster_volume = 0
            
            for i in cluster_nodes:
                degree_i = np.sum(adjacency[i, :])
                cluster_volume += degree_i
                
                for j in range(len(cluster_labels)):
                    if cluster_labels[j] != cluster and adjacency[i, j] > 0:
                        cut_size += 1
            
            # Conductance
            complement_volume = np.sum(adjacency) - cluster_volume
            min_volume = min(cluster_volume, complement_volume)
            
            if min_volume > 0:
                conductance = cut_size / min_volume
                total_conductance += conductance
        
        return total_conductance / n_clusters

# Example: Enhanced validation with spectral properties
def validate_spectral_properties():
    """Validate spectral graph theory implementation."""
    
    # Create test graphs
    import networkx as nx
    
    # Complete graph K_5
    G_complete = nx.complete_graph(5)
    
    # Cycle graph C_8  
    G_cycle = nx.cycle_graph(8)
    
    # Path graph P_10
    G_path = nx.path_graph(10)
    
    test_graphs = [
        ("Complete K_5", G_complete),
        ("Cycle C_8", G_cycle), 
        ("Path P_10", G_path)
    ]
    
    for name, nx_graph in test_graphs:
        print(f"\n=== {name} ===")
        
        # Convert to our Graph format
        vertices = set(nx_graph.nodes())
        edges = set(nx_graph.edges())
        graph = Graph(vertices, edges, directed=False)
        
        # Spectral analysis
        analyzer = SpectralGraphAnalysis(graph)
        spectrum = analyzer.compute_graph_spectrum()
        cheeger = analyzer.cheeger_bound()
        rw_props = analyzer.random_walk_analysis()
        
        print(f"Algebraic connectivity: {spectrum['algebraic_connectivity']:.4f}")
        print(f"Spectral gap: {spectrum['spectral_gap']:.4f}")
        print(f"Number of components: {spectrum['number_of_components']}")
        print(f"Cheeger bounds: [{cheeger['cheeger_lower_bound']:.4f}, {cheeger['cheeger_upper_bound']:.4f}]")
        print(f"Mixing time bound: {rw_props['mixing_time_bound']:.4f}")
        print(f"Is bipartite: {rw_props['is_bipartite']}")

if __name__ == "__main__":
    validate_spectral_properties()
```

### Advanced Graph Algorithms

#### Spectral Partitioning

**Theorem** (Cheeger's Inequality): For any graph partition $(S, \bar{S})$ with conductance $\Phi(S)$, the algebraic connectivity $\lambda_2$ of the normalized Laplacian satisfies:
$$\frac{\lambda_2}{2} \leq \min_S \Phi(S) \leq \sqrt{2\lambda_2}$$

This provides a fundamental connection between spectral properties and graph partitioning quality.

## Related Documentation
- [[discrete_mathematics]]
- [[combinatorics]]
- [[probabilistic_graphical_models]]
- [[network_science]]
- [[complexity_theory]] 