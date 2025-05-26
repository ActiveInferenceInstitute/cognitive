---
type: concept
id: free_energy_relationship_001
created: 2024-02-05
modified: 2024-02-05
tags: [active-inference, free-energy, theory]
aliases: [vfe-efe-relationship, free-energy-duality]
---

# Relationship Between VFE and EFE

## Overview

The relationship between Variational Free Energy (VFE) and Expected Free Energy (EFE) is fundamental to understanding Active Inference. While VFE quantifies current model fit, EFE guides future actions through prediction.

## Mathematical Connection

### Present vs Future
- VFE: $F = \mathbb{E}_{Q(x)}[-\ln P(y|x)] + D_{KL}[Q(x)\|P(x)]$
- EFE: $G(\pi) = -\mathbb{E}_{Q(\tilde{x},\tilde{y}|\pi)}[D_{KL}[Q(\tilde{x}|\tilde{y},\pi)\|Q(\tilde{x}|\pi)]] - \mathbb{E}_{Q(\tilde{y}|\pi)}[\ln P(\tilde{y}|C)]$

### Key Differences
1. Temporal Scope
   - VFE: Current state estimation
   - EFE: Future state prediction
   
2. Optimization Target
   - VFE: Minimize perception error
   - EFE: Optimize action selection

3. Component Focus
   - VFE: Accuracy vs Complexity
   - EFE: Epistemic vs Pragmatic value

## Implementation Details

```python
def compute_free_energies(
    model,
    observation: np.ndarray,
    action: Optional[int] = None
) -> Tuple[float, float]:
    """Compute both VFE and EFE for comparison.
    
    Args:
        model: Active Inference model instance
        observation: Current observation
        action: Optional action for EFE computation
        
    Returns:
        Tuple of (VFE, EFE) values
    """
    # Compute VFE
    vfe = model.compute_vfe(
        observation=observation,
        return_components=False
    )
    
    # Compute EFE if action is provided
    efe = None
    if action is not None:
        efe = model.compute_expected_free_energy(
            action_idx=action,
            return_components=False
        )
    
    return vfe, efe

def analyze_free_energy_relationship(
    model,
    time_window: int = 20
) -> Dict[str, np.ndarray]:
    """Analyze relationship between VFE and EFE over time.
    
    Args:
        model: Active Inference model instance
        time_window: Number of time steps to analyze
        
    Returns:
        Dictionary containing analysis results
    """
    results = {
        'time_steps': np.arange(time_window),
        'vfe_values': np.zeros(time_window),
        'efe_values': np.zeros(time_window),
        'correlation': np.zeros(time_window-1),
        'prediction_error': np.zeros(time_window-1)
    }
    
    # Simulate and collect data
    for t in range(time_window):
        # Get current state
        observation = model.get_observation()
        action = model.select_action()
        
        # Compute free energies
        vfe, efe = compute_free_energies(
            model=model,
            observation=observation,
            action=action
        )
        
        # Store values
        results['vfe_values'][t] = vfe
        results['efe_values'][t] = efe
        
        # Update model
        model.step(action)
        
        # Compute relationships for t > 0
        if t > 0:
            # Correlation between VFE and EFE
            results['correlation'][t-1] = np.corrcoef(
                results['vfe_values'][:t],
                results['efe_values'][:t]
            )[0,1]
            
            # Prediction error (how well EFE predicted next VFE)
            results['prediction_error'][t-1] = np.abs(
                results['efe_values'][t-1] - results['vfe_values'][t]
            )
    
    return results
```

## Key Properties

### 1. Temporal Dependency
- VFE depends on current observations
- EFE depends on predicted future states
- Both contribute to belief updating

### 2. Information Flow
- VFE → Belief Update → Action Selection
- EFE → Policy Selection → Action Execution
- Circular causation through action-perception cycle

### 3. Optimization Characteristics
- VFE: Convex optimization
- EFE: Non-convex optimization
- Different convergence properties

## Practical Implications

### 1. Model Design
- Balance between components
- Proper scaling of terms
- Numerical stability

### 2. Algorithm Implementation
- Sequential computation
- Memory requirements
- Computational efficiency

### 3. Performance Analysis
- Convergence metrics
- Behavioral patterns
- Learning dynamics

## Related Concepts
- [[belief_updating]]
- [[policy_selection]]
- [[active_inference_cycle]]
- [[optimization_methods]]

## Common Challenges

### 1. Numerical Issues
- Scale differences
- Gradient computation
- Stability concerns

### 2. Implementation Complexity
- Component balance
- Parameter tuning
- Convergence monitoring

### 3. Analysis Difficulties
- Interpretation of values
- Component attribution
- Performance assessment

## Best Practices

### 1. Implementation
- Use stable numerical methods
- Monitor component ratios
- Implement sanity checks

### 2. Analysis
- Track both measures
- Compare trajectories
- Validate predictions

### 3. Optimization
- Balance update rates
- Monitor convergence
- Validate results

## Advanced Mathematical Analysis

### Rigorous Theoretical Framework

**Definition** (Temporal Free Energy Decomposition): For a dynamical system with state $x_t$ and observations $y_t$, the total free energy decomposes as:
$$\mathcal{F}_{\text{total}} = \underbrace{F_t[q(x_t)]}_{VFE} + \underbrace{\mathbb{E}_{\pi}\left[\sum_{\tau=t+1}^{T} G_\tau(\pi)\right]}_{EFE}$$

**Theorem** (Free Energy Consistency): Under optimal inference, the relationship between VFE and EFE satisfies:
$$\lim_{t \to \infty} \frac{1}{t} \sum_{\tau=1}^t F_\tau = \lim_{T \to \infty} \frac{1}{T} \sum_{\tau=1}^T G_\tau(\pi^*)$$

where $\pi^*$ is the optimal policy.

**Proof Sketch**: By the ergodic theorem and the optimality of the free energy principle, long-term averages of VFE and EFE converge under stationary conditions.

```python
class AdvancedFreeEnergyAnalysis:
    """Advanced analysis of VFE-EFE relationships with rigorous mathematical foundation."""
    
    def __init__(self,
                 model: Any,
                 analysis_horizon: int = 100,
                 statistical_confidence: float = 0.95):
        """Initialize advanced free energy analysis framework.
        
        Args:
            model: Active inference model
            analysis_horizon: Time horizon for analysis
            statistical_confidence: Confidence level for statistical tests
        """
        self.model = model
        self.horizon = analysis_horizon
        self.confidence = statistical_confidence
        
        # Initialize tracking structures
        self.vfe_history = []
        self.efe_history = []
        self.cross_correlations = []
        self.prediction_accuracies = []
        
    def comprehensive_relationship_analysis(self) -> Dict[str, Any]:
        """Perform comprehensive analysis of VFE-EFE relationship.
        
        Returns:
            Complete analysis including statistical tests and theoretical validation
        """
        # Collect temporal data
        temporal_data = self._collect_temporal_data()
        
        # Statistical relationship analysis
        statistical_analysis = self._perform_statistical_analysis(temporal_data)
        
        # Information-theoretic analysis
        information_analysis = self._perform_information_analysis(temporal_data)
        
        # Causal analysis
        causal_analysis = self._perform_causal_analysis(temporal_data)
        
        # Theoretical validation
        theoretical_validation = self._validate_theoretical_predictions(temporal_data)
        
        # Dynamical systems analysis
        dynamical_analysis = self._analyze_dynamical_properties(temporal_data)
        
        return {
            'temporal_data': temporal_data,
            'statistical_analysis': statistical_analysis,
            'information_analysis': information_analysis,
            'causal_analysis': causal_analysis,
            'theoretical_validation': theoretical_validation,
            'dynamical_analysis': dynamical_analysis,
            'summary_metrics': self._compute_summary_metrics(temporal_data)
        }
    
    def _collect_temporal_data(self) -> Dict[str, np.ndarray]:
        """Collect temporal data for VFE and EFE."""
        vfe_values = np.zeros(self.horizon)
        efe_values = np.zeros(self.horizon)
        actions = np.zeros(self.horizon, dtype=int)
        observations = []
        states = []
        
        for t in range(self.horizon):
            # Get current state and observation
            current_obs = self.model.get_observation()
            current_state = self.model.get_state()
            
            observations.append(current_obs)
            states.append(current_state)
            
            # Compute VFE
            vfe_values[t] = self.model.compute_vfe(observation=current_obs)
            
            # Select action and compute EFE
            selected_action = self.model.select_action()
            actions[t] = selected_action
            efe_values[t] = self.model.compute_expected_free_energy(
                action_idx=selected_action
            )
            
            # Update model
            self.model.step(selected_action)
        
        return {
            'vfe': vfe_values,
            'efe': efe_values,
            'actions': actions,
            'observations': np.array(observations),
            'states': np.array(states),
            'time_points': np.arange(self.horizon)
        }
    
    def _perform_statistical_analysis(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Perform statistical analysis of VFE-EFE relationship."""
        vfe, efe = data['vfe'], data['efe']
        
        # Basic statistics
        vfe_stats = {
            'mean': np.mean(vfe),
            'std': np.std(vfe),
            'skewness': scipy.stats.skew(vfe),
            'kurtosis': scipy.stats.kurtosis(vfe)
        }
        
        efe_stats = {
            'mean': np.mean(efe),
            'std': np.std(efe),
            'skewness': scipy.stats.skew(efe),
            'kurtosis': scipy.stats.kurtosis(efe)
        }
        
        # Correlation analysis
        pearson_corr, pearson_p = scipy.stats.pearsonr(vfe, efe)
        spearman_corr, spearman_p = scipy.stats.spearmanr(vfe, efe)
        
        # Cross-correlation analysis
        cross_corr = self._compute_cross_correlation(vfe, efe)
        
        # Stationarity tests
        vfe_stationarity = self._test_stationarity(vfe)
        efe_stationarity = self._test_stationarity(efe)
        
        # Granger causality test
        granger_test = self._granger_causality_test(vfe, efe)
        
        return {
            'vfe_statistics': vfe_stats,
            'efe_statistics': efe_stats,
            'pearson_correlation': {'r': pearson_corr, 'p_value': pearson_p},
            'spearman_correlation': {'r': spearman_corr, 'p_value': spearman_p},
            'cross_correlation': cross_corr,
            'stationarity': {
                'vfe': vfe_stationarity,
                'efe': efe_stationarity
            },
            'granger_causality': granger_test
        }
    
    def _perform_information_analysis(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Perform information-theoretic analysis."""
        vfe, efe = data['vfe'], data['efe']
        
        # Discretize for mutual information calculation
        vfe_discrete = self._discretize_signal(vfe)
        efe_discrete = self._discretize_signal(efe)
        
        # Mutual information
        mutual_info = self._compute_mutual_information(vfe_discrete, efe_discrete)
        
        # Transfer entropy (VFE → EFE and EFE → VFE)
        te_vfe_to_efe = self._compute_transfer_entropy(vfe_discrete, efe_discrete)
        te_efe_to_vfe = self._compute_transfer_entropy(efe_discrete, vfe_discrete)
        
        # Information storage
        info_storage_vfe = self._compute_information_storage(vfe_discrete)
        info_storage_efe = self._compute_information_storage(efe_discrete)
        
        # Complexity measures
        lempel_ziv_vfe = self._compute_lempel_ziv_complexity(vfe_discrete)
        lempel_ziv_efe = self._compute_lempel_ziv_complexity(efe_discrete)
        
        return {
            'mutual_information': mutual_info,
            'transfer_entropy': {
                'vfe_to_efe': te_vfe_to_efe,
                'efe_to_vfe': te_efe_to_vfe
            },
            'information_storage': {
                'vfe': info_storage_vfe,
                'efe': info_storage_efe
            },
            'complexity': {
                'lempel_ziv_vfe': lempel_ziv_vfe,
                'lempel_ziv_efe': lempel_ziv_efe
            }
        }
    
    def _perform_causal_analysis(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Perform causal analysis using various methods."""
        vfe, efe = data['vfe'], data['efe']
        
        # Cross-correlation for lag analysis
        cross_corr_lags = np.correlate(vfe - np.mean(vfe), 
                                     efe - np.mean(efe), mode='full')
        lags = np.arange(-len(efe) + 1, len(vfe))
        
        # Find optimal lag
        optimal_lag_idx = np.argmax(np.abs(cross_corr_lags))
        optimal_lag = lags[optimal_lag_idx]
        
        # Convergent cross mapping (CCM)
        ccm_vfe_to_efe = self._convergent_cross_mapping(vfe, efe)
        ccm_efe_to_vfe = self._convergent_cross_mapping(efe, vfe)
        
        # Phase space reconstruction
        phase_space_analysis = self._phase_space_reconstruction(vfe, efe)
        
        return {
            'optimal_lag': optimal_lag,
            'lag_correlation': cross_corr_lags[optimal_lag_idx],
            'convergent_cross_mapping': {
                'vfe_to_efe': ccm_vfe_to_efe,
                'efe_to_vfe': ccm_efe_to_vfe
            },
            'phase_space_analysis': phase_space_analysis
        }
    
    def _validate_theoretical_predictions(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Validate theoretical predictions about VFE-EFE relationship."""
        vfe, efe = data['vfe'], data['efe']
        
        # Test free energy minimization
        vfe_trend = self._compute_trend(vfe)
        efe_optimization = self._assess_efe_optimization(data['actions'], efe)
        
        # Test consistency theorem
        long_term_vfe_avg = np.mean(vfe)
        long_term_efe_avg = np.mean(efe)
        consistency_error = abs(long_term_vfe_avg - long_term_efe_avg)
        
        # Test prediction accuracy
        prediction_accuracy = self._test_prediction_accuracy(vfe, efe)
        
        # Test information bounds
        information_bounds = self._test_information_bounds(data)
        
        return {
            'vfe_minimization_trend': vfe_trend,
            'efe_optimization_quality': efe_optimization,
            'consistency_theorem_error': consistency_error,
            'prediction_accuracy': prediction_accuracy,
            'information_bounds_satisfied': information_bounds
        }
    
    def _analyze_dynamical_properties(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze dynamical systems properties."""
        vfe, efe = data['vfe'], data['efe']
        
        # Lyapunov exponent estimation
        lyapunov_vfe = self._estimate_lyapunov_exponent(vfe)
        lyapunov_efe = self._estimate_lyapunov_exponent(efe)
        
        # Attractor reconstruction
        attractor_vfe = self._reconstruct_attractor(vfe)
        attractor_efe = self._reconstruct_attractor(efe)
        
        # Fractal dimension
        fractal_dim_vfe = self._compute_fractal_dimension(vfe)
        fractal_dim_efe = self._compute_fractal_dimension(efe)
        
        # Recurrence analysis
        recurrence_vfe = self._recurrence_analysis(vfe)
        recurrence_efe = self._recurrence_analysis(efe)
        
        return {
            'lyapunov_exponents': {
                'vfe': lyapunov_vfe,
                'efe': lyapunov_efe
            },
            'attractors': {
                'vfe': attractor_vfe,
                'efe': attractor_efe
            },
            'fractal_dimensions': {
                'vfe': fractal_dim_vfe,
                'efe': fractal_dim_efe
            },
            'recurrence_analysis': {
                'vfe': recurrence_vfe,
                'efe': recurrence_efe
            }
        }
    
    def _compute_cross_correlation(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Compute cross-correlation between two signals."""
        # Normalize signals
        x_norm = (x - np.mean(x)) / np.std(x)
        y_norm = (y - np.mean(y)) / np.std(y)
        
        # Compute cross-correlation
        cross_corr = np.correlate(x_norm, y_norm, mode='full')
        lags = np.arange(-len(y) + 1, len(x))
        
        return {
            'cross_correlation': cross_corr,
            'lags': lags,
            'max_correlation': np.max(np.abs(cross_corr)),
            'lag_at_max': lags[np.argmax(np.abs(cross_corr))]
        }
    
    def _test_stationarity(self, signal: np.ndarray) -> Dict[str, Any]:
        """Test stationarity using Augmented Dickey-Fuller test."""
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(signal)
            
            return {
                'adf_statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < 0.05
            }
        except ImportError:
            # Fallback: simple variance test
            n_segments = 5
            segment_length = len(signal) // n_segments
            segment_vars = [
                np.var(signal[i*segment_length:(i+1)*segment_length])
                for i in range(n_segments)
            ]
            var_stability = np.std(segment_vars) / np.mean(segment_vars)
            
            return {
                'variance_stability': var_stability,
                'is_stationary': var_stability < 0.2  # Heuristic threshold
            }
    
    def _granger_causality_test(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Simplified Granger causality test."""
        # Use simple autoregressive model for causality testing
        lag_order = min(5, len(x) // 10)
        
        # Test if x Granger-causes y
        x_to_y_causality = self._test_granger_direction(x, y, lag_order)
        y_to_x_causality = self._test_granger_direction(y, x, lag_order)
        
        return {
            'x_to_y': x_to_y_causality,
            'y_to_x': y_to_x_causality,
            'bidirectional': x_to_y_causality and y_to_x_causality
        }
    
    def _test_granger_direction(self, cause: np.ndarray, effect: np.ndarray, lag: int) -> bool:
        """Test Granger causality in one direction."""
        n = min(len(cause), len(effect)) - lag
        
        # Construct lagged matrices
        X_lagged = np.column_stack([
            effect[lag-i:-i] if i > 0 else effect[lag:]
            for i in range(1, lag + 1)
        ])
        X_with_cause = np.column_stack([
            X_lagged,
            np.column_stack([
                cause[lag-i:-i] if i > 0 else cause[lag:]
                for i in range(1, lag + 1)
            ])
        ])
        y = effect[lag:]
        
        # Fit models
        try:
            from sklearn.linear_model import LinearRegression
            
            model_without_cause = LinearRegression().fit(X_lagged, y)
            model_with_cause = LinearRegression().fit(X_with_cause, y)
            
            # Compare R-squared
            r2_without = model_without_cause.score(X_lagged, y)
            r2_with = model_with_cause.score(X_with_cause, y)
            
            # F-test approximation
            improvement = r2_with - r2_without
            return improvement > 0.01  # Threshold for significance
            
        except ImportError:
            # Fallback: correlation-based test
            return np.abs(np.corrcoef(cause[:-1], effect[1:])[0, 1]) > 0.1
    
    def _discretize_signal(self, signal: np.ndarray, n_bins: int = 10) -> np.ndarray:
        """Discretize continuous signal for information-theoretic measures."""
        return np.digitize(signal, bins=np.linspace(signal.min(), signal.max(), n_bins))
    
    def _compute_mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute mutual information between discretized signals."""
        # Joint histogram
        joint_hist, _, _ = np.histogram2d(x, y, bins=max(len(np.unique(x)), len(np.unique(y))))
        joint_prob = joint_hist / np.sum(joint_hist)
        
        # Marginal probabilities
        p_x = np.sum(joint_prob, axis=1)
        p_y = np.sum(joint_prob, axis=0)
        
        # Mutual information
        mi = 0.0
        for i in range(len(p_x)):
            for j in range(len(p_y)):
                if joint_prob[i, j] > 0:
                    mi += joint_prob[i, j] * np.log(
                        joint_prob[i, j] / (p_x[i] * p_y[j])
                    )
        
        return mi
    
    def _compute_transfer_entropy(self, source: np.ndarray, target: np.ndarray, lag: int = 1) -> float:
        """Compute transfer entropy from source to target."""
        # Simplified transfer entropy computation
        # Full implementation would require proper conditioning
        
        n = len(target) - lag
        source_lagged = source[:n]
        target_current = target[lag:]
        target_past = target[:n]
        
        # Approximate using conditional mutual information
        # This is a simplified version
        return self._compute_mutual_information(source_lagged, target_current) * 0.5
    
    def _compute_information_storage(self, signal: np.ndarray, lag: int = 1) -> float:
        """Compute information storage (active information storage)."""
        if len(signal) <= lag:
            return 0.0
        
        past = signal[:-lag]
        present = signal[lag:]
        
        return self._compute_mutual_information(past, present)
    
    def _compute_lempel_ziv_complexity(self, signal: np.ndarray) -> float:
        """Compute Lempel-Ziv complexity."""
        # Convert to string for pattern analysis
        signal_str = ''.join(map(str, signal))
        
        # Simple LZ complexity approximation
        patterns = set()
        i = 0
        while i < len(signal_str):
            j = i + 1
            while j <= len(signal_str):
                pattern = signal_str[i:j]
                if pattern not in patterns:
                    patterns.add(pattern)
                    i = j
                    break
                j += 1
            else:
                break
        
        return len(patterns) / len(signal_str)
    
    def _compute_summary_metrics(self, data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Compute summary metrics for the analysis."""
        vfe, efe = data['vfe'], data['efe']
        
        return {
            'vfe_efe_correlation': np.corrcoef(vfe, efe)[0, 1],
            'vfe_stability': 1.0 / (1.0 + np.std(np.diff(vfe))),
            'efe_stability': 1.0 / (1.0 + np.std(np.diff(efe))),
            'joint_entropy': self._compute_joint_entropy(vfe, efe),
            'information_integration': self._compute_information_integration(vfe, efe)
        }
    
    def _compute_joint_entropy(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute joint entropy of two signals."""
        x_discrete = self._discretize_signal(x)
        y_discrete = self._discretize_signal(y)
        
        joint_hist, _, _ = np.histogram2d(x_discrete, y_discrete, 
                                        bins=[len(np.unique(x_discrete)), 
                                             len(np.unique(y_discrete))])
        joint_prob = joint_hist / np.sum(joint_hist)
        
        # Joint entropy
        joint_entropy = -np.sum(joint_prob[joint_prob > 0] * 
                               np.log(joint_prob[joint_prob > 0]))
        
        return joint_entropy
    
    def _compute_information_integration(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute information integration measure."""
        x_discrete = self._discretize_signal(x)
        y_discrete = self._discretize_signal(y)
        
        # Individual entropies
        h_x = self._compute_entropy(x_discrete)
        h_y = self._compute_entropy(y_discrete)
        
        # Joint entropy
        h_xy = self._compute_joint_entropy(x, y)
        
        # Information integration (reduction in entropy due to interaction)
        return h_x + h_y - h_xy
    
    def _compute_entropy(self, signal: np.ndarray) -> float:
        """Compute entropy of discretized signal."""
        _, counts = np.unique(signal, return_counts=True)
        probabilities = counts / len(signal)
        return -np.sum(probabilities * np.log(probabilities))

# Example usage and validation
def validate_advanced_relationship_analysis():
    """Validate advanced VFE-EFE relationship analysis."""
    
    # Mock model for testing
    class MockModel:
        def __init__(self):
            self.t = 0
            self.state = np.random.randn(5)
        
        def get_observation(self):
            return self.state + 0.1 * np.random.randn(5)
        
        def get_state(self):
            return self.state.copy()
        
        def compute_vfe(self, observation):
            return np.linalg.norm(observation - self.state)**2
        
        def compute_expected_free_energy(self, action_idx):
            return np.random.exponential(1.0) + 0.1 * action_idx
        
        def select_action(self):
            return np.random.randint(0, 3)
        
        def step(self, action):
            self.state += 0.1 * (action - 1) * np.random.randn(5)
            self.t += 1
    
    # Initialize analysis
    model = MockModel()
    analyzer = AdvancedFreeEnergyAnalysis(model, analysis_horizon=50)
    
    # Perform comprehensive analysis
    results = analyzer.comprehensive_relationship_analysis()
    
    print("Analysis completed successfully!")
    print(f"VFE-EFE correlation: {results['summary_metrics']['vfe_efe_correlation']:.3f}")
    print(f"Information integration: {results['summary_metrics']['information_integration']:.3f}")
    print(f"VFE stationarity: {results['statistical_analysis']['stationarity']['vfe']['is_stationary']}")

if __name__ == "__main__":
    validate_advanced_relationship_analysis()
```

### Spectral Analysis of Free Energy Dynamics

**Definition** (Free Energy Spectral Density): The power spectral density of the free energy time series:
$$S_{FE}(\omega) = \lim_{T \to \infty} \frac{1}{T} \left|\int_0^T F(t) e^{-i\omega t} dt\right|^2$$

provides insight into the temporal dynamics and characteristic frequencies of free energy minimization.

```python
class SpectralFreeEnergyAnalysis:
    """Spectral analysis of free energy dynamics."""
    
    def __init__(self, sampling_rate: float = 1.0):
        """Initialize spectral analyzer.
        
        Args:
            sampling_rate: Sampling rate of the time series
        """
        self.fs = sampling_rate
    
    def power_spectral_analysis(self,
                              vfe_signal: np.ndarray,
                              efe_signal: np.ndarray) -> Dict[str, Any]:
        """Analyze power spectra of VFE and EFE signals.
        
        Args:
            vfe_signal: VFE time series
            efe_signal: EFE time series
            
        Returns:
            Spectral analysis results
        """
        from scipy import signal
        
        # Compute power spectral densities
        freqs_vfe, psd_vfe = signal.welch(vfe_signal, fs=self.fs, nperseg=len(vfe_signal)//4)
        freqs_efe, psd_efe = signal.welch(efe_signal, fs=self.fs, nperseg=len(efe_signal)//4)
        
        # Cross-spectral density
        freqs_cross, psd_cross = signal.csd(vfe_signal, efe_signal, fs=self.fs)
        
        # Coherence
        freqs_coh, coherence = signal.coherence(vfe_signal, efe_signal, fs=self.fs)
        
        # Dominant frequencies
        dominant_freq_vfe = freqs_vfe[np.argmax(psd_vfe)]
        dominant_freq_efe = freqs_efe[np.argmax(psd_efe)]
        
        return {
            'vfe_spectrum': {'frequencies': freqs_vfe, 'psd': psd_vfe},
            'efe_spectrum': {'frequencies': freqs_efe, 'psd': psd_efe},
            'cross_spectrum': {'frequencies': freqs_cross, 'psd': psd_cross},
            'coherence': {'frequencies': freqs_coh, 'coherence': coherence},
            'dominant_frequencies': {
                'vfe': dominant_freq_vfe,
                'efe': dominant_freq_efe
            },
            'spectral_similarity': self._compute_spectral_similarity(psd_vfe, psd_efe)
        }
    
    def _compute_spectral_similarity(self, psd1: np.ndarray, psd2: np.ndarray) -> float:
        """Compute spectral similarity between two PSDs."""
        # Normalize PSDs
        psd1_norm = psd1 / np.sum(psd1)
        psd2_norm = psd2 / np.sum(psd2)
        
        # Compute KL divergence as similarity measure
        kl_div = np.sum(psd1_norm * np.log((psd1_norm + 1e-10) / (psd2_norm + 1e-10)))
        
        return np.exp(-kl_div)  # Convert to similarity measure
```

## References
- [[friston_2015]] - Active Inference Theory
- [[parr_2019]] - Relationship Analysis
- [[da_costa_2020]] - Computational Implementation 