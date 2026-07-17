---

type: mathematical_concept

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

1. Optimization Target

   - VFE: Minimize perception error

   - EFE: Optimize action selection

1. Component Focus

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

See the canonical package documentation for a complete runnable example.


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

