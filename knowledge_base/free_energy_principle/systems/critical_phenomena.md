---
title: "Critical Phenomena and the Free Energy Principle"
type: concept
status: stable
created: 2025-02-06
updated: 2025-02-06
tags:
  - free_energy_principle
  - systems
  - critical_phenomena
  - phase_transitions
  - self_organized_criticality
  - power_laws
  - edge_of_chaos
  - ising_model
semantic_relations:
  - type: foundation
    links:
      - [[knowledge_base/free_energy_principle/mathematics/core_principle|Core Principle]]
      - [[knowledge_base/free_energy_principle/mathematics/variational_free_energy|Variational Free Energy]]
  - type: relates
    links:
      - [[self_organization|Self-Organization]]
      - [[emergence|Emergence]]
      - [[complex_adaptation|Complex Adaptation]]
      - [[network_dynamics|Network Dynamics]]
      - [[resilience|Resilience]]
      - [[knowledge_base/free_energy_principle/biology/homeostasis|Homeostasis]]
---

# Critical Phenomena and the Free Energy Principle

## Overview

**Critical phenomena** occur at the boundary between distinct phases of matter or dynamical regimes, where systems exhibit long-range correlations, power-law distributions, and maximal susceptibility to perturbation. The **critical brain hypothesis** proposes that biological neural systems self-tune to operate near a critical point, gaining computational advantages such as maximal dynamic range, optimal information transmission, and sensitivity to inputs.

The Free Energy Principle provides a principled account of why living systems should be found near criticality. A system minimizing variational free energy must balance two competing demands:

1. **Accuracy**: Faithfully encoding sensory signals requires high sensitivity -- the hallmark of critical dynamics.
2. **Complexity**: Maintaining a tractable generative model requires regularization, which pulls the system toward ordered (subcritical) regimes.

The optimal tradeoff between accuracy and complexity places the system near -- but not exactly at -- the critical point. This **slightly subcritical** regime maximizes the mutual information between internal states and environmental causes while maintaining stability.

Critical phenomena under the FEP connect to fundamental questions about the nature of biological computation, the origin of neural variability, and the universality of scaling laws across biological systems.

## Mathematical Framework

### Free Energy Near a Phase Transition

Consider a system with an order parameter $m$ (e.g., mean neural activity, magnetization) and a control parameter $\beta$ (e.g., inverse temperature, coupling strength). The Landau free energy expansion near a continuous (second-order) phase transition takes the form:

$$
F(m; \beta) = a_0 + a_2(\beta) \, m^2 + a_4 \, m^4 + \cdots
$$

where $a_2(\beta) = a_2'(\beta - \beta_c)$ changes sign at the critical point $\beta_c$. For $\beta < \beta_c$ (disordered phase), the minimum is at $m = 0$; for $\beta > \beta_c$ (ordered phase), symmetry breaking yields $m \neq 0$.

### Variational Free Energy at Criticality

For an FEP agent whose generative model includes a phase-transition-capable system, the variational free energy decomposes as:

$$
F[q] = \underbrace{D_{\mathrm{KL}}[q(m) \,\|\, p(m)]}_{\text{complexity}} - \underbrace{\mathbb{E}_{q}[\ln p(s \mid m)]}_{\text{accuracy}}
$$

At criticality, the prior $p(m)$ is broad (flat free energy landscape), so complexity cost is low. Meanwhile, the divergent susceptibility $\chi = \partial m / \partial h \to \infty$ ensures maximal accuracy (sensitivity to external fields $h$). The variational free energy is thus minimized near the critical point.

### Scaling Laws

At a continuous phase transition, physical quantities obey power-law scaling characterized by **critical exponents**:

| Quantity | Symbol | Scaling Law | Exponent |
|----------|--------|-------------|----------|
| Order parameter | $m$ | $m \sim (\beta_c - \beta)^{\beta_{\text{exp}}}$ | $\beta_{\text{exp}}$ |
| Susceptibility | $\chi$ | $\chi \sim \|\beta - \beta_c\|^{-\gamma}$ | $\gamma$ |
| Correlation length | $\xi$ | $\xi \sim \|\beta - \beta_c\|^{-\nu}$ | $\nu$ |
| Specific heat | $C$ | $C \sim \|\beta - \beta_c\|^{-\alpha}$ | $\alpha$ |
| Correlation function | $G(r)$ | $G(r) \sim r^{-(d-2+\eta)}$ | $\eta$ |

These exponents satisfy **scaling relations** (e.g., the Rushbrooke identity $\alpha + 2\beta_{\text{exp}} + \gamma = 2$) that reflect the underlying **universality class** -- the exponents depend only on dimensionality $d$ and symmetry of the order parameter, not on microscopic details.

### Renormalization Group Near the Critical Point

The renormalization group (RG) provides the natural framework for understanding critical phenomena. Under a coarse-graining transformation $\mathcal{R}$, the effective free energy at scale $\ell$ transforms as:

$$
F_{\ell'} = \mathcal{R}[F_\ell]
$$

At the critical point, $F$ is a **fixed point** of the RG transformation: $F^* = \mathcal{R}[F^*]$. The linearized RG around the fixed point determines the critical exponents:

$$
\mathcal{R}[F^* + \delta F] \approx F^* + \Lambda \, \delta F
$$

where the eigenvalues of $\Lambda$ classify perturbations as **relevant** (growing under coarse-graining, $|\lambda| > 1$), **marginal** ($|\lambda| = 1$), or **irrelevant** (shrinking, $|\lambda| < 1$). The FEP connection: a system that minimizes free energy across scales naturally flows toward RG fixed points -- criticality is an attractor of multi-scale free energy minimization.

### Susceptibility and Fisher Information

The susceptibility at the critical point has a deep connection to Fisher information, which plays a central role in the FEP:

$$
\chi = \frac{\partial^2 F}{\partial h^2}\bigg|_{h=0} \propto I_F(\beta)
$$

where $I_F(\beta)$ is the Fisher information of the control parameter. At criticality, Fisher information diverges, meaning the system is maximally informative about its control parameter. Under the FEP, agents that self-tune to criticality thereby maximize the Fisher information of their generative models -- achieving optimal inference.

## Key Concepts

### The Critical Brain Hypothesis

The critical brain hypothesis (Beggs & Plenz, 2003; Shew & Plenz, 2013) proposes that cortical networks operate near a critical point between:

- **Subcritical (ordered) regime**: Activity dies out quickly. Low sensitivity, limited computational capacity, but high stability.
- **Supercritical (disordered) regime**: Activity propagates uncontrollably (seizure-like). High sensitivity but no stable representations.
- **Critical point**: Branching ratio $\sigma = 1$, where each active neuron activates on average one other neuron.

Evidence for near-critical brain dynamics includes:

1. **Neuronal avalanches** following power-law size distributions: $P(s) \sim s^{-3/2}$
2. **Long-range temporal correlations** with $1/f$-like power spectra
3. **Maximal dynamic range** -- the range of stimulus intensities that can be discriminated
4. **Maximal mutual information** between stimulus and neural response

### Self-Organized Criticality (SOC)

Self-organized criticality (Bak, Tang, & Wiesenfeld, 1987) is the property whereby a system naturally evolves toward the critical point without external tuning. Under the FEP, SOC emerges because:

1. Free energy minimization drives the system toward the regime of maximal model evidence.
2. Model evidence is maximized when the generative model has maximal Fisher information.
3. Fisher information diverges at the critical point.
4. Therefore, free energy minimization acts as the self-tuning mechanism that drives the system toward criticality.

The sandpile model is the canonical example: grains of sand added one at a time to a pile naturally produce a critical slope where avalanches of all sizes occur.

### Universality Classes and the FEP

Universality is the remarkable observation that systems with very different microscopic details can exhibit identical critical behavior. Under the FEP, this has a natural explanation: **universality classes correspond to equivalence classes of generative models**. Systems in the same universality class are those whose coarse-grained generative models (at the RG fixed point) are identical, regardless of fine-grained differences.

| Universality Class | $d$ | Symmetry | $\beta_{\text{exp}}$ | $\gamma$ | $\nu$ | Biological Example |
|--------------------|-----|----------|----------|----------|--------|-------------------|
| 2D Ising | 2 | $\mathbb{Z}_2$ | 1/8 | 7/4 | 1 | Cortical surface dynamics |
| 3D Ising | 3 | $\mathbb{Z}_2$ | 0.326 | 1.237 | 0.630 | Volume neural activity |
| Mean-field | $\ge 4$ | $\mathbb{Z}_2$ | 1/2 | 1 | 1/2 | Fully connected networks |
| Directed percolation | any | None (absorbing) | varies | varies | varies | Neural avalanches |
| Branching process | -- | -- | 3/2 (size) | -- | -- | Neuronal cascades |

### Edge-of-Chaos Dynamics

The **edge of chaos** is a closely related concept from the theory of computation in dynamical systems. A cellular automaton or recurrent network at the edge of chaos exhibits:

- **Class IV behavior** (Wolfram): Complex, long-lived transients capable of universal computation.
- **Maximum Lyapunov exponent near zero**: $\lambda_{\max} \approx 0$, poised between exponential divergence (chaos, $\lambda > 0$) and exponential convergence (order, $\lambda < 0$).
- **Maximal computational capacity**: The system can store, transmit, and modify information simultaneously.

Under the FEP, the edge of chaos is the dynamical regime where the generative model achieves maximal expressiveness (complex enough to capture environmental structure) while remaining tractable (stable enough to perform inference).

### Criticality and Optimal Inference

A formal connection between criticality and optimal inference can be established through the **evidence lower bound** (ELBO):

$$
\ln p(s) \ge -F[q] = \mathbb{E}_q[\ln p(s, m)] - \mathbb{E}_q[\ln q(m)]
$$

At the critical point, the entropy term $-\mathbb{E}_q[\ln q(m)]$ is maximized (the posterior is maximally broad), while the energy term $\mathbb{E}_q[\ln p(s, m)]$ retains structure. This yields the tightest possible ELBO -- the best variational approximation to the true posterior -- confirming that criticality is the optimal operating point for approximate Bayesian inference.

## Python Code Example

### 2D Ising Model Simulation Showing Critical Behavior

```python
"""
2D Ising model simulation demonstrating critical phenomena.
Shows the phase transition and relates it to free energy minimization.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


# ── Ising Model Core ───────────────────────────────────────────────
class IsingModel:
    """2D Ising model with Metropolis-Hastings dynamics."""

    def __init__(self, L: int = 64, beta: float = 0.44):
        self.L = L
        self.beta = beta  # inverse temperature
        self.spins = np.random.choice([-1, 1], size=(L, L))

    def energy(self) -> float:
        """Total energy: E = -J * sum_{<i,j>} s_i * s_j."""
        s = self.spins
        return -float(
            np.sum(s * np.roll(s, 1, axis=0)) +
            np.sum(s * np.roll(s, 1, axis=1))
        )

    def magnetization(self) -> float:
        """Order parameter: m = (1/N) * sum_i s_i."""
        return np.abs(np.mean(self.spins))

    def susceptibility_from_fluctuations(self, mag_samples: np.ndarray) -> float:
        """Susceptibility from fluctuation-dissipation: chi = beta * N * Var(m)."""
        N = self.L ** 2
        return self.beta * N * np.var(mag_samples)

    def metropolis_step(self):
        """One full sweep of Metropolis-Hastings updates."""
        L = self.L
        for _ in range(L * L):
            i, j = np.random.randint(0, L, size=2)
            s = self.spins
            # Sum of nearest neighbors
            nn_sum = (
                s[(i + 1) % L, j] + s[(i - 1) % L, j] +
                s[i, (j + 1) % L] + s[i, (j - 1) % L]
            )
            dE = 2.0 * s[i, j] * nn_sum
            if dE <= 0 or np.random.rand() < np.exp(-self.beta * dE):
                self.spins[i, j] *= -1

    def simulate(self, n_equil: int = 200, n_sample: int = 300
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """Run simulation: equilibrate then sample."""
        for _ in range(n_equil):
            self.metropolis_step()
        mags = np.zeros(n_sample)
        energies = np.zeros(n_sample)
        for t in range(n_sample):
            self.metropolis_step()
            mags[t] = self.magnetization()
            energies[t] = self.energy() / (self.L ** 2)
        return mags, energies


# ── Sweep across temperatures ─────────────────────────────────────
L = 32  # Lattice size (use 64+ for publication quality)
beta_c_exact = np.log(1 + np.sqrt(2)) / 2  # ~ 0.4407 (Onsager)
betas = np.linspace(0.20, 0.70, 25)

results = {
    "beta": betas,
    "magnetization": np.zeros(len(betas)),
    "susceptibility": np.zeros(len(betas)),
    "energy": np.zeros(len(betas)),
    "specific_heat": np.zeros(len(betas)),
    "free_energy_proxy": np.zeros(len(betas)),
}

print(f"Ising model (L={L}), exact beta_c = {beta_c_exact:.4f}\n")
print(f"{'beta':>8} {'<|m|>':>8} {'chi':>10} {'<E>/N':>10} {'C_v':>10}")
print("-" * 50)

for idx, beta in enumerate(betas):
    model = IsingModel(L=L, beta=beta)
    mags, energies = model.simulate(n_equil=300, n_sample=500)

    m_mean = np.mean(mags)
    chi = model.susceptibility_from_fluctuations(mags)
    e_mean = np.mean(energies)
    c_v = beta ** 2 * L ** 2 * np.var(energies)

    # Free energy proxy: F ~ <E> - (1/beta) * S
    # Using F ~ -ln(Z)/N approximated from energy and entropy
    entropy_proxy = np.log(np.std(mags) + 1e-10)
    f_proxy = e_mean - entropy_proxy / beta if beta > 0 else e_mean

    results["magnetization"][idx] = m_mean
    results["susceptibility"][idx] = chi
    results["energy"][idx] = e_mean
    results["specific_heat"][idx] = c_v
    results["free_energy_proxy"][idx] = f_proxy

    print(f"{beta:>8.3f} {m_mean:>8.3f} {chi:>10.2f} {e_mean:>10.4f} {c_v:>10.2f}")


# ── Visualization ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# (a) Magnetization (order parameter)
axes[0, 0].plot(betas, results["magnetization"], "o-", color="steelblue", markersize=4)
axes[0, 0].axvline(beta_c_exact, color="red", linestyle="--", label=r"$\beta_c$ (exact)")
axes[0, 0].set_xlabel(r"$\beta$ (inverse temperature)")
axes[0, 0].set_ylabel(r"$\langle |m| \rangle$")
axes[0, 0].set_title("(a) Order Parameter")
axes[0, 0].legend()

# (b) Susceptibility (diverges at criticality)
axes[0, 1].plot(betas, results["susceptibility"], "s-", color="darkorange", markersize=4)
axes[0, 1].axvline(beta_c_exact, color="red", linestyle="--", label=r"$\beta_c$ (exact)")
axes[0, 1].set_xlabel(r"$\beta$ (inverse temperature)")
axes[0, 1].set_ylabel(r"$\chi$")
axes[0, 1].set_title("(b) Susceptibility (Fisher Information)")
axes[0, 1].legend()

# (c) Specific heat
axes[1, 0].plot(betas, results["specific_heat"], "^-", color="forestgreen", markersize=4)
axes[1, 0].axvline(beta_c_exact, color="red", linestyle="--", label=r"$\beta_c$ (exact)")
axes[1, 0].set_xlabel(r"$\beta$ (inverse temperature)")
axes[1, 0].set_ylabel(r"$C_v$")
axes[1, 0].set_title("(c) Specific Heat")
axes[1, 0].legend()

# (d) Free energy proxy
axes[1, 1].plot(betas, results["free_energy_proxy"], "d-", color="purple", markersize=4)
axes[1, 1].axvline(beta_c_exact, color="red", linestyle="--", label=r"$\beta_c$ (exact)")
axes[1, 1].set_xlabel(r"$\beta$ (inverse temperature)")
axes[1, 1].set_ylabel("Free Energy Proxy")
axes[1, 1].set_title("(d) Variational Free Energy")
axes[1, 1].legend()

plt.suptitle(f"2D Ising Model Phase Transition (L={L})", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("ising_critical_phenomena.png", dpi=150)
plt.show()


# ── Avalanche analysis at criticality ─────────────────────────────
def measure_avalanches(L: int = 64, beta: float = None, n_avalanches: int = 5000):
    """
    Measure avalanche size distribution at the critical point.
    An 'avalanche' is triggered by flipping one spin and counting
    the cascade of subsequent flips.
    """
    if beta is None:
        beta = beta_c_exact

    model = IsingModel(L=L, beta=beta)
    # Equilibrate
    for _ in range(500):
        model.metropolis_step()

    sizes = []
    for _ in range(n_avalanches):
        # Perturb one spin
        i, j = np.random.randint(0, L, size=2)
        model.spins[i, j] *= -1

        # Count cascade via relaxation steps
        cascade_size = 0
        for _ in range(50):  # max cascade length
            flipped = 0
            for _ in range(L * L // 10):  # partial sweep
                ii, jj = np.random.randint(0, L, size=2)
                s = model.spins
                nn_sum = (
                    s[(ii + 1) % L, jj] + s[(ii - 1) % L, jj] +
                    s[ii, (jj + 1) % L] + s[ii, (jj - 1) % L]
                )
                dE = 2.0 * s[ii, jj] * nn_sum
                if dE < 0:
                    s[ii, jj] *= -1
                    flipped += 1
            cascade_size += flipped
            if flipped == 0:
                break
        if cascade_size > 0:
            sizes.append(cascade_size)

    return np.array(sizes)


print("\n=== Avalanche Analysis at Criticality ===")
avalanche_sizes = measure_avalanches(L=32, n_avalanches=2000)
if len(avalanche_sizes) > 100:
    # Log-binned histogram
    bins = np.logspace(0, np.log10(avalanche_sizes.max()), 30)
    hist, edges = np.histogram(avalanche_sizes, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = hist > 0

    # Fit power law exponent
    log_s = np.log10(centers[mask])
    log_p = np.log10(hist[mask])
    slope, intercept = np.polyfit(log_s, log_p, 1)
    print(f"Power-law exponent (avalanche size): tau = {-slope:.2f}")
    print(f"  (expected for mean-field branching: tau = 1.50)")
```

## Diagnostic Table: Signatures of Criticality

| Observable | Subcritical | Critical | Supercritical | Measurement |
|------------|-------------|----------|---------------|-------------|
| Correlation length $\xi$ | Short (finite) | Divergent ($\to \infty$) | Short (finite) | Spatial correlation decay |
| Susceptibility $\chi$ | Low | Divergent (peak) | Low | Response to perturbation |
| Avalanche size dist. | Exponential cutoff | Power law $P(s) \sim s^{-\tau}$ | Bimodal (system-spanning) | Event size histogram |
| Autocorrelation time | Short | Divergent (critical slowing) | Short | Temporal correlation decay |
| Mutual information | Low | Maximal | Low | Info between input/output |
| Dynamic range | Narrow | Maximal | Narrow | Range of discriminable inputs |
| Branching ratio $\sigma$ | $\sigma < 1$ | $\sigma = 1$ | $\sigma > 1$ | Propagation of activity |
| Variational free energy | High (poor fit) | Minimal (optimal) | High (unstable) | ELBO computation |

## Interpretation: Why Criticality Matters for the FEP

The connection between criticality and the FEP can be summarized in three principles:

1. **Criticality maximizes model evidence.** At the critical point, the generative model achieves the optimal accuracy-complexity tradeoff. The prior is maximally non-committal (high entropy, low complexity cost), while the likelihood achieves maximal sensitivity (divergent susceptibility, high accuracy).

2. **Self-organized criticality is free energy minimization.** The self-tuning mechanism that drives SOC systems to the critical point can be understood as gradient descent on variational free energy across the space of control parameters.

3. **Universality reflects model compression.** The fact that many different microscopic systems share the same critical behavior (universality) reflects the fact that coarse-grained generative models -- those that capture only the relevant (slow, large-scale) degrees of freedom -- converge to the same form at the RG fixed point.

## References

1. Beggs, J. M., & Plenz, D. (2003). Neuronal avalanches in neocortical circuits. *Journal of Neuroscience*, 23(35), 11167-11177.
2. Bak, P., Tang, C., & Wiesenfeld, K. (1987). Self-organized criticality: An explanation of the 1/f noise. *Physical Review Letters*, 59(4), 381.
3. Shew, W. L., & Plenz, D. (2013). The functional benefits of criticality in the cortex. *The Neuroscientist*, 19(1), 88-100.
4. Friston, K. J. (2019). A free energy principle for a particular physics. *arXiv:1906.10184*.
5. Mora, T., & Bialek, W. (2011). Are biological systems poised at criticality? *Journal of Statistical Physics*, 144, 268-302.
6. Tkacik, G., et al. (2015). Thermodynamics and signatures of criticality in a network of neurons. *Proceedings of the National Academy of Sciences*, 112(37), 11508-11513.
7. Hesse, J., & Gross, T. (2014). Self-organized criticality as a fundamental property of neural systems. *Frontiers in Systems Neuroscience*, 8, 166.
8. Hidalgo, J., et al. (2014). Information-based fitness and the emergence of criticality in living systems. *Proceedings of the National Academy of Sciences*, 111(28), 10095-10100.
9. Munoz, M. A. (2018). Colloquium: Criticality and dynamical scaling in living systems. *Reviews of Modern Physics*, 90(3), 031001.
10. Wilson, K. G. (1971). Renormalization group and critical phenomena. *Physical Review B*, 4(9), 3174.
11. Onsager, L. (1944). Crystal statistics. I. A two-dimensional model with an order-disorder transition. *Physical Review*, 65(3-4), 117.

## See Also

- [[self_organization|Self-Organization]] -- Self-organizing dynamics leading to criticality
- [[emergence|Emergence]] -- Emergent properties at phase transitions
- [[network_dynamics|Network Dynamics]] -- Network topology and critical coupling
- [[resilience|Resilience]] -- Stability near critical tipping points
- [[complex_adaptation|Complex Adaptation]] -- Adaptive benefits of near-critical operation
- [[knowledge_base/free_energy_principle/mathematics/core_principle|Core Principle]] -- Foundational free energy formalism
- [[knowledge_base/free_energy_principle/biology/homeostasis|Homeostasis]] -- Homeostatic regulation near critical regimes
