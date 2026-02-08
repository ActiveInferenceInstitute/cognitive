---
title: "Information Geometry and the Free Energy Principle"
type: mathematical_concept
status: active
created: 2025-01-01
updated: 2025-02-06
tags:
  - free_energy_principle
  - information_geometry
  - fisher_information
  - natural_gradient
  - statistical_manifold
  - riemannian_geometry
semantic_relations:
  - type: foundation
    links:
      - [[core_principle|Core Principle]]
      - [[variational_free_energy|Variational Free Energy]]
  - type: relates
    links:
      - [[markov_blankets|Markov Blankets]]
      - [[advanced_formulations|Advanced Formulations]]
      - [[knowledge_base/free_energy_principle/cognitive/perception|Perception]]
  - type: extends
    links:
      - [[expected_free_energy|Expected Free Energy]]
---

# Information Geometry and the Free Energy Principle

## Introduction

Information geometry provides the natural mathematical language for understanding the Free Energy Principle. It treats probability distributions as points on a curved manifold (the **statistical manifold**) equipped with a Riemannian metric derived from the Fisher information. Free energy minimization, in this geometric picture, becomes gradient descent on a curved surface, and the KL divergence becomes a measure of "distance" between probability distributions.

This geometric perspective reveals deep structural connections between inference, thermodynamics, and differential geometry that are not apparent in the purely analytical formulation of the FEP.

## Statistical Manifolds

### Definition

A **statistical manifold** `M` is a smooth manifold whose points correspond to probability distributions. If we parameterize a family of distributions by `theta = (theta_1, ..., theta_n)`, then each `theta` specifies a distribution `p(x | theta)`, and the set of all such distributions forms the manifold.

```
M = {p(x | theta) : theta in Theta subset R^n}
```

**Examples**:
- **Gaussian manifold**: `M = {N(mu, sigma^2) : mu in R, sigma > 0}` -- a 2D manifold parameterized by mean and variance
- **Categorical manifold**: `M = {Cat(p_1, ..., p_k) : p_i >= 0, sum p_i = 1}` -- a (k-1)-dimensional simplex
- **Exponential family**: `M = {p(x|theta) = h(x) exp(theta . T(x) - A(theta))}` -- natural parameterization

### The Fisher Information Metric

The statistical manifold is equipped with a natural Riemannian metric: the **Fisher information matrix**.

```
g_ij(theta) = E_p(x|theta)[partial_i ln p(x|theta) * partial_j ln p(x|theta)]
             = -E_p(x|theta)[partial_i partial_j ln p(x|theta)]
```

Where `partial_i = partial / partial theta_i`.

**Properties**:
- Positive semi-definite (and typically positive definite)
- Invariant under reparameterization (up to the usual tensor transformation law)
- Unique (up to scaling) metric that is invariant under sufficient statistics (Cencov's theorem)

**For Gaussians** `N(mu, sigma^2)`:
```
g = [[1/sigma^2, 0], [0, 2/sigma^4]]
```

The geometry is hyperbolic -- the space of Gaussian distributions has negative curvature. Distributions with small variance are "far apart" in the Fisher metric even if their means are close.

**For the FEP**: The Fisher information metric on the space of recognition densities defines the geometry of the inference problem. Free energy minimization follows geodesics (or natural gradients) on this manifold.

### Geodesics and Distances

The geodesic distance between two distributions `p(x|theta_1)` and `p(x|theta_2)` is:

```
d(theta_1, theta_2) = integral_0^1 sqrt(g_{ij}(theta(t)) * dtheta_i/dt * dtheta_j/dt) dt
```

Where `theta(t)` is the geodesic path connecting `theta_1` to `theta_2`.

For exponential families, geodesics have elegant closed-form expressions. The `e-geodesic` (exponential geodesic) connects distributions through mixture in the natural parameter space; the `m-geodesic` (mixture geodesic) connects through mixture in the expectation parameter space.

## KL Divergence as Divergence Function

### Definition and Geometry

The KL divergence is NOT a distance (it is asymmetric and does not satisfy the triangle inequality), but it is a **divergence function** -- a generalized notion of separation on the statistical manifold.

```
D_KL[p(x|theta) || p(x|theta')] = E_{p(x|theta)}[ln p(x|theta) / p(x|theta')]
```

The KL divergence induces the Fisher metric through its Hessian:

```
g_{ij}(theta) = partial^2 D_KL[p(x|theta) || p(x|theta')] / partial theta'_i partial theta'_j |_{theta'=theta}
```

This means the Fisher metric captures the **local** behavior of KL divergence: for nearby distributions (small `delta_theta`):

```
D_KL[p(x|theta) || p(x|theta + delta_theta)] approx 1/2 * delta_theta^T * g(theta) * delta_theta
```

### Dual Structure

Information geometry reveals a fundamental **duality** in the structure of the statistical manifold. There are two affine connections:

- **e-connection** (exponential): the natural connection for exponential families
- **m-connection** (mixture): the natural connection for mixture families

These are dual with respect to the Fisher metric:

```
g(nabla^e_X Y, Z) + g(Y, nabla^m_X Z) = X * g(Y, Z)
```

**Relevance to FEP**: The e-connection corresponds to updating beliefs in the natural parameter space (gradient descent on free energy); the m-connection corresponds to updating in the expectation parameter space. The duality between these two perspectives maps onto the duality between energy and entropy in the thermodynamic decomposition of free energy.

## Natural Gradient Descent

### The Problem with Ordinary Gradients

Ordinary gradient descent on free energy uses the Euclidean gradient:

```
theta_{t+1} = theta_t - eta * nabla_theta F
```

But the Euclidean gradient is not invariant under reparameterization. The same update in different coordinate systems produces different results. This is a problem because the "true" optimization landscape is the statistical manifold, which has intrinsic curvature that Euclidean gradients ignore.

### Natural Gradient

The **natural gradient** (Amari, 1998) accounts for the geometry of the statistical manifold:

```
theta_{t+1} = theta_t - eta * g(theta)^{-1} * nabla_theta F
```

Where `g(theta)^{-1}` is the inverse Fisher information matrix. This is equivalent to steepest descent on the statistical manifold -- the direction of maximum change in F per unit change in the Fisher metric.

**Properties**:
- **Reparameterization invariant**: Same update regardless of coordinate system
- **Efficient**: Converges faster than ordinary gradient descent, especially for ill-conditioned problems
- **Fisher-efficient**: At the optimum, the natural gradient achieves the Cramer-Rao lower bound

### Natural Gradient and the FEP

Under the FEP, perception (updating the recognition density) proceeds via:

```
dmu/dt = -g(mu)^{-1} * nabla_mu F
```

This is natural gradient descent on variational free energy. The Fisher information metric automatically provides the correct step size: large steps where the curvature is low (uncertain beliefs can change freely), small steps where the curvature is high (certain beliefs resist change).

**Neural interpretation**: The Fisher information matrix captures the sensitivity of neural responses to changes in encoded variables. Natural gradient descent means the brain uses this sensitivity to scale its updates optimally -- a biologically plausible mechanism that requires only local information.

For Gaussian recognition densities `q(s) = N(mu, Sigma)`:

```
g = Sigma^{-1} = Pi  (the precision matrix)
```

So the natural gradient update becomes:

```
dmu/dt = -Pi^{-1} * nabla_mu F = -Sigma * nabla_mu F
```

Predictions are updated proportionally to uncertainty -- the less certain you are, the more you update. This is exactly what Kalman filtering and predictive coding produce.

## Free Energy on the Statistical Manifold

### Geometric Interpretation

On the statistical manifold, free energy has a clear geometric interpretation:

```
F[q] = D_KL[q(s) || p(s|o)] - ln p(o)
```

- `q(s)` is a point on the manifold of recognition densities
- `p(s|o)` is a point on the same manifold (the true posterior)
- `D_KL[q || p(s|o)]` is the divergence from q to the true posterior
- Minimizing F moves q toward p(s|o) along the natural gradient direction

The trajectory of free energy minimization traces a path on the statistical manifold, and the natural gradient ensures this path is a geodesic (in the appropriate geometry).

### The Projection Interpretation

When the family of recognition densities is restricted (e.g., mean-field or Gaussian), the minimizer of F is the **information projection** (I-projection) of the true posterior onto the restricted family:

```
q* = argmin_{q in Q} D_KL[q || p(s|o)]
```

Where `Q` is the restricted family. This is the distribution in `Q` that is closest to the true posterior in KL divergence.

The I-projection has the property of **moment matching**: for exponential family Q, the I-projection matches the expected sufficient statistics of the true posterior. For Gaussian Q, this means q* matches the mean and covariance of the true posterior (Laplace approximation).

### Dual Projections

There are two types of projections:

- **I-projection** (information projection): `q* = argmin_{q in Q} D_KL[q || p]` -- used in variational inference (FEP)
- **M-projection** (moment projection): `q* = argmin_{q in Q} D_KL[p || q]` -- used in expectation propagation

These correspond to the e-connection and m-connection, respectively. The FEP uses the I-projection, which is the one that minimizes the free energy bound.

## Fisher Information and Precision

### Relationship

In the FEP, precision (inverse variance) plays a central role. Fisher information and precision are deeply connected:

For a Gaussian likelihood `p(o|s) = N(g(s), Sigma_o)`:

```
Fisher information about s from o:
I(s) = J^T * Sigma_o^{-1} * J = J^T * Pi_o * J
```

Where `J = partial g / partial s` is the Jacobian and `Pi_o = Sigma_o^{-1}` is the observation precision.

**Interpretation**: The Fisher information about hidden states, as carried by observations, is the observation precision weighted by the sensitivity of observations to states. More precise observations (lower noise) and more sensitive mappings (larger Jacobian) provide more information.

### Cramer-Rao Bound

The Fisher information sets a fundamental limit on estimation accuracy:

```
Var(hat{s}) >= I(s)^{-1}
```

No unbiased estimator can have variance less than the inverse Fisher information. In the FEP context:

- The recognition density's precision `Pi_q` cannot exceed the Fisher information `I(s)`
- Optimal inference achieves `Pi_q = I(s)` -- the Cramer-Rao bound
- Predictive coding achieves this bound under Gaussian assumptions

### Precision and Attention

The Fisher information matrix depends on the observation model parameters, which can themselves be inferred. This creates a second-order optimization:

```
Level 1: Infer states s by minimizing F w.r.t. q(s)
Level 2: Infer precision Pi by minimizing F w.r.t. q(Pi)
```

Level 2 corresponds to **attention** -- the brain optimizes the gain (precision) of prediction error units to weight reliable information channels more heavily. In information-geometric terms, the brain is selecting which region of the statistical manifold to concentrate its inference in.

## Connection to Thermodynamics

### Stochastic Thermodynamics and Information Geometry

Recent work connects information geometry, stochastic thermodynamics, and the FEP:

**Entropy production** in a stochastic system can be written as:

```
Sigma_dot = dD_KL[p(x,t) || p_ss(x)] / dt + Q
```

Where `p_ss` is the steady-state distribution and `Q` is the heat dissipation rate.

In information-geometric terms, entropy production is related to the Fisher-Rao metric length of the path traced by the probability distribution through the statistical manifold:

```
Sigma_dot >= (dL/dt)^2 / (2 * D)
```

Where `L` is the Fisher-Rao length and `D` is a diffusion coefficient. This is the **thermodynamic uncertainty relation**.

**For the FEP**: The rate of free energy minimization (how fast the organism updates its beliefs) is bounded by the entropy production (the thermodynamic cost of inference). This provides a physical basis for the computational cost of inference.

### The Geometry of Non-Equilibrium Steady States

At a non-equilibrium steady state (NESS), the system traces a closed orbit on the statistical manifold (due to the solenoidal flow component). The area enclosed by this orbit in the Fisher metric is related to the entropy production:

```
Sigma_ss = integral_orbit Q * nabla ln p . ds
```

This geometric picture reveals that:
- Systems at equilibrium sit at a fixed point on the manifold (no solenoidal flow)
- Living systems (NESS) constantly orbit on the manifold, producing entropy
- The FEP describes how these orbits are structured: they minimize free energy while maintaining the non-equilibrium orbit

## Alpha-Divergences and Generalized Free Energies

### Alpha-Divergence Family

The KL divergence is a special case of the **alpha-divergence** family:

```
D_alpha[p || q] = (4 / (1 - alpha^2)) * (1 - integral p^((1+alpha)/2) * q^((1-alpha)/2) dx)
```

Special cases:
- `alpha = 1`: `D_1[p || q] = D_KL[p || q]` (standard KL)
- `alpha = -1`: `D_{-1}[p || q] = D_KL[q || p]` (reverse KL)
- `alpha = 0`: `D_0[p || q]` = Hellinger distance (squared)

### Generalized Free Energies

Each alpha value yields a different free energy functional:

```
F_alpha = D_alpha[q(s) || p(s|o)] + const
```

The standard FEP uses `alpha = 1` (I-projection). Different values trade off:
- **alpha = 1** (FEP standard): Mode-seeking. q tends to cover one mode of p. Underestimates variance.
- **alpha = -1** (moment projection): Mean-seeking. q tends to cover all modes of p. Overestimates variance.
- **alpha = 0** (Hellinger): Symmetric, balanced compromise.

The choice of alpha may vary across brain regions or cognitive functions -- perception might use `alpha = 1` (seeking the most likely interpretation), while planning might use `alpha = -1` (considering all possibilities).

## Amari's Embedding and the FEP

### The Exponential Family Embedding

For recognition densities in the exponential family:

```
q(s | eta) = h(s) * exp(eta . T(s) - A(eta))
```

Where `eta` are natural parameters, `T(s)` are sufficient statistics, and `A(eta)` is the log-partition function.

The expectation parameters are:
```
mu = E_q[T(s)] = nabla_eta A(eta)
```

**Key insight**: The mapping `eta <-> mu` is a Legendre transform, and:

```
A(eta) + A*(mu) = eta . mu
```

Where `A*(mu)` is the convex conjugate (negative entropy). This duality is fundamental to both information geometry and thermodynamics (Legendre transforms connect energy and entropy representations).

### Free Energy in Natural Parameters

```
F(eta) = D_KL[q(s|eta) || p(s)] - E_{q(eta)}[ln p(o|s)]
       = A(eta) - eta . mu_p + A*(mu_p) - E_{q(eta)}[ln p(o|s)]
```

The natural gradient in eta space:

```
deta/dt = -nabla_eta F = -(mu - mu_p) + nabla_eta E_{q(eta)}[ln p(o|s)]
```

This is a prediction error: the difference between expected sufficient statistics under q and under the prior, plus a data-driven correction. This directly implements predictive coding in the natural parameter space.

## Practical Implications

### For Neural Network Implementation

Natural gradient descent in neural networks is approximated by various methods:
- **Fisher diagonal**: Approximate `g^{-1}` with only diagonal entries
- **K-FAC**: Kronecker-factored approximate curvature
- **ADAM optimizer**: Implicitly approximates natural gradient through adaptive learning rates

For deep active inference, natural gradient methods improve convergence speed and stability.

### For Understanding Neural Computation

Information geometry suggests:
1. Neural populations encode points on statistical manifolds
2. Neural dynamics implement natural gradient flows
3. Precision weighting is the neural implementation of the Fisher metric
4. Attention mechanisms optimize the local geometry of inference

## Key References

1. Amari, S. (2016). *Information Geometry and Its Applications*. Springer.
2. Amari, S. (1998). Natural gradient works efficiently in learning. *Neural Computation*, 10(2), 251-276.
3. Ay, N., Jost, J., Le, H. V., & Schwachhofer, L. (2017). *Information Geometry*. Springer.
4. Parr, T., Da Costa, L., & Friston, K. (2020). Markov blankets, information geometry and stochastic thermodynamics. *Philosophical Transactions of the Royal Society A*, 378(2164), 20190159.
5. Caticha, A. (2015). The basics of information geometry. In *AIP Conference Proceedings* (Vol. 1641, pp. 15-26).
6. Nielsen, F. (2020). An elementary introduction to information geometry. *Entropy*, 22(10), 1100.
7. Da Costa, L., Parr, T., Sengupta, B., & Friston, K. (2021). Neural dynamics under active inference: plausibility and efficiency of energy minimization. *Entropy*, 23(4), 454.
