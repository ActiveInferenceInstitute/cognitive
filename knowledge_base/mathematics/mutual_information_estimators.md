---
title: Mutual Information Estimators
description: "Survey of mutual information estimators, including plug-in histogram, nearest-neighbour KSG and variational bounds such as MINE and InfoNCE, and their use in Active Inference."
type: mathematical_concept
status: draft
created: 2025-08-08
updated: 2026-10-09
tags:
- information-theory
- estimation
semantic_relations:
  - type: relates
    links:
      - "[[mutual_information]]"
      - "[[epistemic_value]]"
  - type: relates_to
    links:
      - "[[entropy]]"
      - "[[kl_divergence]]"
      - "[[expected_free_energy]]"
---


# Mutual Information Estimators

[[mutual_information]] has a closed form when the joint distribution is known, as it is in the discrete generative models of this repository, where $I(s;o)$ follows directly from the matrices $A$ and $q(s)$. Estimators matter when only samples are available: logged agent trajectories, neural recordings, or a model whose likelihood cannot be evaluated. All of them trade bias against variance, and none is free of assumptions.

## Plug-in (histogram) estimator

Count joint occurrences, normalise to $\hat p(x,y)$, and evaluate $\hat I = \sum \hat p \ln\frac{\hat p(x,y)}{\hat p(x)\hat p(y)}$. It is consistent but biased upwards, because sampling noise makes variables look dependent. For independent variables with alphabet sizes $k_x, k_y$ and $N$ samples, the expected bias is approximately $(k_x - 1)(k_y - 1)/(2N)$ nats. The Miller-Madow correction subtracts this term: $\hat I_{\mathrm{MM}} = \hat I - (k_x-1)(k_y-1)/(2N)$.

A simulation with two independent uniform variables on 4 symbols each (2000 repetitions) gives a mean plug-in estimate of $0.0474$ nats at $N = 100$, against the predicted $0.045$, and $0.0045$ at $N = 1000$, against $0.0045$. The true value is zero in both cases. The bias is negligible only when $N \gg k_x k_y$.

## Nearest-neighbour estimators (KSG)

For continuous variables the Kraskov-Stoegbauer-Grassberger estimator uses distances to the $k$-th nearest neighbour in the joint space and counts neighbours in each marginal, avoiding explicit density estimation. It has small bias for moderate dimension but degrades as dimension grows, and it needs a distance metric on which both variables are comparable.

## Variational (neural) lower bounds

These estimators optimise a critic $T(x,y)$ and return a lower bound on $I$.

| bound | expression | optimal critic |
|-------|-----------|----------------|
| Donsker-Varadhan (MINE) | $\mathbb{E}_{p(x,y)}[T] - \ln \mathbb{E}_{p(x)p(y)}[e^{T}]$ | $T^\ast = \ln \frac{p(x,y)}{p(x)p(y)} + c$ |
| NWJ | $\mathbb{E}_{p(x,y)}[T] - \mathbb{E}_{p(x)p(y)}[e^{T-1}]$ | $T^\ast = 1 + \ln \frac{p(x,y)}{p(x)p(y)}$ |
| InfoNCE | $\ln N + \mathbb{E}\big[\ln \frac{e^{T(x_i,y_i)}}{\sum_j e^{T(x_i,y_j)}}\big]$ | log density ratio up to a constant |

On the joint $\begin{pmatrix}0.4 & 0.1\\0.1 & 0.4\end{pmatrix}$ (true $I = 0.1927$), both Donsker-Varadhan and NWJ return exactly $0.1927$ at their optimal critics. With the suboptimal critic $T = \tfrac12 \ln\frac{p(x,y)}{p(x)p(y)}$ the Donsker-Varadhan value drops to $0.1491$, and it can never exceed the truth. InfoNCE cannot report more than $\ln N$, so a large $I$ needs a large batch. Donsker-Varadhan and NWJ estimates have high variance when the true $I$ is large, because the exponential term is dominated by rare pairs.

## Use in Active Inference

In the discrete models here, the information gain is computed exactly. `DiscreteGenerativeModel.expected_free_energy` returns it as its fourth value, derived from $A$ and the predicted state distribution, so no estimator is needed. Estimators become relevant for continuous or sampled generative models, where the exact expression in [[epistemic_value]] is unavailable, and for diagnostics on recorded behaviour, such as how much an agent's observations told it about a hidden state. Treat any estimated epistemic value as a noisy proxy: biased estimates can reorder policies whose true information gains are close.

## Reproducing the plug-in numbers

```python
import numpy as np

rng = np.random.default_rng(0)


def plug_in_mi(x, y, kx, ky):
    counts = np.zeros((kx, ky))
    np.add.at(counts, (x, y), 1)
    p = counts / counts.sum()
    outer = p.sum(axis=1, keepdims=True) @ p.sum(axis=0, keepdims=True)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / outer[mask])))


for n in (100, 1000):
    estimates = [
        plug_in_mi(rng.integers(0, 4, n), rng.integers(0, 4, n), 4, 4) for _ in range(2000)
    ]
    print(n, round(float(np.mean(estimates)), 4), "predicted bias", 9 / (2 * n))

# Variational bounds at the optimal critic for a known joint.
joint = np.array([[0.4, 0.1], [0.1, 0.4]])
marginals = np.outer(joint.sum(axis=1), joint.sum(axis=0))
ratio = np.log(joint / marginals)
dv = np.sum(joint * ratio) - np.log(np.sum(marginals * np.exp(ratio)))
t_nwj = 1.0 + ratio
nwj = np.sum(joint * t_nwj) - np.sum(marginals * np.exp(t_nwj - 1.0))
half = 0.5 * ratio
dv_half = np.sum(joint * half) - np.log(np.sum(marginals * np.exp(half)))
print(round(float(dv), 4), round(float(nwj), 4), round(float(dv_half), 4))
```

## See also

- [[mutual_information]] · [[epistemic_value]] · [[entropy]] · [[kl_divergence]]

## References

- Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information. Physical Review E, 69(6), 066138.
- Paninski, L. (2003). Estimation of entropy and mutual information. Neural Computation, 15(6), 1191-1253.
- Nguyen, X., Wainwright, M. J., & Jordan, M. I. (2010). Estimating divergence functionals and the likelihood ratio by convex risk minimization. IEEE Transactions on Information Theory, 56(11), 5847-5861.
- Belghazi, M. I., Baratin, A., Rajeshwar, S., Ozair, S., Bengio, Y., Courville, A., & Hjelm, R. D. (2018). Mutual information neural estimation. Proceedings of the 35th International Conference on Machine Learning (PMLR 80), 531-540.
- Poole, B., Ozair, S., van den Oord, A., Alemi, A. A., & Tucker, G. (2019). On variational bounds of mutual information. Proceedings of the 36th International Conference on Machine Learning (PMLR 97), 5171-5180.
- van den Oord, A., Li, Y., & Vinyals, O. (2018). Representation learning with contrastive predictive coding. arXiv:1807.03748.
