---
title: Random Walk
description: A process built from successive random steps, its diffusive scaling, recurrence properties, biased and correlated variants, and use as a null model for animal and collective movement and for directed search.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - stochastic_processes
  - diffusion
  - movement_ecology
semantic_relations:
  - type: relates
    links:
      - "[[stochastic_processes]]"
      - "[[wiener_process]]"
      - "[[diffusion_processes]]"
      - "[[foraging_optimization]]"
      - "[[collective_behavior_ants]]"
      - "[[markov_chain_monte_carlo]]"
---

# Random Walk

## Overview
A random walk is a sequence of positions $S_n=S_0+X_1+\dots+X_n$ in which each step $X_i$ is drawn at random. In the simplest case the steps are independent and identically distributed, for instance $\pm1$ with equal probability on the integers. Random walks are the basic discrete model of undirected movement, diffusion and accumulated noise.

## Scaling and limits
For independent steps with mean zero and variance $\sigma^2$, the expected position remains at its starting point while the variance grows linearly, $\mathrm{Var}(S_n)=n\sigma^2$. Typical displacement therefore scales as $\sqrt{n}$, which is diffusive spreading and is slower than the linear growth of directed motion. By the central limit theorem, $S_n/\sqrt{n}$ is approximately Gaussian, and Donsker's theorem shows that, suitably rescaled, the walk converges to Brownian motion, which is the Wiener process. The corresponding continuum equation is the diffusion equation $\partial_t p=D\nabla^2p$.

## Recurrence
Pólya showed in 1921 that the simple symmetric walk on the integer lattice returns to its starting point with probability one in one and two dimensions, so it is recurrent, and with probability less than one in three or more dimensions, so it is transient. Hitting times, first-passage probabilities and the gambler's ruin problem are standard applications.

## Variants
- **Biased walks** have a nonzero mean step and drift.
- **Correlated random walks** let step directions persist, giving a short-time ballistic regime before diffusive scaling returns.
- **Lévy walks** draw step lengths from heavy-tailed distributions, which produce super-diffusive spreading and have been proposed as efficient search strategies when targets are sparse and unpredictable.
- **Walks on graphs** underlie Markov chain models, PageRank and the sampling methods of Markov chain Monte Carlo.

## Relation to Active Inference
A random walk is a null model, against which directed behaviour can be compared. Searching without information about where resources are corresponds to a random or Lévy walk, whereas an Active Inference agent uses its beliefs to select moves that resolve uncertainty or approach preferred states, an approach often called epistemic foraging. In models of ants and other animals, stochastic exploration combined with feedback from pheromone trails is a classic example of undirected search that becomes directed through the environment. The link is a comparison of modelling choices, not a derivation.

## Related concepts
- [[wiener_process]] — the continuous-time limit of the random walk
- [[stochastic_processes]] — the general theory of random sequences
- [[diffusion_processes]] — continuum description of random spreading
- [[foraging_optimization]] — search strategies that improve on random movement
- [[collective_behavior_ants]] — insect movement models that use random exploration

## References
- Feller, W. (1968). An Introduction to Probability Theory and Its Applications, Volume 1 (3rd ed.). Wiley.
- Berg, H. C. (1993). Random Walks in Biology (expanded ed.). Princeton University Press.
- Codling, E. A., Plank, M. J., & Benhamou, S. (2008). Random walk models in biology. Journal of the Royal Society Interface, 5, 813-834.
