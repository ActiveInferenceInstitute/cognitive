---
title: Ant Colony Implementation Guide
type: guide
status: stable
created: 2025-01-01
updated: 2026-02-07
tags:
  - ant_colony
  - swarm_intelligence
  - multi_agent
  - active_inference
semantic_relations:
  - type: relates_to
    links:
      - [[code/Things/Ant_Colony/AGENTS]]
      - [[knowledge_base/cognitive/swarm_intelligence]]
      - [[knowledge_base/biology/README]]
---

# Ant Colony Implementation Guide

## Overview

The Ant Colony system demonstrates **swarm intelligence** through Active Inference, modeling how individual agents with simple generative models produce emergent collective behavior through stigmergic coordination — communication via environmental modification (pheromone trails).

## Architecture

### Colony Structure

```
code/Things/Ant_Colony/
├── ant_colony/
│   ├── main.py              # Entry point and simulation runner
│   ├── agents/              # Individual ant agent implementations
│   └── environment/         # Grid world with pheromone dynamics
├── config/
│   └── colony_config.yaml   # Colony parameters and simulation settings
└── visualization/           # Real-time colony visualization
```

### Agent Model

Each ant agent implements Active Inference with:

- **Observations**: Local pheromone concentrations, food/nest proximity, neighboring ants
- **Hidden States**: Estimated position relative to food sources and nest
- **Preferences**: Finding food (when foraging) or returning to nest (when carrying food)
- **Actions**: Movement in cardinal directions, pheromone deposition

### Stigmergic Coordination

Ants communicate indirectly through pheromone trails:

1. **Exploration phase**: Ants random-walk, depositing weak exploration pheromone
2. **Discovery phase**: Upon finding food, ants deposit strong food-trail pheromone
3. **Recruitment phase**: Other ants follow high-concentration trails
4. **Optimization phase**: Shorter paths accumulate more pheromone (positive feedback)
5. **Evaporation**: Pheromone decays over time, pruning suboptimal paths

## Running the Simulation

```bash
# Basic run
python3 code/Things/Ant_Colony/ant_colony/main.py

# With custom configuration
python3 code/Things/Ant_Colony/ant_colony/main.py \
    --config code/Things/Ant_Colony/config/colony_config.yaml
```

### Configuration Parameters

Key parameters in `colony_config.yaml`:

| Parameter | Description | Default |
|---|---|---|
| `num_ants` | Number of agents in the colony | 50 |
| `grid_size` | Size of the environment grid | 100×100 |
| `pheromone_decay` | Evaporation rate per timestep | 0.01 |
| `pheromone_deposit` | Amount deposited per step | 1.0 |
| `num_food_sources` | Number of food patches | 3 |
| `max_steps` | Simulation duration | 1000 |

## Key Concepts

### Free Energy in Swarm Systems

In the colony, free energy minimization operates at two levels:

- **Individual**: Each ant minimizes surprise by following pheromone gradients and seeking expected observations (food near high pheromone)
- **Collective**: The colony as a whole converges on efficient foraging strategies through emergent path optimization

### Precision Weighting

Ants modulate precision (confidence) based on context:

- High precision on pheromone signals → exploit known trails
- Low precision → explore novel territory

## Related Resources

- [[knowledge_base/cognitive/swarm_intelligence|Swarm Intelligence Theory]] — theoretical foundations
- [[code/Things/Ant_Colony/AGENTS|Ant Colony Agent Documentation]] — detailed agent specifications
- [[docs/guides/learning_paths/active_inference_myrmecology_learning_path|Myrmecology Learning Path]] — educational pathway
- [[docs/guides/agent_development|Agent Development Guide]] — general agent development patterns
