---
title: Cognitive Agents Index
type: agents
status: stable
created: 2026-02-06
tags:
  - agents
  - index
  - cognitive_modeling
semantic_relations:
  - type: organizes
    links:
      - [[code/Things/README]]
      - [[code/tools/src/models/active_inference/AGENTS]]
      - [[knowledge_base/AGENTS]]
---

# Cognitive Agents Index

This document serves as the central entry point for all agent architectures and implementations within the Cognitive Modeling Framework.

## 🏗️ Agent Logic & Frameworks

### Core Frameworks

- **Active Inference Agent** (`code/tools/src/models/active_inference/`): The primary implementation of the Active Inference agent, supporting belief updating, planning, and learning. [[code/tools/src/models/active_inference/AGENTS|View Documentation]]
- **Generic POMDP** (`knowledge_base/agents/GenericPOMDP/`): A flexible Partially Observable Markov Decision Process framework. [[knowledge_base/agents/AGENTS|View Documentation]]

### Specialized Implementations

- **Ant Colony** (`code/Things/Ant_Colony/`): Swarm intelligence and stigmergic coordination models. [[code/Things/Ant_Colony/AGENTS|View Documentation]]
- **BioFirm** (`knowledge_base/BioFirm/`): Biological firm theory and organizational active inference. [[knowledge_base/BioFirm/AGENTS|View Documentation]]
- **Continuous Time** (`knowledge_base/agents/Continuous_Time/`): Agents operating in continuous state-spaces. [[knowledge_base/agents/AGENTS|View Documentation]]

### Additional Implementations

- **Simple POMDP** (`code/Things/Simple_POMDP/`): Educational discrete Active Inference. [[code/Things/Simple_POMDP/AGENTS|View Documentation]]
- **Generic Thing** (`code/Things/Generic_Thing/`): Base active inference entity. [[code/Things/Generic_Thing/AGENTS|View Documentation]]

## 📚 Knowledge Base

For theoretical foundations and design patterns, refer to the **Knowledge Base**:

- [[knowledge_base/AGENTS|Active Inference Knowledge Base]]
- [[knowledge_base/cognitive/active_inference_agent|Active Inference Agent Theory]]
- [[knowledge_base/agents/architectures_overview|Agent Architectures Overview]]
- [[knowledge_base/agents/index|Agent Architecture Index]]

## 🛠️ Development Tools

- [[code/tools/AGENTS|Agent Development Tools]]: Utilities for building, testing, and visualizing agents.

---
> **Note**: Each subdirectory in `code/Things/` or `knowledge_base/` contains specific `AGENTS.md` and `README.md` files detailing the unique logic and usage of those agents.
