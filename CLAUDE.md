# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cognitive Modeling Framework built on Active Inference principles. The repository is both a Python codebase and an Obsidian knowledge vault, with extensive markdown documentation interconnected via `[[wiki-style links]]`.

**Dual license**: Code is MIT, documentation/knowledge base is CC BY-NC-SA 4.0.

## Repository Architecture

```
cognitive/
  code/
    tools/src/          # Core Python framework
      models/active_inference/  # ActiveInferenceModel base class, dispatcher
      models/matrices/          # Matrix operations (MatrixOps, MatrixInitializer)
      utils/                    # matrix_utils, create_node, visualization helpers
      visualization/            # MatrixPlotter
      analysis/                 # Analysis tools
    Things/              # Self-contained agent implementations (each has its own config)
      Generic_Thing/     # Base cognitive agent with Markov blankets
      Simple_POMDP/      # Discrete Active Inference (educational)
      Generic_POMDP/     # Extended POMDP framework
      Continuous_Generic/ # Continuous state-space models
      Ant_Colony/        # Swarm intelligence / stigmergy
      BioFirm/           # Biological firm theory
      Path_Network/      # Path network agent (has its own venv)
      KG_Multi_Agent/    # Knowledge graph multi-agent system
    tests/               # Centralized test suite
    scripts/             # Utility scripts (e.g., verify_links.py)
  knowledge_base/        # Theoretical foundations (cognitive, math, biology, philosophy, ontology)
  docs/                  # Guides, API docs, examples, research, RxInfer integration docs
  config.yaml            # Global configuration (inference params, visualization, logging)
```

### Key Design Patterns

- **ActiveInferenceModel** (`code/tools/src/models/active_inference/base.py`): Abstract base class all agents extend. Provides belief updating, policy inference, precision updating, and free energy calculation.
- **ActiveInferenceDispatcher** (`dispatcher.py`): Routes operations to specific inference implementations (variational, sampling, mean-field) and policy types (discrete, continuous, hierarchical).
- **Matrix system**: Agents are configured via YAML files specifying A (likelihood), B (transition), C (preference), D (prior), and E (action distribution) matrices with shape constraints.
- **Each Thing is self-contained**: Has its own config, README, AGENTS.md, and often its own output directory. Some (Path_Network) have their own virtual environments.

## Running Tests

```bash
# Run full test suite from repo root
python -m pytest code/tests/ -v

# Run a specific test file
python -m pytest code/tests/test_simple_pomdp.py -v

# Run with coverage
python -m pytest code/tests/ -v --cov=code/tools/src --cov-report=term-missing

# Run using the test runner script
python code/tests/run_tests.py

# Run benchmarks
python code/tests/run_benchmarks.py
```

Test configuration is in `code/tests/test_config.yaml`. Shared fixtures (sample matrices, belief vectors, output directories) are in `code/tests/conftest.py`. Matplotlib is set to `Agg` backend in conftest for non-interactive test runs.

## Dependencies

Python 3.8+. Core dependencies: `numpy`, `scipy`, `matplotlib`, `pyyaml`, `pytest`, `pytest-cov`. No global requirements.txt exists; some Things have their own (e.g., Path_Network).

## Conventions

### Documentation
- Every folder must have both `AGENTS.md` (technical documentation) and `README.md` (overview/navigation).
- Markdown files use YAML frontmatter with `semantic_relations` for Obsidian graph connections.
- Use `[[wiki-link]]` syntax for internal cross-references. The repo is an Obsidian vault.

### Code Style
- Python: snake_case for files and methods. Type hints required. Complete docstrings on public methods.
- **No test fixture methods in tests** - always use real data/computation.
- Remove non-semantic adjectives from names ("enhanced_", "new_", "improved_" are banned; "continuous_time", "hierarchical_" are fine).

### Configuration
- Agent configurations use YAML files specifying matrix shapes, constraints, initialization methods, and inference parameters.
- Global config at `config.yaml` defines default Active Inference parameters (precision, temporal_horizon, learning_rate, etc.).
