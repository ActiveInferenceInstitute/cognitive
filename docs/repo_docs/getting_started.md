---

title: Getting Started Guide

type: guide

status: stable

created: 2024-02-28

tags:

  - guide

  - setup

  - introduction

  - documentation

semantic_relations:

  - type: introduces

    links:

      - [[cognitive_modeling_concepts]]

      - [[installation_guide]]

      - [[basic_usage]]

---

# Getting Started with Cognitive Modeling

This guide will help you get started with our cognitive modeling framework, which implements active inference, message passing, and probabilistic programming concepts for building intelligent agents.

## Prerequisites

- Python 3.8 or higher

- Basic understanding of:

  - Probability theory

  - Bayesian inference

  - Python programming

## Installation

1. Clone the repository:

```bash

git clone https://github.com/yourusername/cognitive.git

cd cognitive

```

1. Create a virtual environment:

```bash

python -m venv venv

source venv/bin/activate  # On Windows: venv\Scripts\activate

```

1. Install dependencies:

```bash

# If you use a specific subproject, install its requirements (examples):

# Ant Colony simulator

python3 -m pip install -r Things/Ant_Colony/requirements.txt

# Generic Thing (if needed for local experiments)

python3 -m pip install -r Things/Generic_Thing/requirements.txt

```

## Quick Start

### 1. Basic Active Inference Agent

Here's a minimal example of creating an active inference agent:

```python

from cognitive.agents import ActiveInferenceAgent

from cognitive.models import GenerativeModel

from cognitive.inference import MessagePassing

# Create a simple generative model

model = GenerativeModel()

model.add_state("position", domain=[-1, 1])

model.add_observation("sensor", depends_on=["position"])

model.add_action("move", affects=["position"])

# Initialize the agent

agent = ActiveInferenceAgent(model)

# Run inference

agent.infer()

# Take action

action = agent.act()

```

### 2. Running Examples

Explore our example implementations:

```bash

# Run basic active inference example

python examples/active_inference_basic.py

# Run POMDP agent example

python examples/pomdp_agent.py

```

## Core Concepts

### 1. Active Inference

Active inference is a framework that unifies perception, learning, and action under the free energy principle. Agents:

- Maintain beliefs about their environment

- Update beliefs through sensory observations

- Take actions to minimize expected free energy

### 2. Message Passing

Our implementation uses message passing for efficient inference:

- Factor graphs represent probabilistic models

- Messages propagate beliefs between nodes

- Variational inference approximates posterior distributions

### 3. Agent Architecture

Agents are structured with:

- Generative models of the environment

- Inference mechanisms for belief updating

- Policy selection for action choice

- Learning mechanisms for model adaptation

## Next Steps

1. Read the [[docs/concepts/overview|Core Concepts]] guide

1. Try the [[docs/tutorials/index|Tutorials]]

1. Explore [[docs/examples/index|Example Implementations]]

1. Review the [[docs/api/reference|API Reference]]

## Running Tests (TDD)

Use Python’s pytest to exercise the repository tests and follow TDD:

```bash

# Run the top-level test suite

python3 -m pytest -q tests

# Run Generic Thing test suite (rich visualization coverage)

python3 Things/Generic_Thing/tests/run_test_suite.py

# Run Continuous Generic visual tests

python3 -m pytest -q Things/Continuous_Generic/tests

```

## Validating and Fixing Documentation Links

Generate a repository-wide file/index graph and validate Obsidian-style wikilinks:

```bash

# Produce inventory (JSON/CSV/TXT) for the whole repo

python3 docs/repo_docs/repo_scripts/list_file_directory.py --root . --output docs/repo_docs/repo_scripts/output

# Analyze and fix broken/ambiguous/missing backlinks across knowledge_base/ and docs/

python3 docs/repo_docs/repo_scripts/fix_links.py --root . --output docs/repo_docs/repo_scripts/output

```

Artifacts are written under `docs/repo_docs/repo_scripts/output`. Review the generated `link_analysis_report.md` and commit changes after inspection.

## Common Patterns

### 1. Defining Models

```python

from cognitive.models import GenerativeModel

def create_model():

    model = GenerativeModel()

    # Add state variables

    model.add_state("x", domain=continuous)

    model.add_state("v", domain=continuous)

    # Add observations

    model.add_observation("sensor", distribution="gaussian")

    # Add actions

    model.add_action("accelerate", affects=["v"])

    return model

```

### 2. Implementing Inference

```python

from cognitive.inference import VariationalInference

def run_inference(agent, observation):

    # Update beliefs using variational inference

    agent.update_beliefs(

        observation=observation,

        method=VariationalInference(),

        iterations=10

    )

```

### 3. Policy Selection

```python

def select_action(agent):

    # Compute expected free energy for each policy

    policies = agent.get_policies()

    # Select policy minimizing expected free energy

    best_policy = agent.select_policy(policies)

    # Get next action from policy

    action = best_policy.next_action()

    return action

```

## Troubleshooting

### Common Issues

1. **Convergence Problems**

   - Increase number of inference iterations

   - Adjust learning rates

   - Check model specification

1. **Performance Issues**

   - Use efficient message passing implementations

   - Optimize model structure

   - Consider parallel processing

1. **Memory Usage**

   - Implement streaming inference

   - Manage belief history

   - Use sparse representations

## Getting Help

- Check the [[docs/troubleshooting|Troubleshooting Guide]]

- Join our [Community Forum](https://forum.example.com)

- Submit issues on [GitHub](https://github.com/yourusername/cognitive/issues)

## Contributing

We welcome contributions! See our [[development/contribution_guide|Contribution Guide]] for:

- Code style guidelines

- Pull request process

- Testing requirements

- Documentation standards

## License

Repository: [ActiveInferenceInstitute/cognitive](https://github.com/ActiveInferenceInstitute/cognitive)

- Code and executable examples: MIT License (see `LICENSE`)

- Documentation and knowledge base content: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

