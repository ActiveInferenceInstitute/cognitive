---
title: Obsidian Linking Standards
type: guide
status: active
created: 2024-02-07
tags:
- linking
- standards
- obsidian
- navigation
---


# Obsidian Linking Standards

## Overview

This document establishes consistent linking standards for the knowledge base to ensure proper Obsidian wiki-style navigation and connection between concepts.

## Link Format Standards

### 1. Basic Wiki Links

Use double brackets for all internal references:

```markdown

filename

directory/filename

relative/path/filename

```

### 2. Display Text (Avoid When Possible)

Obsidian supports display text, but for consistency, prefer the simple format:

```markdown

# Preferred

free_energy_principle

# Avoid unless necessary

Free Energy Principle

```

### 3. Section Links

Link to specific sections within files:

```markdown

filename#section-heading

knowledge_base/cognitive/active_inference#mathematical-framework

```

## Directory Structure Mapping

### Mathematics References

```markdown

knowledge_base/mathematics/free_energy_principle

knowledge_base/cognitive/bayesian_inference

knowledge_base/mathematics/variational_inference

knowledge_base/mathematics/information_theory

knowledge_base/mathematics/probability_theory

knowledge_base/mathematics/message_passing

knowledge_base/mathematics/factor_graphs

knowledge_base/mathematics/expected_free_energy

knowledge_base/mathematics/path_integral

knowledge_base/mathematics/stochastic_processes

knowledge_base/mathematics/dynamical_systems

knowledge_base/mathematics/optimization_theory

```

### Cognitive Science References

```markdown

knowledge_base/cognitive/active_inference

knowledge_base/cognitive/predictive_processing

knowledge_base/cognitive/free_energy_principle

knowledge_base/cognitive/generative_model

knowledge_base/cognitive/hierarchical_inference

knowledge_base/cognitive/attention_patterns

knowledge_base/cognitive/learning_mechanisms

knowledge_base/cognitive/decision_making

knowledge_base/cognitive/swarm_intelligence

knowledge_base/cognitive/collective_behavior

knowledge_base/cognitive/social_cognition

```

### Agent Architecture References

```markdown

knowledge_base/agents/GenericPOMDP/README

knowledge_base/agents/Continuous_Time/README

knowledge_base/agents/architectures_overview

knowledge_base/agents/index

```

### Systems Theory References

```markdown

knowledge_base/systems/systems_theory

knowledge_base/systems/complex_systems

knowledge_base/systems/emergence

knowledge_base/systems/network_theory

knowledge_base/systems/adaptive_systems

```

### Biology References

```markdown

knowledge_base/biology/evolutionary_dynamics

knowledge_base/free_energy_principle/biology/neural_systems

knowledge_base/cognitive/collective_behavior

docs/research/complex_systems/adaptation

knowledge_base/biology/myrmecology

knowledge_base/cognitive/swarm_intelligence

```

### Philosophy References

```markdown

knowledge_base/philosophy/pragmatism

knowledge_base/philosophy/operationalism

knowledge_base/philosophy/peircean_semiotics

```

### Ontology References

```markdown

knowledge_base/ontology/cognitive_ontology

knowledge_base/ontology/hyperspatial/hyperspace_ontology

```

## Common Link Patterns

### Cross-Domain Connections

When linking between domains, use clear relative paths:

```markdown

# From mathematics to cognitive

knowledge_base/cognitive/active_inference

# From cognitive to mathematics  

knowledge_base/mathematics/free_energy_principle

# From agents to both

knowledge_base/cognitive/active_inference

knowledge_base/mathematics/expected_free_energy

```

### Bidirectional Linking

Ensure concepts link to each other:

```markdown

# In active_inference.md

Related concepts: knowledge_base/mathematics/free_energy_principle, knowledge_base/cognitive/bayesian_inference

# In free_energy_principle.md  

Applications: knowledge_base/cognitive/active_inference, knowledge_base/cognitive/predictive_processing

```

### Hierarchical Navigation

Link upward and downward in concept hierarchies:

```markdown

# Parent concept

knowledge_base/cognitive/learning_mechanisms

# Child concepts

knowledge_base/cognitive/reinforcement_learning

knowledge_base/cognitive/associative_learning

knowledge_base/cognitive/social_learning

```

## Link Validation Checklist

### File Existence

- [ ] All linked files exist in the knowledge base

- [ ] Relative paths are correct

- [ ] No broken or dangling links

### Consistency

- [ ] Use consistent naming conventions

- [ ] Avoid redundant display text

- [ ] Maintain parallel link structures

### Completeness

- [ ] Key concepts link to related concepts

- [ ] Cross-domain connections are established

- [ ] Bidirectional linking is implemented

### Accessibility

- [ ] Links provide clear navigation paths

- [ ] Concept discovery is supported

- [ ] Multiple pathways to content exist

## Implementation Guidelines

### When Adding New Content

1. **Identify Related Concepts**: List all concepts the new content relates to

1. **Create Outbound Links**: Link from new content to existing concepts

1. **Update Inbound Links**: Add links to new content from related files

1. **Verify Path Accuracy**: Test all links in Obsidian

1. **Update Index Files**: Add new content to relevant index files

### When Reorganizing Content

1. **Document Current Links**: Record all existing links to/from moved content

1. **Update Relative Paths**: Adjust paths based on new location

1. **Test All Connections**: Verify links work in new structure

1. **Update References**: Modify any hardcoded paths in documentation

1. **Create Redirects**: Consider alias links for moved content

## Automated Validation

### Link Checking Script

```bash

# Example validation command

find knowledge_base -name "*.md" -exec grep -l "\[\[.*\]\]" {} \; | \

while read file; do

    echo "Checking links in: $file"

    grep -o "\[\^*\]\]" "$file" | sort | uniq

done

```

### Common Issues to Detect

- Broken relative paths

- Non-existent target files

- Inconsistent naming

- Missing reciprocal links

- Orphaned content

## Best Practices Summary

1. **Use Simple Format**: Prefer `[[filename]]` over `[[filename|Display Text]]`

1. **Consistent Paths**: Use relative paths consistently

1. **Bidirectional Links**: Ensure concepts link to each other

1. **Regular Validation**: Check links when adding/moving content

1. **Clear Navigation**: Provide multiple pathways to content

1. **Index Maintenance**: Keep index files updated with new links

## Examples of Proper Linking

### Mathematics to Cognitive Science

```markdown

# In mathematics/free_energy_principle.md

## Applications

The free energy principle has important applications in:

- knowledge_base/cognitive/active_inference

- knowledge_base/cognitive/predictive_processing

- knowledge_base/cognitive/attention_patterns

- knowledge_base/agents/architectures_overview

```

### Cognitive Science Cross-References

```markdown

# In cognitive/active_inference.md

## Mathematical Foundations

Active inference builds on several mathematical concepts:

- knowledge_base/mathematics/free_energy_principle

- knowledge_base/cognitive/bayesian_inference

- knowledge_base/mathematics/expected_free_energy

- knowledge_base/mathematics/variational_inference

## Related Cognitive Concepts

- predictive_processing

- attention_patterns

- decision_making

```

### Agent Architecture Integration

```markdown

# In agents/architectures_overview.md

## Mathematical Foundations

- knowledge_base/mathematics/free_energy_principle

- knowledge_base/mathematics/active_inference_theory

- knowledge_base/mathematics/pomdp_framework

## Cognitive Principles

- knowledge_base/cognitive/active_inference

- knowledge_base/cognitive/hierarchical_inference

- knowledge_base/cognitive/learning_mechanisms

```

This linking standard ensures seamless navigation and discoverability throughout the knowledge base while maintaining Obsidian's powerful graph-based relationship visualization.

