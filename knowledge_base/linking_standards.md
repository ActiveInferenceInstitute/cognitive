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

[[filename]]

[[directory/filename]]

[[../relative/path/filename]]

```

### 2. Display Text (Avoid When Possible)

Obsidian supports display text, but for consistency, prefer the simple format:

```markdown

# Preferred

[[free_energy_principle]]

# Avoid unless necessary

[[free_energy_principle|Free Energy Principle]]

```

### 3. Section Links

Link to specific sections within files:

```markdown

[[filename#section-heading]]

[[../cognitive/active_inference#mathematical-framework]]

```

## Directory Structure Mapping

### Mathematics References

```markdown

[[../mathematics/free_energy_principle]]

[[../mathematics/bayesian_inference]]

[[../mathematics/variational_inference]]

[[../mathematics/information_theory]]

[[../mathematics/probability_theory]]

[[../mathematics/message_passing]]

[[../mathematics/factor_graphs]]

[[../mathematics/expected_free_energy]]

[[../mathematics/path_integral]]

[[../mathematics/stochastic_processes]]

[[../mathematics/dynamical_systems]]

[[../mathematics/optimization_theory]]

```

### Cognitive Science References

```markdown

[[../cognitive/active_inference]]

[[../cognitive/predictive_processing]]

[[../cognitive/free_energy_principle]]

[[../cognitive/generative_model]]

[[../cognitive/hierarchical_inference]]

[[../cognitive/attention_patterns]]

[[../cognitive/learning_mechanisms]]

[[../cognitive/decision_making]]

[[../cognitive/swarm_intelligence]]

[[../cognitive/collective_behavior]]

[[../cognitive/social_cognition]]

```

### Agent Architecture References

```markdown

[[../agents/GenericPOMDP/README]]

[[../agents/Continuous_Time/README]]

[[../agents/architectures_overview]]

[[../agents/index]]

```

### Systems Theory References

```markdown

[[../systems/systems_theory]]

[[../systems/complex_systems]]

[[../systems/emergence]]

[[../systems/network_theory]]

[[../systems/adaptive_systems]]

```

### Biology References

```markdown

[[../biology/evolutionary_dynamics]]

[[../biology/neural_systems]]

[[../biology/collective_behavior]]

[[../biology/adaptation]]

[[../biology/myrmecology]]

[[../biology/swarm_intelligence]]

```

### Philosophy References

```markdown

[[../philosophy/pragmatism]]

[[../philosophy/operationalism]]

[[../philosophy/peircean_semiotics]]

```

### Ontology References

```markdown

[[../ontology/cognitive_ontology]]

[[../ontology/hyperspatial/hyperspace_ontology]]

```

### Citation References

```markdown

[[../citations/friston_2017]]

[[../citations/parr_2019]]

[[../citations/shannon_1948]]

[[../citations/README]]

```

## Common Link Patterns

### Cross-Domain Connections

When linking between domains, use clear relative paths:

```markdown

# From mathematics to cognitive

[[../cognitive/active_inference]]

# From cognitive to mathematics  

[[../mathematics/free_energy_principle]]

# From agents to both

[[../cognitive/active_inference]]

[[../mathematics/expected_free_energy]]

```

### Bidirectional Linking

Ensure concepts link to each other:

```markdown

# In active_inference.md

Related concepts: [[../mathematics/free_energy_principle]], [[../mathematics/bayesian_inference]]

# In free_energy_principle.md  

Applications: [[../cognitive/active_inference]], [[../cognitive/predictive_processing]]

```

### Hierarchical Navigation

Link upward and downward in concept hierarchies:

```markdown

# Parent concept

[[../cognitive/learning_mechanisms]]

# Child concepts

[[../cognitive/reinforcement_learning]]

[[../cognitive/associative_learning]]

[[../cognitive/social_learning]]

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

    grep -o "\[\[[^]]*\]\]" "$file" | sort | uniq

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

- [[../cognitive/active_inference]]

- [[../cognitive/predictive_processing]]

- [[../cognitive/attention_patterns]]

- [[../agents/architectures_overview]]

```

### Cognitive Science Cross-References

```markdown

# In cognitive/active_inference.md

## Mathematical Foundations

Active inference builds on several mathematical concepts:

- [[../mathematics/free_energy_principle]]

- [[../mathematics/bayesian_inference]]

- [[../mathematics/expected_free_energy]]

- [[../mathematics/variational_inference]]

## Related Cognitive Concepts

- [[predictive_processing]]

- [[attention_patterns]]

- [[decision_making]]

```

### Agent Architecture Integration

```markdown

# In agents/architectures_overview.md

## Mathematical Foundations

- [[../mathematics/free_energy_principle]]

- [[../mathematics/active_inference_theory]]

- [[../mathematics/pomdp_framework]]

## Cognitive Principles

- [[../cognitive/active_inference]]

- [[../cognitive/hierarchical_inference]]

- [[../cognitive/learning_mechanisms]]

```

This linking standard ensures seamless navigation and discoverability throughout the knowledge base while maintaining Obsidian's powerful graph-based relationship visualization.

