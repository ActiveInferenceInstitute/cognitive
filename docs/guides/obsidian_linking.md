# Obsidian Linking Guide

## Overview
This guide explains how to effectively use Obsidian's [[wikilink]] syntax in our cognitive modeling documentation and knowledge base.

## Link Types

### Basic Wikilinks
- Standard format: `[[filename]]`
- With alias: `[[filename|display text]]`
- Section linking: `[[filename#section]]`
- Block references: `[[filename#^block-id]]`

## Best Practices

### File Naming Conventions
- Use lowercase with underscores: `[[cognitive_model]]`
- Be consistent and descriptive: `[[belief_update_algorithm]]`
- Avoid spaces and special characters
- Use singular form for concept pages

### Link Organization

#### Hierarchical Linking
```markdown
- [[parent_concept]]
  - [[child_concept_1]]
  - [[child_concept_2]]
```

#### Bidirectional Linking
Always consider reciprocal links in related documents:
```markdown
// In model_a.md
Related: [[model_b]]

// In model_b.md
Related: [[model_a]]
```

### Link Categories

#### Concept Links
- Link to fundamental concepts: `[[active_inference]]`
- Link to theoretical foundations: `[[free_energy_principle]]`

#### Implementation Links
- Link to code implementations: `[[belief_propagation_impl]]`
- Link to test files: `[[belief_tests]]`

#### Documentation Links
- Link to guides: `[[getting_started]]`
- Link to examples: `[[example_agent]]`

## YAML Frontmatter
Use frontmatter to enhance link relationships:

```yaml
---
title: Belief Update Algorithm
related:
  - [[free_energy]]
  - [[message_passing]]
tags:
  - algorithm
  - inference
---
```

## Link Visualization

### Graph View
- Use Obsidian's graph view to visualize relationships
- Color-code different types of notes
- Use filters to focus on specific relationships

### Local Graphs
- Enable local graphs for contextual relationships
- Use depth settings appropriately
- Consider link direction

## Common Patterns

### Knowledge Maps
```markdown
## Topic Map
- [[core_concept]]
  - [[sub_concept_1]] - Brief description
  - [[sub_concept_2]] - Brief description
```

### Implementation References
```markdown
## Implementation
- Algorithm: [[algorithm_name]]
- Tests: [[test_suite]]
- Examples: [[usage_example]]
```

### Version Links
```markdown
## Version History
- [[v1_implementation]]
- [[v2_implementation]] (current)
- [[v3_proposal]]
```

## Integration with Code

### Code Documentation Links
```python
# Link to documentation: [[matrix_operations]]
def update_matrix():
    pass
```

### Test References
```python
# Test cases documented in: [[matrix_test_cases]]
def test_matrix_update():
    pass
```

## Troubleshooting

### Common Issues
1. Broken links
2. Circular references
3. Missing backlinks

### Solutions
- Regular link validation
- Graph view analysis
- Consistent naming patterns

## Related Guides
- [[obsidian_usage]]
- [[documentation_style]]
- [[knowledge_organization]]
- [[graph_visualization]]

## References
- [[obsidian_official_docs]]
- [[knowledge_base_structure]]
- [[linking_best_practices]] 