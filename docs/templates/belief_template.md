---
type: belief
id: "{{belief_id}}"
created: {{date}}
modified: {{date}}
tags: [belief, cognitive-model]
aliases: []
---

# Belief: {{belief_name}}

## Metadata
- **Type**: {{belief_type}}
- **Domain**: {{domain}}
- **Confidence**: {{confidence_level}}

## Structure
### Prior Distribution
```yaml
distribution_type: {{distribution}}
parameters:
  mean: {{mean_value}}
  variance: {{variance}}
  bounds: [{{lower_bound}}, {{upper_bound}}]
```

### Dependencies
- Conditional dependencies
- Causal relationships
- [[belief_dependencies]]

## Content
### Semantic Description
- Belief description
- Context
- Implications

### Evidence
- Supporting observations
- Historical data
- [[evidence_links]]

## Dynamics
### Update Rules
- Learning rate
- Update conditions
- Temporal dynamics

### Constraints
- Logical constraints
- Domain constraints
- Value bounds

## Relationships
### Influences
- Affected beliefs
- Influenced actions
- [[influence_links]]

### Sources
- Information sources
- Reliability metrics
- [[source_links]]

## Implementation
### Parameters
```yaml
precision: {{precision}}
learning_rate: {{learning_rate}}
decay_factor: {{decay_factor}}
```

### Active Inference Integration
```yaml
free_energy_contribution: {{contribution}}
precision_weight: {{weight}}
```

## Notes
- Uncertainty considerations
- Update history
- Performance metrics

## References
- Related beliefs
- Documentation
- Research basis 