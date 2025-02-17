---
type: agent
id: "{{agent_id}}"
created: {{date}}
modified: {{date}}
tags: [agent, cognitive-model]
aliases: []
---

# Agent: {{agent_name}}

## Metadata
- **Type**: {{agent_type}}
- **Domain**: {{domain}}
- **Purpose**: {{purpose}}

## Properties
### Beliefs
- Initial beliefs and priors
- [[belief_links]]

### Goals
- Primary objectives
- Secondary objectives
- [[goal_links]]

### Actions
- Available actions
- Action constraints
- [[action_links]]

### Observations
- Observation space
- Sensory capabilities
- [[observation_links]]

## State
### Current State
- Active beliefs
- Current goals
- Recent actions
- Latest observations

### History
- State transitions
- Decision history
- Learning progress

## Relationships
### Dependencies
- Required resources
- External dependencies

### Interactions
- Agent interactions
- Environment interactions
- [[relationship_links]]

## Implementation
### Parameters
```yaml
learning_rate: 0.01
exploration_rate: 0.1
discount_factor: 0.95
```

### Active Inference Configuration
```yaml
precision: high
temporal_horizon: 5
inference_depth: 3
```

## Notes
- Implementation details
- Performance observations
- Optimization opportunities

## References
- Related research
- Documentation links
- External resources 