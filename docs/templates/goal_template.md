---

type: goal

id: "{{goal_id}}"

created: "{{date}}"

modified: "{{date}}"

tags: [goal, cognitive-model, motivation]

aliases: ["{{goal_name}}", "{{goal_alias}}"]

related_goals: ["{{related_goal_1}}", "{{related_goal_2}}"]

---

# Goal: {{goal_name}}

## Metadata

- **Type**: {{goal_type}}

- **Domain**: {{domain}}

- **Priority**: {{priority}}

- **Status**: {{status}}

- **Version**: {{version}}

- **Timeframe**: {{timeframe}}

## Overview

{{goal_description}}

## Structure

### Goal Hierarchy

```mermaid

graph TD

    A[Goal: {{goal_name}}] --> B[Sub-Goal 1]

    A --> C[Sub-Goal 2]

    D[Parent Goal] --> A

    A --> E[Action 1]

    A --> F[Action 2]

```

### Parameters

```yaml

parameters:

  priority: {{priority}}

  utility: {{utility}}

  deadline: {{deadline}}

  effort_estimate: {{effort_estimate}}

  success_criteria:

    - {{criterion_1}}

    - {{criterion_2}}

```

### Success Criteria

- Primary criteria

- Secondary criteria

- Measurement methods

- Criterion 1

- Criterion 2

## Planning

### Action Plan

```mermaid

gantt

    title Goal Achievement Plan

    dateFormat  YYYY-MM-DD

    Sub-Goal 1 :sg1, {{start_date_1}}, {{duration_1}}

    Action 1.1 :a11, after sg1, {{duration_1_1}}

    Action 1.2 :a12, after a11, {{duration_1_2}}

    Sub-Goal 2 :sg2, {{start_date_2}}, {{duration_2}}

    Action 2.1 :a21, after sg2, {{duration_2_1}}

    Action 2.2 :a22, after a21, {{duration_2_2}}

```

### Implementation

See `docs/examples/README.md` for runnable examples.


### Planning Strategies

- Hierarchical planning

- Temporal planning

- Resource allocation

- Contingency planning

## Motivation

### Utility Function

```mermaid

graph LR

    A[Goal State] --> B[Utility Calculation]

    C[Current State] --> B

    D[Effort Required] --> B

    E[Time Constraints] --> B

    B --> F[Goal Utility]

```

### Priority Dynamics

- Initial priority

- Dynamic adjustment

- Context sensitivity

- Competing goals resolution

### Motivation Factors

- Intrinsic motivation

- Extrinsic motivation

- Temporal discounting

- Motivation Factor 1

- Motivation Factor 2

## Decision Process

### Goal Selection

```mermaid

sequenceDiagram

    participant B as Beliefs

    participant G as Goals

    participant U as Utility Calculation

    participant S as Goal Selection

    participant A as Action Selection

    B->>U: Current state assessment

    G->>U: Goal parameters

    U->>S: Goal utilities

    S->>A: Selected goal

    A->>S: Action outcomes

```

### Policy Integration

- Goal-directed policies

- Habitual policies

- Exploration strategies

- Policy 1

- Policy 2

### Conflict Resolution

- Goal conflicts

- Resource conflicts

- Temporal conflicts

- Priority-based resolution

## Relationships

### Dependencies

- Required resources

- Prerequisite goals

- External dependencies

- Dependency 1

- Dependency 2

### Interactions

- Goal interactions

- Belief interactions

- Action interactions

- Interaction 1

- Interaction 2

## Evaluation

### Performance Metrics

- Achievement rate

- Efficiency

- Time to completion

- Resource utilization

### Validation Methods

- Progress tracking

- Success verification

- Utility assessment

## Implementation Details

### Parameters

```yaml

expected_utility: {{expected_utility}}

achievement_probability: {{achievement_probability}}

effort_cost: {{effort_cost}}

time_sensitivity: {{time_sensitivity}}

failure_penalty: {{failure_penalty}}

```

### Active Inference Configuration

```yaml

expected_free_energy: {{expected_free_energy}}

precision: {{precision}}

temporal_horizon: {{temporal_horizon}}

exploration_factor: {{exploration_factor}}

```

## Notes

- Implementation details

- Performance observations

- Optimization opportunities

- Known limitations

## References

- Related research

- Documentation links

- External resources

- Reference 1

- Reference 2

## Related Goals

- Related Goal 1

- Related Goal 2

