# Document Linking Template

---
title: Document Linking Template
type: template
status: stable
created: 2024-02-06
tags:
  - template
  - linking
  - documentation
semantic_relations:
  - type: implements
    links: [[linking_completeness]]
  - type: extends
    links: [[ai_documentation_style]]
---

## Required Link Sections

### 1. Metadata Links
```yaml
---
title: Document Title
type: [document_type]
status: [status]
semantic_relations:
  # Core relationships (Required)
  - type: implements
    links: [[base_concept]]
  - type: extends
    links: [[parent_concept]]
  
  # Dependencies (Required)
  - type: requires
    links: 
      - [[dependency1]]
      - [[dependency2]]
  
  # Related Content (Optional)
  - type: related
    links: [[related_content]]
---
```

### 2. Concept Definition Links
```markdown
#BEGIN_CONCEPT_LINKS
- Parent Concept: [[parent_concept]]
- Core Implementation: [[implementation]]
- Validation: [[validation_spec]]
- Examples: [[usage_examples]]
#END_CONCEPT_LINKS
```

### 3. Implementation Links
```markdown
#BEGIN_IMPLEMENTATION_LINKS
- Interface: [[api_spec]]
- Tests: [[test_suite]]
- Examples: [[implementation_examples]]
- Validation: [[implementation_validation]]
#END_IMPLEMENTATION_LINKS
```

### 4. Knowledge Graph Links
```markdown
#BEGIN_KNOWLEDGE_LINKS
- Ontology: [[domain_ontology]]
- Classification: [[concept_classification]]
- Relationships: [[concept_relationships]]
#END_KNOWLEDGE_LINKS
```

## Link Categories

### 1. Hierarchical Links
```markdown
## Hierarchy
- Parent: [[parent_concept]]
- Children:
  - [[child_concept_1]]
  - [[child_concept_2]]
- Siblings:
  - [[sibling_1]]
  - [[sibling_2]]
```

### 2. Implementation Links
```markdown
## Implementation
- Core: [[core_implementation]]
- Extensions:
  - [[extension_1]]
  - [[extension_2]]
- Tests:
  - [[test_suite]]
  - [[benchmarks]]
```

### 3. Documentation Links
```markdown
## Documentation
- Guide: [[user_guide]]
- API: [[api_reference]]
- Examples: [[example_collection]]
- Tutorials: [[tutorial_series]]
```

### 4. Research Links
```markdown
## Research
- Papers: [[research_papers]]
- Experiments: [[experiment_results]]
- Analysis: [[data_analysis]]
- Citations: [[citations]]
```

## Link Annotations

### 1. Relationship Annotations
```markdown
- [[concept]] {type: prerequisite, confidence: 0.9}
- [[implementation]] {type: implements, version: "1.0"}
- [[test]] {type: validates, coverage: 0.95}
```

### 2. Context Annotations
```markdown
- [[concept]] {context: "theoretical_foundation"}
- [[example]] {context: "practical_application"}
- [[test]] {context: "validation"}
```

### 3. Status Annotations
```markdown
- [[feature]] {status: "stable", since: "1.0"}
- [[api]] {status: "deprecated", replaced_by: "[[new_api]]"}
- [[concept]] {status: "draft", review_required: true}
```

## Validation Blocks

### 1. Link Validation
```python
# @link_validation
{
    "required_links": {
        "concept": ["parent", "implementation", "validation"],
        "implementation": ["interface", "tests", "examples"],
        "documentation": ["guide", "api", "examples"]
    }
}
```

### 2. Relationship Validation
```python
# @relationship_validation
{
    "bidirectional": ["implements", "extends", "requires"],
    "hierarchical": ["parent", "child", "sibling"],
    "temporal": ["precedes", "follows", "replaces"]
}
```

## Integration Examples

### 1. Concept Documentation
```markdown
# Concept: Active Inference

## Core Links
- Theory: [[free_energy_principle]]
- Implementation: [[active_inference_impl]]
- Validation: [[active_inference_tests]]

## Related Concepts
- [[predictive_coding]]
- [[belief_updating]]
- [[action_selection]]

## Applications
- [[robot_control]]
- [[decision_making]]
- [[learning_systems]]
```

### 2. Implementation Documentation
```markdown
# Implementation: Belief Updating

## Specification Links
- Design: [[belief_update_design]]
- Interface: [[belief_update_api]]
- Tests: [[belief_update_tests]]

## Dependencies
- [[matrix_operations]]
- [[probability_utils]]
- [[optimization_methods]]

## Examples
- [[basic_belief_update]]
- [[advanced_scenarios]]
```

### 3. Documentation Integration
```markdown
# Guide: System Overview

## Component Links
- [[architecture_overview]]
- [[component_interactions]]
- [[deployment_guide]]

## Implementation Links
- [[core_components]]
- [[extension_points]]
- [[integration_patterns]]

## Reference Links
- [[api_documentation]]
- [[example_collection]]
- [[troubleshooting_guide]]
```

## Best Practices

### 1. Link Organization
- Group related links logically
- Maintain consistent structure
- Use appropriate annotations
- Include validation blocks

### 2. Link Maintenance
- Regular link validation
- Update bidirectional links
- Remove obsolete links
- Add new relationships

### 3. Link Quality
- Clear relationship types
- Appropriate context
- Meaningful descriptions
- Proper categorization

## Related Templates
- [[concept_template]]
- [[implementation_template]]
- [[documentation_template]]
- [[validation_template]]

## References
- [[linking_completeness]]
- [[documentation_standards]]
- [[validation_framework]]
- [[knowledge_organization]] 