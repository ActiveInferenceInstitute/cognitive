---
title: Obsidian Linking Patterns
type: guide
status: stable
created: 2024-02-12
tags:
  - obsidian
  - linking
  - organization
semantic_relations:
  - type: implements
    links: [[../documentation_standards]]
  - type: relates
    links:
      - [[obsidian_usage]]
      - [[folder_structure]]
---

# Obsidian Linking Patterns

## Link Types Overview

### Basic Links
1. Direct Links
   ```markdown
   [[filename]]
   ```

2. Aliased Links
   ```markdown
   [[filename|Display Text]]
   ```

3. Section Links
   ```markdown
   [[filename#Section Name]]
   ```

4. Block Links
   ```markdown
   [[filename#^block-id]]
   ```

### Advanced Links

1. Relative Path Links
   ```markdown
   [[../parent_folder/filename]]
   [[./current_folder/filename]]
   ```

2. Multi-section Links
   ```markdown
   [[filename#section1#subsection]]
   ```

3. Embedded Links
   ```markdown
   ![[filename]]
   ![[filename#section]]
   ```

## Link Organization

### Semantic Grouping
```markdown
## Theory
- [[active_inference|Active Inference Theory]]
- [[free_energy|Free Energy Principle]]

## Implementation
- [[active_inference_impl|Implementation Details]]
- [[free_energy_calc|Calculations]]
```

### Hierarchical Organization
```markdown
## Core Concepts
- [[parent_concept]]
  - [[child_concept_1]]
  - [[child_concept_2]]
    - [[grandchild_1]]
    - [[grandchild_2]]
```

## Link Patterns

### Knowledge Base Links
```markdown
## Concept Definition
- Base: [[concept_name]]
- Theory: [[concept_theory]]
- Implementation: [[concept_implementation]]
- Examples: [[concept_examples]]
```

### Documentation Links
```markdown
## Documentation Structure
- Guide: [[user_guide]]
- API: [[api_documentation]]
- Examples: [[usage_examples]]
- Tests: [[test_documentation]]
```

### Code Links
```markdown
## Code References
- Source: [[source_file]]
- Tests: [[test_file]]
- Examples: [[example_file]]
- Benchmarks: [[benchmark_file]]
```

## Link Management

### Frontmatter Links
```yaml
---
related_concepts:
  - [[concept_a]]
  - [[concept_b]]
dependencies:
  - [[dependency_1]]
  - [[dependency_2]]
implementations:
  - [[impl_1]]
  - [[impl_2]]
---
```

### Link Categories
```markdown
## Theoretical Links
- [[theory_a]]
- [[theory_b]]

## Implementation Links
- [[impl_x]]
- [[impl_y]]

## Test Links
- [[test_1]]
- [[test_2]]
```

## Best Practices

### Link Naming
1. Use descriptive names
2. Maintain consistency
3. Follow naming conventions
4. Use appropriate aliases

### Link Organization
1. Group related links
2. Use clear hierarchy
3. Maintain bidirectional links
4. Document relationships

### Link Maintenance
1. Regular validation
2. Update broken links
3. Clean unused links
4. Check consistency

## Automation

### Link Validation
```python
def validate_links(content: str) -> List[str]:
    """Validate all links in content."""
    pattern = r'\[\[(.*?)\]\]'
    links = re.findall(pattern, content)
    return validate_link_targets(links)
```

### Link Generation
```python
def generate_backlinks(source: str, target: str) -> str:
    """Generate bidirectional links."""
    return f"""
    // In {source}:
    [[{target}]]
    
    // In {target}:
    [[{source}]]
    """
```

## Common Patterns

### Theory Documentation
```markdown
## Theoretical Foundation
- Base Theory: [[base_theory]]
- Extensions: [[theory_extension]]
- Applications: [[theory_application]]
```

### Implementation Documentation
```markdown
## Implementation Details
- Core: [[core_implementation]]
- Modules: [[module_documentation]]
- Tests: [[test_documentation]]
```

### Research Documentation
```markdown
## Research Notes
- Papers: [[research_papers]]
- Experiments: [[experiment_notes]]
- Results: [[research_results]]
```

## Link Visualization

### Graph View
- Node types
- Link types
- Clustering
- Filtering

### Local Graphs
- Depth settings
- Node filtering
- Link filtering
- Layout options

## Troubleshooting

### Common Issues
1. Broken links
2. Circular references
3. Ambiguous links
4. Missing backlinks

### Solutions
1. Regular validation
2. Clear naming
3. Proper organization
4. Consistent patterns

## Related Documentation
- [[obsidian_usage]]
- [[folder_structure]]
- [[documentation_standards]] 