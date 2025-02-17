---
title: Obsidian Folder Structure
type: guide
status: stable
created: 2024-02-12
tags:
  - obsidian
  - organization
  - structure
semantic_relations:
  - type: implements
    links: [[../documentation_standards]]
  - type: relates
    links:
      - [[obsidian_usage]]
      - [[obsidian_linking]]
---

# Obsidian Folder Structure

## Core Organization

### Root Structure
```
cognitive/                  # Root project directory
├── knowledge_base/        # Primary knowledge content
│   ├── cognitive/        # Cognitive science concepts
│   ├── mathematics/      # Mathematical foundations
│   ├── systems/         # Systems theory
│   └── research/        # Research notes
├── docs/                 # Documentation
│   ├── api/            # API documentation
│   ├── guides/         # User guides
│   └── examples/       # Usage examples
└── templates/           # Note templates
```

### Knowledge Base Organization

#### Cognitive Directory
```
cognitive/
├── agents/              # Agent models
├── perception/         # Perception systems
├── learning/          # Learning mechanisms
└── memory/            # Memory systems
```

#### Mathematics Directory
```
mathematics/
├── probability/        # Probability theory
├── statistics/        # Statistical methods
├── optimization/      # Optimization theory
└── inference/         # Inference methods
```

## File Naming Conventions

### General Rules
1. Use lowercase with underscores
2. Be descriptive but concise
3. Include category prefixes when helpful
4. Maintain consistent naming patterns

### Examples
```
# Good names
active_inference.md
bayesian_inference.md
free_energy_principle.md

# Bad names
ActiveInference.md
bayesian-inference.md
FEP.md
```

## Directory Principles

### Organization Rules
1. Group related content
2. Maintain clear hierarchy
3. Avoid deep nesting
4. Use meaningful names

### Special Directories
- `_attachments/` - For media files
- `_templates/` - For note templates
- `_archive/` - For archived content

## Metadata Structure

### YAML Frontmatter
```yaml
---
title: Document Title
type: note_type
status: draft/stable/archived
created: YYYY-MM-DD
modified: YYYY-MM-DD
tags:
  - primary_tag
  - secondary_tag
aliases:
  - alternative_name
---
```

### Required Fields
- title
- type
- status
- created
- tags

## Link Organization

### Internal Structure
- Use relative paths when possible
- Maintain bidirectional links
- Group related links together
- Use consistent link text

### Example Structure
```markdown
## Related Concepts
- [[../theory/concept_a|Concept A]]
- [[../implementation/concept_b|Implementation B]]

## See Also
- [[../guides/related_guide|Related Guide]]
- [[../examples/example|Example Usage]]
```

## Best Practices

### Directory Management
1. Regular cleanup of unused files
2. Consistent file organization
3. Clear naming patterns
4. Proper categorization

### File Organization
1. Group related files
2. Use appropriate subdirectories
3. Maintain clear hierarchy
4. Follow naming conventions

### Link Management
1. Regular link validation
2. Consistent link formatting
3. Proper link categorization
4. Bidirectional linking

## Automation Tools

### Directory Scripts
```python
def organize_files():
    """Organize files according to conventions."""
    # Implementation
```

### Link Checkers
```python
def validate_links():
    """Validate internal links."""
    # Implementation
```

## Related Documentation
- [[obsidian_usage]]
- [[obsidian_linking]]
- [[documentation_standards]] 