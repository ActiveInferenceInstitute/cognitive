# Package Documentation Guide

## Overview

This guide outlines the documentation standards and organization for the Cognitive Modeling package. It combines Python docstrings, markdown documentation, and Obsidian's knowledge management capabilities.

## Documentation Structure

### Directory Organization

```text

docs/

├── api/              # API reference documentation

├── concepts/         # Core theoretical concepts

├── examples/         # Usage examples and tutorials

├── guides/           # How-to guides and tutorials

└── tools/           # Tool-specific documentation

```

## Documentation Types

### 1. API Documentation

- Located in `docs/api/`

- Generated from docstrings

- Follows Google docstring format

Example:

```python

def update_beliefs(observation: np.ndarray, prior: np.ndarray) -> np.ndarray:

    """Updates agent beliefs based on new observations.

    Args:

        observation (np.ndarray): Current observation vector

        prior (np.ndarray): Prior belief state

    Returns:

        np.ndarray: Updated belief state

    See Also:

        [[belief_update_algorithm]]

        [[free_energy_computation]]

    """

    pass

```

### 2. Concept Documentation

- Located in `docs/concepts/`

- Theoretical foundations

- Mathematical formulations

- Links to implementations

Example: `concepts/free_energy.md`:

```markdown

# Free Energy Principle

## Mathematical Foundation

...

## Implementation

See [[free_energy_impl]] for code implementation.

```

### 3. Example Documentation

- Located in `docs/examples/`

- Practical usage examples

- Step-by-step tutorials

- Interactive notebooks

## Documentation Standards

### Python Docstrings

1. Use Google style docstrings

1. Include type hints

1. Link to relevant documentation

1. Provide usage examples

### Markdown Files

1. Clear headings hierarchy

1. Consistent formatting

1. Proper linking

1. Code examples

### YAML Frontmatter

```yaml

---

title: Package Documentation

category: guide

status: stable

related:

  - [[docstring_guide]]

  - [[api_reference]]

tags:

  - documentation

  - standards

---

```

## Integration Patterns

### Code-to-Doc Links

```python

# Implementation of algorithm described in [[algorithm_spec]]

class BeliefUpdater:

    pass

```

### Doc-to-Code Links

```markdown

See implementation in `src/models/belief_update.py`:

[[belief_update_impl#update_method]]

```

### Test Documentation

```python

def test_belief_update():

    """Test case described in [[belief_update_tests]]"""

    pass

```

## Maintenance

### Documentation Updates

1. Keep in sync with code

1. Update related documents

1. Maintain version consistency

1. Review broken links

### Version Control

1. Document version dependencies

1. Track breaking changes

1. Maintain changelog

1. Update migration guides

## Tools and Automation

### Documentation Generation

- Use sphinx for API docs

- Auto-generate from docstrings

- Maintain cross-references

- Update graphs

### Link Validation and Inventory

- Generate repository inventory and wikilink graph:

  ```bash

  python3 docs/repo_docs/repo_scripts/list_file_directory.py --root . --output docs/repo_docs/repo_scripts/output

  ```

- Fix broken/ambiguous/missing links:

  ```bash

  python3 docs/repo_docs/repo_scripts/fix_links.py --root . --output docs/repo_docs/repo_scripts/output

  ```

### Quality Checks

- Link validation

- Style consistency

- Coverage metrics

- Reference integrity

## Best Practices

### Writing Style

1. Clear and concise

1. Consistent terminology

1. Proper citations

1. Progressive disclosure

### Organization

1. Logical grouping

1. Clear navigation

1. Proper linking

1. Version management

### Code Examples

1. Runnable snippets

1. Clear context

1. Error handling

1. Best practices

## Related Guides

- [[api_documentation]]

- [[docstring_standards]]

- [[example_writing]]

- [[documentation_workflow]]

## References

- [[python_documentation]]

- [[sphinx_usage]]

- [[obsidian_integration]]

- [[documentation_tools]]

