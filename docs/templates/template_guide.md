---
title: Template Guide
type: guide
status: stable
created: 2024-02-12
tags:
  - templates
  - documentation
  - standards
semantic_relations:
  - type: implements
    links: [[../guides/documentation_standards]]
  - type: relates
    links:
      - [[ai_concept_template]]
      - [[guide_template]]
---

# Template Guide

## Overview

This guide provides comprehensive documentation for using templates in the cognitive modeling framework. It covers template types, usage patterns, and customization.

## Template Types

### 1. Documentation Templates

#### Concept Template
```markdown
---
title: ${title}
type: concept
status: draft
created: ${date}
tags:
  - concept
  - ${category}
semantic_relations:
  - type: implements
    links: []
  - type: relates
    links: []
---

# ${title}

## Overview

Brief description of the concept.

## Theory

Theoretical foundation and background.

## Implementation

Implementation details and considerations.

## Usage

Usage examples and patterns.

## Related Concepts
- [[related_concept_1]]
- [[related_concept_2]]
```

#### Guide Template
```markdown
---
title: ${title}
type: guide
status: draft
created: ${date}
tags:
  - guide
  - ${category}
semantic_relations:
  - type: implements
    links: []
  - type: relates
    links: []
---

# ${title}

## Overview

Brief guide description.

## Usage

Step-by-step instructions.

## Examples

Usage examples.

## Best Practices

Recommended practices.

## Related Guides
- [[related_guide_1]]
- [[related_guide_2]]
```

### 2. Code Templates

#### Module Template
```python
"""${module_name}

This module provides functionality for ${purpose}.

Example:
    >>> from ${module} import ${class}
    >>> obj = ${class}()
    >>> obj.method()

Attributes:
    CONSTANT: Description
"""

from typing import List, Dict, Optional

class ${class_name}:
    """${class_description}
    
    Attributes:
        attr1: Description
        attr2: Description
    """
    
    def __init__(self):
        """Initialize the class."""
        pass
    
    def method(self) -> None:
        """Method description."""
        pass
```

#### Test Template
```python
"""Tests for ${module_name}."""

import pytest
from ${module} import ${class}

@pytest.fixture
def setup():
    """Test fixture setup."""
    return ${class}()

def test_method(setup):
    """Test method functionality."""
    result = setup.method()
    assert result == expected
```

## Template Usage

### Using Documentation Templates
1. Choose appropriate template
2. Copy template content
3. Replace variables
4. Fill in content
5. Update metadata
6. Add links

### Using Code Templates
1. Select template type
2. Copy template
3. Update module info
4. Implement functionality
5. Add documentation
6. Write tests

## Template Customization

### Variables
- ${title} - Document title
- ${date} - Creation date
- ${category} - Content category
- ${module_name} - Module name
- ${class_name} - Class name
- ${purpose} - Module purpose

### Sections
- Required sections
- Optional sections
- Custom sections
- Section order

### Metadata
- Title
- Type
- Status
- Tags
- Relations

## Best Practices

### Documentation
1. Use consistent formatting
2. Follow naming conventions
3. Complete all sections
4. Add relevant links
5. Include examples

### Code
1. Add type hints
2. Write docstrings
3. Include examples
4. Add tests
5. Follow style guide

## Template Management

### Organization
- Group by type
- Use clear names
- Maintain hierarchy
- Regular updates

### Maintenance
- Version control
- Regular review
- Update examples
- Fix issues

## Related Documentation
- [[ai_concept_template]]
- [[guide_template]]
- [[documentation_standards]] 