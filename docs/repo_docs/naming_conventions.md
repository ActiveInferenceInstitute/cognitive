# Naming Conventions Guide

---

title: Naming Conventions Guide

type: guide

status: stable

created: 2024-02-06

tags:

  - conventions

  - naming

  - standards

  - organization

semantic_relations:

  - type: implements

    links: [[documentation_standards]]

  - type: relates

    links:

      - [[knowledge_organization]]

      - [[ai_file_organization]]

---

## Overview

This guide establishes comprehensive naming conventions for all components in our cognitive modeling framework, ensuring consistency and clarity across documentation, code, and resources.

## File Naming

### 1. Documentation Files

```python

# @doc_file_patterns

doc_patterns = {

    "concepts": {

        "pattern": "{concept_name}.md",

        "example": "active_inference.md",

        "rules": {

            "lowercase": True,

            "separators": "_",

            "max_length": 50

        }

    },

    "guides": {

        "pattern": "{category}_{topic}.md",

        "example": "ai_documentation_style.md",

        "rules": {

            "category_prefix": True,

            "descriptive_name": True

        }

    },

    "templates": {

        "pattern": "{type}_template.md",

        "example": "concept_template.md",

        "rules": {

            "template_suffix": True,

            "type_prefix": True

        }

    }

}

```

### 2. Code Files

```python

# @code_file_patterns

code_patterns = {

    "implementation": {

        "pattern": "{module}_{component}.py",

        "example": "belief_updater.py",

        "rules": {

            "lowercase": True,

            "descriptive": True,

            "max_length": 40

        }

    },

    "tests": {

        "pattern": "test_{module}_{feature}.py",

        "example": "test_belief_updating.py",

        "rules": {

            "test_prefix": True,

            "match_implementation": True

        }

    },

    "utilities": {

        "pattern": "{category}_utils.py",

        "example": "matrix_utils.py",

        "rules": {

            "utils_suffix": True,

            "category_prefix": True

        }

    }

}

```

## Component Naming

### 1. Class Names

See [[code_organization]] for implementation context.

```python

# @class_patterns

class_patterns = {

    "agents": {

        "pattern": "{Type}Agent",

        "example": "ActiveInferenceAgent",

        "rules": {

            "PascalCase": True,

            "descriptive_prefix": True,

            "agent_suffix": True

        }

    },

    "models": {

        "pattern": "{Type}Model",

        "example": "BeliefModel",

        "rules": {

            "PascalCase": True,

            "model_suffix": True

        }

    },

    "components": {

        "pattern": "{Role}{Type}",

        "example": "BeliefUpdater",

        "rules": {

            "PascalCase": True,

            "role_prefix": True

        }

    }

}

```

### 2. Method Names

```python

# @method_patterns

method_patterns = {

    "actions": {

        "pattern": "{verb}_{object}",

        "example": "update_beliefs",

        "rules": {

            "snake_case": True,

            "verb_first": True

        }

    },

    "properties": {

        "pattern": "{object}_{attribute}",

        "example": "belief_state",

        "rules": {

            "snake_case": True,

            "noun_first": True

        }

    },

    "callbacks": {

        "pattern": "on_{event}",

        "example": "on_belief_update",

        "rules": {

            "on_prefix": True,

            "event_focus": True

        }

    }

}

```

## Documentation Structure

### 1. Section Headers

```python

# @section_patterns

section_patterns = {

    "main_sections": {

        "pattern": "## {Category}",

        "example": "## Overview",

        "rules": {

            "title_case": True,

            "max_words": 3

        }

    },

    "subsections": {

        "pattern": "### {Number}. {Title}",

        "example": "### 1. Implementation Details",

        "rules": {

            "numbered": True,

            "title_case": True

        }

    }

}

```

### 2. Link References

See [[linking_patterns]] for detailed linking guidelines.

```python

# @link_patterns

link_patterns = {

    "internal": {

        "pattern": "{category}/{name}",

        "example": "knowledge_base/cognitive/active_inference",

        "rules": {

            "category_prefix": True,

            "lowercase_path": True

        }

    },

    "aliased": {

        "pattern": "{display}",

        "example": "Active Inference",

        "rules": {

            "descriptive_alias": True,

            "consistent_display": True

        }

    }

}

```

## Metadata Conventions

### 1. YAML Frontmatter

```yaml

# @frontmatter_patterns

frontmatter:

  title:

    pattern: "{Type}: {Description}"

    example: "Guide: Active Inference Implementation"

    rules:

      - title_case: true

      - max_length: 60

  tags:

    pattern: ["{category}", "{subcategory}", "{specific}"]

    example: ["implementation", "active-inference", "agent"]

    rules:

      - lowercase: true

      - hyphen_separator: true

  semantic_relations:

    pattern:

      type: "{relationship_type}"

      links: ["{target}"]

    example:

      type: "implements"

      links: ["active_inference"]

```

### 2. Code Documentation

```python

# @docstring_patterns

docstring_patterns = {

    "class": {

        "pattern": """

        {Description}

        See {concept} for theoretical background.

        Attributes:

            {name} ({type}): {description}

        """,

        "rules": {

            "theoretical_link": True,

            "attribute_docs": True

        }

    },

    "method": {

        "pattern": """

        {Description}

        See {implementation} for details.

        Args:

            {name} ({type}): {description}

        Returns:

            {type}: {description}

        """,

        "rules": {

            "implementation_link": True,

            "complete_signature": True

        }

    }

}

```

## Validation Rules

### 1. Naming Validation

```python

# @validation_rules

validation_rules = {

    "files": {

        "pattern_compliance": 1.0,    # 100% compliance

        "length_limits": True,

        "character_set": "[a-z0-9_-]"

    },

    "components": {

        "case_compliance": 1.0,       # 100% compliance

        "prefix_suffix": True,

        "descriptive_names": True

    },

    "documentation": {

        "section_format": True,

        "link_format": True,

        "metadata_format": True

    }

}

```

### 2. Quality Checks

See [[quality_metrics]] for implementation.

```python

# @quality_metrics

naming_quality = {

    "consistency": {

        "pattern_adherence": 0.95,    # 95% pattern compliance

        "case_consistency": 1.0,      # 100% case consistency

        "separator_usage": 1.0        # 100% separator consistency

    },

    "clarity": {

        "descriptive_names": 0.9,     # 90% descriptive quality

        "length_compliance": 0.95,    # 95% length compliance

        "abbreviation_usage": 0.8     # 80% abbreviation compliance

    }

}

```

## Implementation Details

### 1. Name Processing


### 2. Name Validation


### 3. Name Analysis


## Implementation Examples

### 1. Name Pattern Matching


### 2. Name Formatting


### 3. Name Suggestion


## Integration Components

### 1. IDE Integration


### 2. Git Integration


### 3. Documentation Integration


## Best Practices

### 1. General Guidelines

- Use descriptive names

- Maintain consistent patterns

- Follow case conventions

- Limit name length

### 2. Documentation

- Follow [[documentation_standards]]

- Use [[ai_documentation_style]]

- Implement [[linking_patterns]]

- Validate with [[quality_metrics]]

### 3. Code Style

- Follow [[code_organization]]

- Use [[implementation_patterns]]

- Maintain [[code_quality]]

- Check [[style_guide]]

## Related Documentation

- [[documentation_standards]]

- [[knowledge_organization]]

- [[code_organization]]

- [[style_guide]]

## References

- [[implementation_patterns]]

- [[quality_metrics]]

- [[validation_framework]]

- [[code_quality]]
