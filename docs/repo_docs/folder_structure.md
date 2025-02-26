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
│   ├── biology/        # Biological concepts
│   ├── BioFirm/       # BioFirm specific content
│   ├── agents/        # Agent-based models
│   ├── citations/     # Citation management
│   └── index.md       # Knowledge base index
├── docs/               # Documentation
│   ├── api/          # API documentation
│   ├── guides/       # User guides
│   ├── examples/     # Usage examples
│   ├── repo_docs/    # Repository documentation
│   ├── templates/    # Documentation templates
│   ├── implementation/ # Implementation docs
│   ├── development/  # Development guides
│   ├── tools/       # Documentation tools
│   ├── config/      # Configuration docs
│   ├── research/    # Research documentation
│   └── README.md    # Documentation overview
├── tests/             # Test suite
├── tools/             # Development tools
├── .obsidian/         # Obsidian configuration
├── Things/            # Project management
├── Output/            # Generated outputs
├── .benchmarks/       # Performance benchmarks
├── config.yaml        # Project configuration
├── project_structure.md # Project structure doc
└── LICENSE            # Project license

# Development Files
.coverage              # Test coverage data
.pytest_cache/         # Pytest cache
.git/                 # Git repository
.gitignore           # Git ignore patterns
```

### Documentation Structure
```
docs/
├── api/              # API documentation
├── guides/           # User guides
├── examples/         # Usage examples
├── repo_docs/        # Repository documentation
├── templates/        # Documentation templates
├── implementation/   # Implementation docs
├── development/     # Development guides
├── tools/          # Documentation tools
├── config/         # Configuration docs
├── research/       # Research documentation
└── README.md       # Documentation overview
```

### Knowledge Base Structure
```
knowledge_base/
├── cognitive/       # Cognitive science concepts
├── mathematics/     # Mathematical foundations
├── systems/        # Systems theory
├── biology/       # Biological concepts
├── BioFirm/      # BioFirm specific content
├── agents/       # Agent-based models
├── citations/    # Citation management
└── index.md      # Knowledge base index
```

### Advanced Organization

#### Version Control
```
.git/                 # Git repository
.gitignore           # Git ignore patterns
.gitattributes       # Git attributes
```

#### Development Configuration
```
.vscode/             # VS Code settings
.env                 # Environment variables
pyproject.toml       # Python project config
requirements.txt     # Python dependencies
```

#### CI/CD Structure
```
.github/
├── workflows/       # GitHub Actions
├── ISSUE_TEMPLATE/  # Issue templates
└── PULL_REQUEST_TEMPLATE/ # PR templates
```

## Naming Conventions

### File Types and Extensions
1. Documentation
   - `.md` - Markdown documentation
   - `.rst` - ReStructuredText docs
   - `.ipynb` - Jupyter notebooks

2. Source Code
   - `.py` - Python source
   - `.pyi` - Python type stubs
   - `.pyx` - Cython source

3. Configuration
   - `.yaml`/`.yml` - YAML config
   - `.json` - JSON data
   - `.toml` - TOML config

### Directory Naming
1. Knowledge Base
   - Use lowercase with underscores
   - Include category prefixes
   - Be descriptive and specific

2. Source Code
   - Use snake_case for Python
   - Group by functionality
   - Include version if needed

3. Documentation
   - Use descriptive names
   - Include content type
   - Follow hierarchy

## Organization Rules

### 1. Knowledge Base
- Group by domain
- Maintain clear hierarchy
- Use consistent naming
- Include metadata

### 2. Source Code
- Separate core and utils
- Group by functionality
- Follow language conventions
- Include tests

### 3. Documentation
- Organize by type
- Maintain versions
- Include examples
- Follow standards

## Metadata Standards

### YAML Frontmatter
```yaml
---
title: Document Title
type: content_type
status: draft/stable/archived
created: YYYY-MM-DD
modified: YYYY-MM-DD
version: X.Y.Z
tags:
  - primary_tag
  - secondary_tag
semantic_relations:
  - type: implements/relates/extends
    links:
      - [[related_doc]]
      - [[another_doc]]
---
```

### Required Fields
1. Core Metadata
   - title
   - type
   - status
   - created
   - tags

2. Optional Fields
   - modified
   - version
   - semantic_relations
   - aliases

## Directory Management

### 1. Creation Guidelines
- Follow structure
- Use templates
- Include README
- Add .gitkeep

### 2. Maintenance
- Regular cleanup
- Version control
- Archive old files
- Update documentation

### 3. Organization
- Clear hierarchy
- Logical grouping
- Consistent naming
- Proper metadata

## Automation

### Directory Scripts
```python
def organize_directory():
    """Organize directory structure.
    
    - Validates structure
    - Checks naming
    - Updates metadata
    - Generates docs
    """
    # Implementation
```

### Validation Tools
```python
def validate_structure():
    """Validate directory structure.
    
    - Check organization
    - Verify naming
    - Validate metadata
    - Test links
    """
    # Implementation
```

## Best Practices

### 1. Organization Principles
- **Modularity**
  - Keep related files together
  - Minimize dependencies between modules
  - Use clear boundaries between components
  - Follow single responsibility principle

- **Hierarchy**
  - Maximum 3-4 levels deep
  - Clear parent-child relationships
  - Logical grouping of related items
  - Consistent structure across similar components

- **Scalability**
  - Plan for growth
  - Easy to add new components
  - Maintainable structure
  - Version control friendly

### 2. Naming Conventions
- **Consistency**
  - Follow established patterns
  - Use descriptive names
  - Maintain naming hierarchy
  - Document exceptions

- **Clarity**
  - Self-documenting names
  - Avoid abbreviations
  - Use standard terminology
  - Include purpose in name

- **Versioning**
  - Include version in relevant files
  - Clear version progression
  - Archive old versions
  - Document version changes

### 3. Documentation Standards
- **Completeness**
  - All components documented
  - Clear purpose and usage
  - Dependencies listed
  - Examples included

- **Maintenance**
  - Regular updates
  - Version tracking
  - Change documentation
  - Review process

- **Accessibility**
  - Easy to find
  - Clear navigation
  - Search friendly
  - Cross-referenced

### 4. Development Workflow
- **Setup**
  - Clear installation steps
  - Environment configuration
  - Dependency management
  - Development tools

- **Testing**
  - Test organization
  - Coverage requirements
  - Test data management
  - CI/CD integration

- **Deployment**
  - Environment separation
  - Configuration management
  - Release process
  - Monitoring setup

### 5. Quality Control
- **Code Quality**
  - Linting standards
  - Formatting rules
  - Documentation requirements
  - Review process

- **Testing Standards**
  - Unit test organization
  - Integration test structure
  - Test data management
  - Coverage requirements

- **Security**
  - Access control
  - Sensitive data handling
  - Security scanning
  - Vulnerability management

### 6. Maintenance
- **Regular Tasks**
  - Dependency updates
  - Security patches
  - Performance optimization
  - Documentation updates

- **Cleanup**
  - Remove unused files
  - Archive old versions
  - Update dependencies
  - Maintain documentation

- **Monitoring**
  - Performance tracking
  - Error logging
  - Usage analytics
  - Health checks

### 7. Collaboration
- **Version Control**
  - Branch management
  - Commit guidelines
  - PR process
  - Review requirements

- **Documentation**
  - Clear contribution guidelines
  - Setup instructions
  - Style guides
  - Review process

- **Communication**
  - Issue tracking
  - Discussion forums
  - Documentation updates
  - Team notifications

## Related Documentation
- [[documentation_standards]]
- [[naming_conventions]]
- [[version_control]]
- [[metadata_standards]]

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

## Related Documentation
- [[obsidian_usage]]
- [[obsidian_linking]]
- [[documentation_standards]] 