---
title: Obsidian Usage Guide
type: guide
status: stable
created: 2024-02-12
tags:
  - obsidian
  - guide
  - overview
semantic_relations:
  - type: implements
    links: [[documentation_standards]]
  - type: relates
    links:
      - [[obsidian/folder_structure]]
      - [[obsidian/linking_patterns]]
---

# Obsidian Usage Guide

## Overview

This guide provides a comprehensive overview of using Obsidian for knowledge management in the cognitive modeling framework. For detailed information, see the specialized guides referenced throughout.

## Quick Start

### Installation
1. Download Obsidian from [obsidian.md](https://obsidian.md)
2. Install and open Obsidian
3. Open the cognitive modeling vault
4. Configure recommended settings

### Essential Features
- Markdown editing
- Wiki-style linking
- Graph visualization
- Plugin system

## Knowledge Organization

### Directory Structure
See [[obsidian/folder_structure|Folder Structure Guide]] for complete details.

```
cognitive/
├── knowledge_base/    # Core knowledge
├── docs/             # Documentation
└── templates/        # Note templates
```

### Content Types
1. Knowledge Base
   - Concepts
   - Theories
   - Implementations
   - Research notes

2. Documentation
   - Guides
   - API docs
   - Examples
   - Tutorials

3. Templates
   - Note templates
   - Code templates
   - Documentation templates

## Linking System

### Link Types
See [[obsidian/linking_patterns|Linking Patterns Guide]] for complete details.

```markdown
# Basic Links
[[filename]]
[[filename|alias]]

# Section Links
[[filename#section]]

# Block References
[[filename#^block-id]]
```

### Link Organization
- Group related links
- Use consistent patterns
- Maintain bidirectional links
- Follow naming conventions

## Templates

### Using Templates
1. Open command palette (Ctrl/Cmd + P)
2. Select "Templates: Insert template"
3. Choose appropriate template
4. Fill in template fields

### Template Types
See [[templates/template_guide|Template Guide]] for complete details.

1. Note Templates
   ```markdown
   ---
   title: ${title}
   type: ${type}
   created: ${date}
   ---
   
   # ${title}
   
   ## Overview
   
   ## Content
   ```

2. Documentation Templates
   ```markdown
   ---
   title: ${title}
   type: documentation
   status: draft
   ---
   
   # ${title}
   
   ## Purpose
   
   ## Usage
   ```

## Plugins

### Core Plugins
1. File Explorer
2. Search
3. Graph View
4. Backlinks
5. Outgoing Links
6. Tags View

### Community Plugins
See [[plugins/plugin_guide|Plugin Guide]] for complete details.

1. Essential Plugins
   - Dataview
   - Calendar
   - Templates
   - Mind Map

2. Development Plugins
   - Code Blocks
   - Mermaid
   - PlantUML
   - Math Preview

## Workflows

### Content Creation
1. Choose appropriate template
2. Fill in metadata
3. Add content
4. Create links
5. Add tags
6. Review and validate

### Content Organization
1. Use consistent structure
2. Follow naming conventions
3. Maintain clean hierarchy
4. Regular cleanup

### Link Management
1. Create meaningful links
2. Validate link targets
3. Add backlinks
4. Check link graph

## Best Practices

### File Management
- Use clear names
- Follow conventions
- Maintain organization
- Regular cleanup

### Content Structure
- Clear hierarchy
- Consistent formatting
- Proper metadata
- Complete documentation

### Link Hygiene
- Valid links
- Meaningful connections
- Bidirectional links
- Regular validation

## Advanced Features

### Graph View
- Node filtering
- Link filtering
- Custom colors
- Layout options

### Search
- Full-text search
- Regular expressions
- File properties
- Tag filtering

### Automation
- Templates
- Hotkeys
- Scripts
- Workflows

## Troubleshooting

### Common Issues
1. Broken links
2. Missing files
3. Plugin conflicts
4. Performance issues

### Solutions
1. Link validation
2. File verification
3. Plugin updates
4. Cache clearing

## Related Documentation
- [[obsidian/folder_structure|Folder Structure Guide]]
- [[obsidian/linking_patterns|Linking Patterns Guide]]
- [[templates/template_guide|Template Guide]]
- [[plugins/plugin_guide|Plugin Guide]] 