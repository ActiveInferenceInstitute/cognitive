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

1. Install and open Obsidian

1. Open the cognitive modeling vault

1. Configure recommended settings

### Essential Features

- Markdown editing

- Wiki-style linking

- Graph visualization

- Plugin system

## Knowledge Organization

### Directory Structure

See [[obsidian/folder_structure|Folder Structure Guide]] for complete details.

```text

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

1. Documentation

   - Guides

   - API docs

   - Examples

   - Tutorials

1. Templates

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

1. Select "Templates: Insert template"

1. Choose appropriate template

1. Fill in template fields

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

1. Documentation Templates

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

1. Search

1. Graph View

1. Backlinks

1. Outgoing Links

1. Tags View

### Community Plugins

See [[plugins/plugin_guide|Plugin Guide]] for complete details.

1. Essential Plugins

   - Dataview

   - Calendar

   - Templates

   - Mind Map

1. Development Plugins

   - Code Blocks

   - Mermaid

   - PlantUML

   - Math Preview

## Workflows

### Content Creation

1. Choose appropriate template

1. Fill in metadata

1. Add content

1. Create links

1. Add tags

1. Review and validate

### Content Organization

1. Use consistent structure

1. Follow naming conventions

1. Maintain clean hierarchy

1. Regular cleanup

### Link Management

1. Create meaningful links

1. Validate link targets

1. Add backlinks

1. Check link graph

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

1. Missing files

1. Plugin conflicts

1. Performance issues

### Solutions

1. Link validation

1. File verification

1. Plugin updates

1. Cache clearing

## Related Documentation

- [[obsidian/folder_structure|Folder Structure Guide]]

- [[obsidian/linking_patterns|Linking Patterns Guide]]

- [[templates/template_guide|Template Guide]]

- [[plugins/plugin_guide|Plugin Guide]]

