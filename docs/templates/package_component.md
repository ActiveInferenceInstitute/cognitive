---
title: Component Documentation Template
type: template
status: stable
created: 2024-02-06
updated: 2026-10-10
tags:
  - template
  - component
  - package
semantic_relations:
  - type: template_for
    links:
      - "[[docs/api/api_reference|API reference]]"
  - type: relates
    links:
      - "[[docs/repo_docs/package_documentation|Package documentation]]"
---

# Component Documentation Template

Copy this page to document one component of the `cognitive` package (source in
`code/tools/src/`). Replace the text in square brackets, delete sections that do
not apply, and make sure every symbol and path you name exists. The public
exports are listed in [[docs/api/api_reference|the API reference]].

Suggested frontmatter for the copied page:

```yaml
---
title: Component Name
type: package_component
status: draft
created: YYYY-MM-DD
updated: YYYY-MM-DD
tags:
  - component
semantic_relations:
  - type: relates
    links:
      - "[[path/to/related_component_1]]"
      - "[[path/to/related_component_2]]"
---
```

## Overview

[Purpose of the component and its role in the package.]

## Installation

```bash
# From the repository root: installs the package and its console scripts
pip install -e ".[dev]"
```

## Core features

- [Feature 1]
- [Feature 2]

## Usage

```python
# Import from the top-level package where the symbol is exported
from cognitive import ActiveInferenceDispatcher
```

Show a complete, runnable example here and run it with the repository's Python
before publishing the page.

## API reference

### Classes

- `ClassName`: [brief description and source file]

### Functions

- `function_name`: [brief description and source file]

## Configuration

[Configuration keys the component reads, for example from `config.yaml`, and
their defaults.]

## Integration

- [How this component is used by other components]
- [Components this one depends on]

## Testing

Tests live in `code/tests/`. Run one file with
`python -m pytest code/tests/test_<name>.py -v`, or the whole suite with
`python -m pytest code/tests/ -v`.

## Performance

[Measured results only. `cognitive-benchmark --repetitions 1` runs the
repository benchmarks.]

## Troubleshooting

1. [Issue]
   - Cause: [cause]
   - Solution: [solution]

## Version history

See `CHANGELOG.md` for released and unreleased changes.

## Related documentation

- [[docs/api/api_reference|API reference]]
- [[docs/repo_docs/package_documentation|Package documentation]]
