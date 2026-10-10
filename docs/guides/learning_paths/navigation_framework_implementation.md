---
title: Learning path authoring guide
type: guide
status: stable
updated: 2026-10-10
semantic_relations:
  - type: relates
    links:
      - "[[AGENTS|Learning paths instructions]]"
      - "[[catalog_of_learning_paths|Catalog of learning paths]]"
      - "[[index|Learning path index]]"
---

# Learning path authoring guide

This page describes how the existing learning paths are organized and the module
template they follow. The binding rules (valid frontmatter, resolving links, no
code that claims to implement a path, registration in the catalog and index) are
in [[AGENTS|Learning paths instructions]]. Earlier versions of this page
specified learner analytics, certificates, mentor sessions, and discussion
forums; none of those exist in this repository, so they are not described here.

## Structure of a path

A path is one Markdown page. Most pages open with a quick start and external
resources, then a prerequisites section, an overview with learning outcomes, a
sequence of modules, projects, and links back to the knowledge base. Paths that
need code point to the executed examples in
[[docs/examples/index|Examples index]] instead of embedding their own.

## Module template

```markdown
## Module X: Topic name

### Learning objectives
- Specific, measurable outcomes

### Conceptual foundation
- Core concepts, with links to knowledge-base pages

### Guided exploration
- Step-by-step activities and discussion prompts

### Practical application
- Projects and scenarios, described conceptually

### Assessment
- Checks that the learner can run themselves
```

## Adding a path

1. Create the page in this directory with `title`, `type: learning_path`,
   `status`, and `updated` frontmatter.
2. Link only to pages that exist; run
   `python code/scripts/verify_links.py . --json`.
3. Add the path to [[catalog_of_learning_paths|the catalog]] and to
   [[index|the index]].
4. Run `python code/scripts/validate_docs.py --json`.
