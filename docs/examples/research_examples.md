---
title: Research examples
type: guide
status: stable
updated: 2026-10-10
semantic_relations:
  - type: relates
    links:
      - "[[docs/guides/research_guides]]"
      - "[[docs/examples/index]]"
      - "[[docs/manuscript/README]]"
---

# Research examples

The research example in this repository is the executable manuscript in
[[docs/manuscript/README|docs/manuscript]]. `cognitive-build-manuscript` runs the
seeded dispatcher and continuous agent, regenerates the figures and variables,
and writes a `build_manifest.json` with the package version, configuration path,
and a SHA-256 hash of the combined source.

```bash
cognitive-build-manuscript --no-pdf --output build/manuscript
```

`--no-pdf` skips LaTeX; `--no-render` skips Pandoc and LaTeX entirely. For the
research workflow around it see [[docs/guides/research_guides|Research guides]],
and for timing diagnostics run `cognitive-benchmark --repetitions 1`.
