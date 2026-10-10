---
title: Folder structure
type: guide
status: stable
created: 2024-02-12
updated: 2026-10-10
tags:
  - obsidian
  - organization
  - structure
semantic_relations:
  - type: implements
    links:
      - "[[docs/repo_docs/documentation_standards|Documentation standards]]"
  - type: relates
    links:
      - "[[docs/repo_docs/obsidian_usage|Obsidian usage]]"
      - "[[docs/repo_docs/obsidian_linking|Obsidian linking]]"
---

# Folder structure

The repository root is both a Python project and an Obsidian vault
(`.obsidian/` holds the vault settings). The authoritative map, with a
description of every directory, is [`project_structure.md`](../../project_structure.md);
this page gives the shape and the rules for adding to it.

## Layout

```text
cognitive/
├── code/
│   ├── tools/src/        # installed as the `cognitive` package
│   │   ├── models/       #   active_inference/ and matrices/
│   │   ├── utils/        #   matrix_utils, create_node, visualization/
│   │   ├── visualization/#   matrix_plots
│   │   ├── analysis/     #   package with only an __init__.py
│   │   └── benchmarks.py #   the cognitive-benchmark entry point
│   ├── Things/           # installed as the `Things` package
│   ├── scripts/          # installed as `scripts`: validators and builders
│   └── tests/            # the central pytest suite
├── knowledge_base/       # the theory vault, one folder per domain
├── docs/                 # api, guides, examples, implementation, research,
│                         #   templates, tools, config, development, agents,
│                         #   manuscript, policy, repo_docs
├── .github/workflows/quality.yml   # the only CI workflow
├── .obsidian/            # vault settings
├── config.yaml           # defaults read by cognitive-create-node
├── pyproject.toml        # package, entry points, ruff, mypy, pytest, coverage
└── README.md, AGENTS.md, CLAUDE.md, CHANGELOG.md, REVIEW_LOG_*.md, LICENSE
```

`code/Things/` holds two executable agent packages, `Simple_POMDP` and
`Continuous_Generic`. The other folders there (`Ant_Colony`, `BioFirm`,
`Generic_POMDP`, `Generic_Thing`, `Baseball_Game`, `KG_Multi_Agent`,
`Path_Network`, `ActiveInferenceInstitute`) contain documentation only.

`knowledge_base/` domains: `cognitive`, `mathematics`, `free_energy_principle`,
`agents`, `biology`, `systems`, `philosophy`, `ontology`, `research`,
`citations`, and `BioFirm`. Each domain has a generated `catalog.md`.

## Rules for adding files

- Put a page in the domain folder whose subject it is. Give it frontmatter and
  at least one `semantic_relations` link to a neighbour.
- Name files in lowercase `snake_case`; see
  [[docs/repo_docs/naming_conventions|Naming conventions]].
- Do not edit `catalog.md` files. After adding or retitling a knowledge-base
  page, run `python code/scripts/kb_index.py --write`.
- Put tests in `code/tests/`, not beside the module, and keep test artifacts in
  temporary directories.
- A new folder gets a `README.md` and, where there is technical detail to
  record, an `AGENTS.md`.
- Generated output (`output/`, `build/`, coverage data, caches) is git-ignored
  and skipped by the link checkers; do not commit it as documentation.
- A new console script needs an entry under `[project.scripts]` in
  `pyproject.toml`.

## Creating a node from a template

```bash
cognitive-create-node --type agent --name my_agent --var agent_type=example
```

The command renders `docs/templates/agent_template.md` with Jinja2 and writes
`knowledge_base/agents/my_agent.md`; it refuses to overwrite an existing file.
Details are in [[docs/repo_docs/package_documentation|Package documentation]]
and the [[docs/templates/template_guide|Template guide]].

## Related documentation

- [[docs/repo_docs/documentation_standards|Documentation standards]]
- [[docs/repo_docs/ai_folder_structure|AI-oriented folder notes]]
- [[docs/repo_docs/obsidian_usage|Obsidian usage]]
