---
title: Cognitive Active Inference — Backlog
type: backlog
status: active
owner: Active Inference Institute
---

# Backlog

Open work only. Finished work is in [`CHANGELOG.md`](CHANGELOG.md) and git
history; this file is not a log. Remove an item when it is done.

## Knowledge base

- **Unresolved concept links.** About 6,500 distinct `[[concept]]` names have no
  page; none has more than two inbound links and about 950 have two. Most are
  single-use names from generated "Related" lists. Triage the two-link ones the
  way the three-link ones were (write a page, add an alias, or unlink). Unlinking
  the single-link names wholesale is a separate editorial call.
- **Scholarship.** Several hundred older cognitive-science pages carry broad
  claims with few citations. Review them page by page. Classic references that
  were not individually looked up (Shannon 1948, Pearl, Koller and Friedman,
  Broadbent, Baddeley and Hitch, Rescorla and Wagner, Wertheimer, Watson, Ryle)
  should be checked against the sources.
- **Illustrative code.** Some concept pages contain pseudocode that is not
  package API. The framing notes and the authoring rules in
  `knowledge_base/AGENTS.md` say so; consider replacing the largest examples with
  runs of the real package.
- `knowledge_base/swarm_intelligence_implementation.md` (988 lines, last updated
  2025-01-01) predates the implemented agents. Check it against `code/Things`.

## Documentation

- `docs/guides/application/parr_2022_chapter_6.md` appears to reproduce a
  published book chapter at length. Decide whether it may stay.
- `docs/implementation/rxinfer/docs/` (about 50 files) is a vendored snapshot of
  the official RxInfer documentation. It has not been audited, and some pages are
  out of date against the live site.
- Mergers suggested by the docs audit: `ast_processing`, `compiler_pipeline` and
  `advanced_systems_engineering_guidebook` into `rxinfer_stack`,
  `systems_engineering_guidebook` and `model_specification`;
  `learning_paths/navigation_framework_implementation.md` into
  `learning_paths/AGENTS.md`; `examples/usage_examples_index.md` and
  `usage_examples.md` into `examples/index.md`.
- `docs/templates/learning_path_template.md` is 1,374 lines of generic content,
  and `cognitive-create-node` accepts four templates that are hand-copy only
  (`ai_concept_template.md`, `guide_template.md`, `learning_path_template.md`,
  `linking_template.md`).
- Several `docs/tools/` pages (`model_context_protocol.md`, `obsidian_usage.md`,
  `network_analysis.md`, `advanced_linking.md`, `automation_scripts.md`,
  `cursor_integration.md`) are broad conceptual write-ups that could be
  consolidated.
- `code/Things/ActiveInferenceInstitute/active_inference_institute.md` lost a
  dated 2020-2029 history and roadmap because nothing in the repository supports
  it. Restore it from sources if it is wanted.

## Code

- **Undeclared dependencies.** `scipy` (used by `NetworkVisualizer`) and `tqdm`
  (used by `docs/repo_docs/repo_scripts`) are imported but not declared.
- **Ignored settings.** `SimplePOMDP` accepts and ignores a per-matrix
  `learning_rate` and the `visualization` fields `formats`, `dpi` and `style`.
  Reject or honour them, as was done for the `inference` section.
- **Overflow warning.** The normalizers emit a `RuntimeWarning` when the total mass
  overflows (inputs near `1e308`) before rescaling. The result is right, but no
  test pins it and a warnings-as-errors run would fail.
- **Manuscript preamble.** `_latex_preamble` leaves mixed `\r\n` and `\n` endings
  for CRLF sources.
- **Fence indentation.** The fence tracker accepts fences indented four or more
  spaces, which differs from CommonMark in a way that errs toward skipping text.
- **Unused option.** The `--mutation` option registered in `conftest.py` is read
  by no test.
- **Aggregation default.** Decide whether `policy_aggregation` should default to
  `marginal` (the standard Bayesian model average) rather than `min` in a future
  major version. Change it in a release of its own, apart from the 1.1.0
  objective change.
- **Mutation coverage.** The mutation set covers the mathematics, the fence
  tracker and one rule of `validate_docs.py`. It does not cover `verify_links.py`
  or `check_markdown_links.py`.

## Repository

- The remote branch `origin/forked` is fully merged into `main` and can be
  deleted.
- `.aii/config.yaml` carries an InstituteOS description copied from an old
  frontmatter block. Correct it with the owners of that manifest.
