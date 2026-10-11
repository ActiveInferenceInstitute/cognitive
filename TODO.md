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

- **Unresolved concept links.** About 5,500 distinct `[[concept]]` names have no page, each
  linked once. Most are single-use names from generated "Related" lists. Unlinking
  them wholesale is an editorial call; the two-link names have been triaged. The checker
  reports them as skipped concept links and does not fail on them.
- **Scholarship.** Several hundred older cognitive-science pages carry broad claims
  with few citations. Review them page by page. Classic references that were not
  individually looked up (Shannon 1948, Pearl, Koller and Friedman, Broadbent, Baddeley and
  Hitch, Rescorla and Wagner, Wertheimer, Watson, Ryle) should be checked against the
  sources.
- **Illustrative code.** Some concept pages contain pseudocode that is not package API.
  The framing notes and the authoring rules in `knowledge_base/AGENTS.md` say so;
  consider replacing the largest examples with runs of the real package.

## Documentation

- `docs/implementation/rxinfer/docs/` (about 50 files) is a vendored snapshot of the
  official RxInfer documentation. It has not been audited, and some pages are out of
  date against the live site.
- `code/Things/ActiveInferenceInstitute/active_inference_institute.md` lost a dated
  2020-2029 history and roadmap because nothing in the repository supports it. Restore
  it from sources if it is wanted.

## Code

- **Aggregation default.** Decide whether `policy_aggregation` should default to
  `marginal` (the standard Bayesian model average) rather than `min` in a future major
  version. Change it in a release of its own, apart from the 1.1.0 objective change.

## Repository

- The remote branch `origin/forked` is fully merged into `main` and can be deleted.
- `.aii/config.yaml` carries an InstituteOS description copied from an old frontmatter
  block. Correct it with the owners of that manifest.
- `.cursorrules` is Cursor-specific guidance that `CLAUDE.md` largely supersedes. Remove
  it if Cursor is no longer used with this repository.
