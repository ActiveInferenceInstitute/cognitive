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

- **Concept pages to write.** About 1,300 distinct `[[concept]]` names have no page. Each is
  linked once and was judged a specific, well-defined concept (for example
  `planum_temporale`, `temporal_difference_learning`, `hermeneutic_phenomenology`,
  `kolmogorov_equations`). Write them in batches by domain. The link checker reports
  them as skipped concept links and does not fail on them.
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
