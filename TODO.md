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

- **Concept pages to write.** About 355 distinct `[[concept]]` names have no page. Each is
  linked once and was judged a specific, well-defined concept. They are advanced mathematics
  and physics (`k_theory`, `motivic_integration`, `topological_quantum_field_theory`,
  `quantum_error_correction`, `gauge_theories`, `grothendieck_topoi`), regional philosophical
  traditions (`ubuntu_philosophy`, `kyoto_school`, `nyaya_logic`, `huayan`) and critical-theory
  topics (`critical_race_theory`, `queer_theory`). Write them only with a reviewer who can
  check the mathematics or the tradition. The link checker reports them as skipped concept
  links and does not fail on them.
- **Scholarship.** About 1,100 of roughly 1,350 older content pages list two or more
  references, and every classic work cited on them (Shannon, Pearl, Koller and Friedman,
  Broadbent, Baddeley and Hitch, Rescorla and Wagner, Wertheimer, Watson, Ryle, Miller, Tolman,
  Hebb, Jaynes, Friston 2010) has been checked against its record. The rest of the pages are
  mostly navigation, index and software-usage pages. Parameter values and "illustrative"
  numbers in the colony-biology pseudocode remain unsourced and are labelled as illustrative.
- **Illustrative code.** Every page with a substantial Python listing says that it is
  illustrative. The largest are very long (`mathematics/calculus`, `dynamical_systems`,
  `free_energy`, `exploration_exploitation`, `epistemic_value`, each over 1,000 lines of
  code). Trim them to the few snippets that carry the idea, or replace them with runs of the
  real package, and keep the parameter values labelled as illustrative until then.

- **Identifier alignment for the RDF graph.** Concept pages are not yet linked to external
  identifiers (Wikidata, MeSH, a mathematics or physics ontology); only citation pages link
  out, through their DOIs. Align them with a reviewer who can check each match, and add the
  links as `skos:exactMatch` or `skos:closeMatch` in `scripts/export_rdf.py`.

## Code

- **Aggregation default.** Decide whether `policy_aggregation` should default to
  `marginal` (the standard Bayesian model average) rather than `min` in a future major
  version. Change it in a release of its own, apart from the 1.1.0 objective change.
