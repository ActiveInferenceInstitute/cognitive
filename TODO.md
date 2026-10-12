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

- **Concept pages left unwritten.** Every `[[concept]]` link now resolves. About 100 of the
  names that had no page were written, with references checked against Crossref, SEP, IEP,
  nLab, Open Library or arXiv; the rest were pointed at an existing page or turned into plain
  text. These were left as plain text because they need a reviewer who can check the subject:
  `k_theory`, `motivic_integration`, `model_categories`, `dg_categories`, `linear_types`,
  `spinor_bundles`, `infinity_groupoids`, `stable_homotopy`, `cylinder_measure`,
  `frechet_spaces`, `nuclear_spaces`, `topological_vector_spaces`, `unbounded_operators`,
  `fredholm_theory`, `tensor_networks`, `quantum_propagator`, `index_theory`,
  `handle_decomposition`, `one_particle_irreducible`, `stationary_phase_approximation`,
  `causal_diamonds`, `orchestrated_reduction`, `object_oriented_ontology`,
  `speculative_materialism`, `new_materialisms`, `queer_theory`, `postcolonial_theory`,
  `racial_formation` and `nichiren_buddhism`. Write them, and link them again where the text
  now reads plainly, only with such a reviewer.
- **Review of the newly written pages.** The 109 pages written in this pass were each checked a
  second time against their sources: every reference was looked up again, about 30 pages had
  claims softened or removed where the source did not support them, and a few references were
  corrected. No subject expert has read them, and books were confirmed by author, title, year and
  publisher, not by page. Have the mathematics and physics pages (`quantum_error_correction`,
  `topological_quantum_field_theory`, `lorentzian_manifolds`, `gauge_theories`, `causal_sets`) and
  the tradition pages (`philosophy/` Indian, Chinese, Japanese, African, Islamic) read by someone
  who can check them.
- **Scholarship.** About 1,100 of roughly 1,350 older content pages list two or more
  references, and every classic work cited on them (Shannon, Pearl, Koller and Friedman,
  Broadbent, Baddeley and Hitch, Rescorla and Wagner, Wertheimer, Watson, Ryle, Miller, Tolman,
  Hebb, Jaynes, Friston 2010) has been checked against its record. The rest of the pages are
  mostly navigation, index and software-usage pages. Parameter values and "illustrative"
  numbers in the colony-biology pseudocode remain unsourced and are labelled as illustrative.
- **Identifier alignment for the RDF graph.** About 720 concept pages carry a `wikidata`
  identifier, exported as `skos:closeMatch`. They were found by label search and screened by a
  language model, not checked one by one: sample them and promote the confirmed ones to
  `skos:exactMatch`. The other concept pages, and MeSH or an ontology for mathematics and physics,
  are not aligned yet.
