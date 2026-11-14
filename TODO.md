---
title: Cognitive Active Inference — Backlog
type: backlog
status: active
owner: Active Inference Institute
updated: 2026-11-12
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
- **Identifier alignment for the RDF graph.** 697 knowledge-base pages carry a
  `wikidata` identifier. The export publishes each as an `rdfs:seeAlso`
  discovery link, not a SKOS mapping (`code/scripts/export_rdf.py`), and only
  reviewed `external_mappings` become mapping assertions: currently 53
  assertions over 47 pages, listed in `docs/okn/alignment-review.md`. Those
  identifiers were found by label search and screened twice by a language model
  (the second pass dropped 23 of 720), not checked one by one. Label search
  and language-model screening do not establish semantic equivalence.
  Review each source concept and target denotation individually before adding
  a reviewed `external_mappings` assertion under the evidence rules in
  `docs/okn/mapping-curation-2026-10-20.md`. Use `skos:exactMatch` only when
  that review supports equivalence; leave unreviewed identifiers as discovery
  links rather than promoting them automatically. The other concept pages,
  and MeSH or an ontology for mathematics and physics, require the same
  explicit review.

## Environment prerequisites

- **Documentation gate dependencies.** Use the shared root gate: `./run.sh docs`
  runs in the locked environment (`uv run --locked --extra dev --python 3.12`)
  and prepares PyYAML and the RDF modules without a separate editable install
  (`docs/development/README.md`). Invoking `code/scripts/validate_docs.py`
  directly in a checkout whose environment lacks them stops at the import;
  `code/tests/test_docs_sync.py` also imports the installed `cognitive` package.
  The arm covers frontmatter, link and manuscript checks, the generated
  documentation index, measured-gate evidence freshness, OKN submission and
  quality checks, and semantic-review freshness.
- **Manuscript toolchain.** `./run.sh manuscript` builds the no-render pipeline
  and HTML; `--pdf` adds the PDF. HTML needs Pandoc and `pandoc-crossref` on
  `PATH`, and the PDF also needs XeLaTeX. `cognitive-build-manuscript
  --no-render` builds figures and hydrated source without either, and
  `--no-pdf` skips only the PDF.
- **GNN conformance.** The canonical pin is `specs/gnn-pin.json`
  (`gnn-importer-pin/v1`, tag `v5.0.3`, commit `8f293d9`) with SHA-256 digests
  for five parser files. It is not on the `v5.1.0` shipped-importer cutover:
  this repository keeps its own reader and the pinned parser probe
  (`docs/gnn.md`, pattern D). `./run.sh gnn --provision` checks the tag and
  provisions the parser; without a runtime, `code/scripts/validate_gnn_spec.py`
  exits 2 and reports the parser as not run, which the gate treats as a
  failure. A moved tag or mismatched digest is a failure, not permission to use
  another revision.
- **Hosted Actions are billing-locked.** Run `43246633657` on `main` at
  `bfa9468` failed in 3 s per job with "The job was not started because your
  account is locked due to a billing issue"; the workflow files and the local
  `./run.sh` arms are the contract, not a hosted success. `.github/GNN.md` and
  `docs/development/README.md` say the same.
