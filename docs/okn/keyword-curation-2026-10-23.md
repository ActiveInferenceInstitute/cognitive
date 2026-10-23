---
title: Reviewed keyword reuse — 2026-10-23
type: documentation
status: stable
created: 2026-10-23
updated: 2026-10-23
---

# Reviewed keyword reuse — 2026-10-23

The keyword registry reuses source-reviewed external terms as document subjects.
It is defined in [the keyword registry](../../code/scripts/rdf_keywords.py) and mirrored
in `daf-okn` at `src/daf_okn/core/keyword_terms.py`. Both expose
`KEYWORD_TERMS: dict[str, str]` and `keyword_iri(tag: str) -> str | None`.
Lookup uses the exact tag spelling, with no case, whitespace, punctuation or
approximate matching. Each admitted spelling has its own review record.

Existing `/page/` identifiers are document resources. Their keyword links do not
make documents instances of an external biological entity, mathematical object
or clinical condition. Topic resources use separate `/concept/` identifiers and
document-to-topic links use `schema:about` and `dcterms:subject`. The registry
does not infer identity, class membership or additional Simple Knowledge
Organization System (SKOS) mapping assertions. Citation, publication, navigation
and redirect roles remain document roles. Existing reviewed page-topic mappings
and their evidence are independent of these keyword decisions.

## Evidence and admission policy

The National Cancer Institute Thesaurus (NCIt) decisions use the official
[NCI Enterprise Vocabulary Services API](https://api-evsrest.nci.nih.gov/api/v1/metadata/terminologies).
Its current release at review was `26.09d`, dated 2026-09-28. Each admitted NCI
concept response explicitly reports `active: true`; the definitions and alternate
senses were inspected before admission. In particular, the requested subjects are
[Emotion, C54266](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C54266?include=full),
[Genetics, C16624](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16624?include=full)
and [Mathematics, C16825](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16825?include=full).

Medical Subject Headings (MeSH) decisions use the National Library of Medicine's
official descriptor and preferred-concept JSON records. Both records report
active status; the preferred concept supplies the scope note. The registry
records the descriptor's actual update date, rather than calling an old update
date a new vocabulary release. The linked live records were retrieved on the
review date.

Statistics Ontology (STATO) decisions use its authors' official
[2026-04-20 release, pinned to commit a5910e6115217a88ee86e2666e12983ef2ef2a42](https://raw.githubusercontent.com/ISA-tools/stato/a5910e6115217a88ee86e2666e12983ef2ef2a42/releases/latest_release/stato.owl).
The selected classes have definitions and are not marked deprecated in that
release. STATO is distributed under CC BY 3.0; the definition summaries here
are paraphrases. The existing `model_fitting` and `computational_modeling`
identifiers are preserved.

Each `KEYWORD_RECORDS` entry contains the exact source tag, target IRI, official
label, definition, evidence URLs, source version, response SHA-256, review date,
scope rationale, inspected source paths and active status. The MeSH hash binds
the descriptor response; its scope response is separately linked and hashed in
the retained evidence receipts. STATO hashes bind the complete pinned release.
No network request occurs during import. Invalid records, inactive terms,
unsupported vocabularies and duplicate source spellings fail explicitly.

The review compared source frontmatter and representative page bodies, with
special attention to scope collisions and the Markov-blanket sample. The full
tag inventory, context excerpts, file hashes, official response bytes and
machine-checkable verification receipts are retained in `daf-okn` under
`output/entity-model-20261023/keyword-sources/`. Dated reduction estimates belong
to that evidence report, rather than to this policy note. These are scoped
source comparisons by the implementation agent; release acceptance requires
the separate independent gate.

## Exact spellings and decisions

The target links identify the reused resources. The source links provide the
official definition and status; MeSH rows additionally link the preferred
concept's scope. Repeated targets represent separately reviewed source spellings.

| Exact source tag | External target | Primary evidence | Scope decision |
| --- | --- | --- | --- |
| `acetylcholine` | [NCIT:C77840](http://purl.obolibrary.org/obo/NCIT_C77840) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C77840?include=full) | The neurotransmitter substance discussed in cholinergic signaling; not an acetylcholine measurement or administration. |
| `algorithms` | [NCIT:C16275](http://purl.obolibrary.org/obo/NCIT_C16275) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16275?include=full) | Defined computational problem-solving procedures in mathematical methods and implementations; plural source spelling is explicit. |
| `amnesia` | [NCIT:C2867](http://purl.obolibrary.org/obo/NCIT_C2867) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C2867?include=full) | Loss of memory as discussed by cognitive disorder sources; not an adverse-event reporting grade. |
| `amygdala` | [NCIT:C12440](http://purl.obolibrary.org/obo/NCIT_C12440) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C12440?include=full) | The amygdaloid brain structure in emotion and salience sources. |
| `arousal` | [MeSH:D001143](http://id.nlm.nih.gov/mesh/D001143) | [definition/status](https://id.nlm.nih.gov/mesh/D001143.json) [scope](https://id.nlm.nih.gov/mesh/M0001712.json) | Physiological and psychological activation in wakefulness, vigilance and neuromodulation; not merely an evaluation service. |
| `artificial_intelligence` | [NCIT:C16309](http://purl.obolibrary.org/obo/NCIT_C16309) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16309?include=full) | Computational systems and methods for intelligent problem solving; not biological cognition itself. |
| `attention` | [MeSH:D001288](http://id.nlm.nih.gov/mesh/D001288) | [definition/status](https://id.nlm.nih.gov/mesh/D001288.json) [scope](https://id.nlm.nih.gov/mesh/M0001941.json) | Selection and focusing of current experience across endogenous, exogenous, divided and computational attention accounts; not a single test score. |
| `awareness` | [NCIT:C54070](http://purl.obolibrary.org/obo/NCIT_C54070) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C54070?include=full) | Awareness and knowledge of current experience in conscious and cognitive sources. |
| `basal-ganglia` | [NCIT:C12447](http://purl.obolibrary.org/obo/NCIT_C12447) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C12447?include=full) | Explicit hyphenated brain-structure spelling in cognitive circuit sources. |
| `basal_ganglia` | [NCIT:C12447](http://purl.obolibrary.org/obo/NCIT_C12447) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C12447?include=full) | The group of deep cerebral nuclei discussed in motor and cognitive circuits; not a murine-only substitute. |
| `bayesian-inference` | [STATO:0000706](http://purl.obolibrary.org/obo/STATO_0000706) | [definition/status](https://raw.githubusercontent.com/ISA-tools/stato/a5910e6115217a88ee86e2666e12983ef2ef2a42/releases/latest_release/stato.owl) | Explicit hyphenated spelling in Bayesian updating sources; not a rule mapping generic inference to Bayesian inference. |
| `bayesian_inference` | [STATO:0000706](http://purl.obolibrary.org/obo/STATO_0000706) | [definition/status](https://raw.githubusercontent.com/ISA-tools/stato/a5910e6115217a88ee86e2666e12983ef2ef2a42/releases/latest_release/stato.owl) | Updating probability distributions from priors and data using Bayes' rule; general inference deliberately remains local. |
| `behavior` | [NCIT:C16326](http://purl.obolibrary.org/obo/NCIT_C16326) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16326?include=full) | Actions and reactions of organisms or systems to their environment. |
| `behaviour` | [NCIT:C16326](http://purl.obolibrary.org/obo/NCIT_C16326) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16326?include=full) | Explicit British spelling in organism behavior and cognitive sources; not an inferred normalization rule. |
| `biochemistry` | [NCIT:C16337](http://purl.obolibrary.org/obo/NCIT_C16337) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16337?include=full) | Chemical substances and vital processes in living organisms. |
| `biodiversity` | [MeSH:D044822](http://id.nlm.nih.gov/mesh/D044822) | [definition/status](https://id.nlm.nih.gov/mesh/D044822.json) [scope](https://id.nlm.nih.gov/mesh/M0026833.json) | Diversity of living organisms in biological and ecological contexts. |
| `bioinformatics` | [NCIT:C17964](http://purl.obolibrary.org/obo/NCIT_C17964) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C17964?include=full) | Computational treatment of biological information in biological and research contexts. |
| `biology` | [NCIT:C16345](http://purl.obolibrary.org/obo/NCIT_C16345) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16345?include=full) | The scientific study of living organisms across biological knowledge-base sources. |
| `biomechanics` | [NCIT:C16346](http://purl.obolibrary.org/obo/NCIT_C16346) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16346?include=full) | Mechanical principles applied to biological movement and structure. |
| `biophysics` | [NCIT:C16349](http://purl.obolibrary.org/obo/NCIT_C16349) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16349?include=full) | Physical theories and methods applied to biological problems. |
| `causality` | [NCIT:C82552](http://purl.obolibrary.org/obo/NCIT_C82552) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C82552?include=full) | General cause-effect relationships in causal models and philosophy; the general NCI definition applies, not its adverse-event alternate sense. |
| `cell-biology` | [NCIT:C17992](http://purl.obolibrary.org/obo/NCIT_C17992) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C17992?include=full) | Explicit hyphenated spelling reviewed in cell-structure contexts; same biological discipline as cell_biology. |
| `cell_biology` | [NCIT:C17992](http://purl.obolibrary.org/obo/NCIT_C17992) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C17992?include=full) | Cell structure and internal processes at microscopic and molecular scales. |
| `cerebellum` | [NCIT:C12445](http://purl.obolibrary.org/obo/NCIT_C12445) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C12445?include=full) | The cerebellar brain structure in motor and cognitive processing; not the rat-specific target. |
| `cognition` | [NCIT:C16446](http://purl.obolibrary.org/obo/NCIT_C16446) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16446?include=full) | Knowing, thinking, learning and judgment across cognitive models and philosophy; models of cognition remain about the same subject. |
| `cognitive_science` | [MeSH:D019336](http://id.nlm.nih.gov/mesh/D019336) | [definition/status](https://id.nlm.nih.gov/mesh/D019336.json) [scope](https://id.nlm.nih.gov/mesh/M0028784.json) | Interdisciplinary study of mental tasks, brain operations and computational accounts. |
| `communication` | [NCIT:C16452](http://purl.obolibrary.org/obo/NCIT_C16452) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16452?include=full) | Information exchange between people, organisms or systems; the official definition explicitly permits objects and groups. |
| `computation` | [NCIT:C61298](http://purl.obolibrary.org/obo/NCIT_C61298) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C61298?include=full) | Calculation by mathematical or logical methods across the mathematical and cognitive sources. |
| `computational_modeling` | [NCIT:C16461](http://purl.obolibrary.org/obo/NCIT_C16461) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16461?include=full) | Preserved: tutorial and computational psychiatry use mathematical computer models, matching the NCI definition rather than its radiation-planning alternate sense. |
| `consciousness` | [MeSH:D003243](http://id.nlm.nih.gov/mesh/D003243) | [definition/status](https://id.nlm.nih.gov/mesh/D003243.json) [scope](https://id.nlm.nih.gov/mesh/M0005030.json) | Awareness of self and environment in cognitive and philosophical accounts; not merely clinical alertness and orientation. |
| `coordination` | [NCIT:C25462](http://purl.obolibrary.org/obo/NCIT_C25462) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C25462?include=full) | Bringing individual movements or agents into common action, including collective and motor coordination. |
| `cybernetics` | [MeSH:D003491](http://id.nlm.nih.gov/mesh/D003491) | [definition/status](https://id.nlm.nih.gov/mesh/D003491.json) [scope](https://id.nlm.nih.gov/mesh/M0005443.json) | Communication and control in living organisms and machines, matching biological and artificial source systems. |
| `decision-making` | [MeSH:D003657](http://id.nlm.nih.gov/mesh/D003657) | [definition/status](https://id.nlm.nih.gov/mesh/D003657.json) [scope](https://id.nlm.nih.gov/mesh/M0005721.json) | Explicit hyphenated spelling in choice and action-selection sources; same reviewed subject. |
| `decision_making` | [MeSH:D003657](http://id.nlm.nih.gov/mesh/D003657) | [definition/status](https://id.nlm.nih.gov/mesh/D003657.json) [scope](https://id.nlm.nih.gov/mesh/M0005721.json) | Choosing among alternatives to determine action in cognitive, mathematical and multi-agent models. |
| `deep_learning` | [NCIT:C176258](http://purl.obolibrary.org/obo/NCIT_C176258) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C176258?include=full) | Artificial multilayer neural-network learning; not biological learning in general. |
| `ecology` | [NCIT:C16526](http://purl.obolibrary.org/obo/NCIT_C16526) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16526?include=full) | Organism-environment relationships across ecosystems, social insects and ecological accounts of cognition. |
| `economics` | [NCIT:C61331](http://purl.obolibrary.org/obo/NCIT_C61331) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C61331?include=full) | Study of production, distribution and consumption and its models in the source economics pages. |
| `education` | [NCIT:C16529](http://purl.obolibrary.org/obo/NCIT_C16529) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16529?include=full) | Teaching and imparting knowledge or skills in research, learning and educational applications. |
| `electrophysiology` | [NCIT:C16540](http://purl.obolibrary.org/obo/NCIT_C16540) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16540?include=full) | Electrical phenomena in living bodies, including neural recordings. |
| `emotion` | [NCIT:C54266](http://purl.obolibrary.org/obo/NCIT_C54266) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C54266?include=full) | Affect and feelings in appraisal, regulation, social interaction and active-inference accounts; no disease or particular emotion is asserted. |
| `entomology` | [NCIT:C16550](http://purl.obolibrary.org/obo/NCIT_C16550) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16550?include=full) | Scientific study of insects, including social-insect models and observations. |
| `episodic_memory` | [MeSH:D061212](http://id.nlm.nih.gov/mesh/D061212) | [definition/status](https://id.nlm.nih.gov/mesh/D061212.json) [scope](https://id.nlm.nih.gov/mesh/M0535198.json) | Memory for experienced events and their context in cognitive memory sources; immune memory is excluded. |
| `ethics` | [NCIT:C39796](http://purl.obolibrary.org/obo/NCIT_C39796) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C39796?include=full) | Moral values and rules across the philosophical traditions represented in the source; does not identify ethics with one tradition. |
| `executive-function` | [NCIT:C88419](http://purl.obolibrary.org/obo/NCIT_C88419) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C88419?include=full) | Explicit hyphenated spelling in cognitive control sources; same mental-process scope. |
| `executive_function` | [NCIT:C88419](http://purl.obolibrary.org/obo/NCIT_C88419) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C88419?include=full) | Mental processes that manage cognitive functions; not a generic software executive. |
| `eye_movements` | [MeSH:D005133](http://id.nlm.nih.gov/mesh/D005133) | [definition/status](https://id.nlm.nih.gov/mesh/D005133.json) [scope](https://id.nlm.nih.gov/mesh/M0008093.json) | Movement of the eyes in oculomotor and perception sources; does not designate a particular saccade. |
| `game-theory` | [MeSH:D005716](http://id.nlm.nih.gov/mesh/D005716) | [definition/status](https://id.nlm.nih.gov/mesh/D005716.json) [scope](https://id.nlm.nih.gov/mesh/M0008969.json) | Explicit hyphenated spelling in strategic-interaction contexts; same mathematical field. |
| `game_theory` | [MeSH:D005716](http://id.nlm.nih.gov/mesh/D005716) | [definition/status](https://id.nlm.nih.gov/mesh/D005716.json) [scope](https://id.nlm.nih.gov/mesh/M0008969.json) | Mathematical analysis of interacting decision makers and their interests; not all decision theory. |
| `genetics` | [NCIT:C16624](http://purl.obolibrary.org/obo/NCIT_C16624) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16624?include=full) | Heredity and inherited variation across gene expression, epigenetics and population genetics; the discipline is the document subject. |
| `hippocampus` | [NCIT:C12444](http://purl.obolibrary.org/obo/NCIT_C12444) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C12444?include=full) | The named brain structure and its subregions in memory and spatial cognition; a subject link does not classify a page as brain tissue. |
| `homeostasis` | [NCIT:C25989](http://purl.obolibrary.org/obo/NCIT_C25989) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C25989?include=full) | Maintenance of the internal physiological environment and active-inference accounts of biological homeostatic regulation. |
| `hormone` | [NCIT:C2315](http://purl.obolibrary.org/obo/NCIT_C2315) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C2315?include=full) | Explicit singular spelling used for endocrine signaling substances, not a treatment regimen. |
| `hormones` | [NCIT:C2315](http://purl.obolibrary.org/obo/NCIT_C2315) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C2315?include=full) | Chemical signaling substances in endocrine contexts; not hormone treatment. |
| `immunology` | [NCIT:C18011](http://purl.obolibrary.org/obo/NCIT_C18011) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C18011?include=full) | Study of immune-system function; not immune-function diagnostic testing. |
| `information-theory` | [MeSH:D007257](http://id.nlm.nih.gov/mesh/D007257) | [definition/status](https://id.nlm.nih.gov/mesh/D007257.json) [scope](https://id.nlm.nih.gov/mesh/M0011324.json) | Explicit hyphenated spelling in entropy, mutual information and free-energy mathematical accounts; same quantitative subject. |
| `information_theory` | [MeSH:D007257](http://id.nlm.nih.gov/mesh/D007257) | [definition/status](https://id.nlm.nih.gov/mesh/D007257.json) [scope](https://id.nlm.nih.gov/mesh/M0011324.json) | Quantitative information transmission and representations across entropy, coding and information measures; not semantic meaning by label alone. |
| `interoception` | [MeSH:D065812](http://id.nlm.nih.gov/mesh/D065812) | [definition/status](https://id.nlm.nih.gov/mesh/D065812.json) [scope](https://id.nlm.nih.gov/mesh/M0591076.json) | Perception of internal bodily stimuli in physiological and active-inference sources. |
| `ion_channels` | [MeSH:D007473](http://id.nlm.nih.gov/mesh/D007473) | [definition/status](https://id.nlm.nih.gov/mesh/D007473.json) [scope](https://id.nlm.nih.gov/mesh/M0011674.json) | Membrane channels that conduct ions, as used by biological excitability and neural signaling sources. |
| `language` | [NCIT:C16779](http://purl.obolibrary.org/obo/NCIT_C16779) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16779?include=full) | Natural-language communication and its formal accounts; source contexts concern words, grammar and meaning rather than a programming-language identifier. |
| `learning` | [NCIT:C19369](http://purl.obolibrary.org/obo/NCIT_C19369) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C19369?include=full) | Acquisition of knowledge or skill, including computational accounts of that process; not limited to a particular learning algorithm. |
| `linguistics` | [MeSH:D008037](http://id.nlm.nih.gov/mesh/D008037) | [definition/status](https://id.nlm.nih.gov/mesh/D008037.json) [scope](https://id.nlm.nih.gov/mesh/M0012527.json) | Scientific study of natural language and its structure, rather than a programming language. |
| `logic` | [MeSH:D008128](http://id.nlm.nih.gov/mesh/D008128) | [definition/status](https://id.nlm.nih.gov/mesh/D008128.json) [scope](https://id.nlm.nih.gov/mesh/M0012674.json) | Principles of valid inference and demonstration, including mathematical logic; not Bayesian inference alone. |
| `machine-learning` | [NCIT:C176231](http://purl.obolibrary.org/obo/NCIT_C176231) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C176231?include=full) | Explicit hyphenated spelling in computational learning sources; same reviewed sense as machine_learning. |
| `machine_learning` | [NCIT:C176231](http://purl.obolibrary.org/obo/NCIT_C176231) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C176231?include=full) | Computational learning from examples in machine-learning sources. |
| `mathematics` | [NCIT:C16825](http://purl.obolibrary.org/obo/NCIT_C16825) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16825?include=full) | Mathematical methods, probability, geometry and formal reasoning; includes the Markov-blanket document as a subject, not a mathematical instance. |
| `matrices` | [STATO:0000642](http://purl.obolibrary.org/obo/STATO_0000642) | [definition/status](https://raw.githubusercontent.com/ISA-tools/stato/a5910e6115217a88ee86e2666e12983ef2ef2a42/releases/latest_release/stato.owl) | Explicit plural spelling in POMDP and numerical-matrix sources; no normalization is performed by lookup. |
| `matrix` | [STATO:0000642](http://purl.obolibrary.org/obo/STATO_0000642) | [definition/status](https://raw.githubusercontent.com/ISA-tools/stato/a5910e6115217a88ee86e2666e12983ef2ef2a42/releases/latest_release/stato.owl) | Rectangular numerical arrays in POMDP matrices and their mathematical structure; not tissue matrix or a dosing unit. |
| `memory_consolidation` | [MeSH:D000069077](http://id.nlm.nih.gov/mesh/D000069077) | [definition/status](https://id.nlm.nih.gov/mesh/D000069077.json) [scope](https://id.nlm.nih.gov/mesh/M000604637.json) | Stabilization of acquired memories in the observed cognitive and neural sources. |
| `metacognition` | [MeSH:D000067489](http://id.nlm.nih.gov/mesh/D000067489) | [definition/status](https://id.nlm.nih.gov/mesh/D000067489.json) [scope](https://id.nlm.nih.gov/mesh/M000599741.json) | Awareness and understanding of one's own thought processes across judgments of learning and monitoring. |
| `metaphysics` | [MeSH:D008678](http://id.nlm.nih.gov/mesh/D008678) | [definition/status](https://id.nlm.nih.gov/mesh/D008678.json) [scope](https://id.nlm.nih.gov/mesh/M0013524.json) | First principles of existence and being across philosophical traditions; not a formal knowledge-representation ontology. |
| `microbiology` | [NCIT:C16851](http://purl.obolibrary.org/obo/NCIT_C16851) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16851?include=full) | Study of microorganisms in the biological sources; not a diagnostic service. |
| `model_fitting` | [STATO:0000218](http://purl.obolibrary.org/obo/STATO_0000218) | [definition/status](https://raw.githubusercontent.com/ISA-tools/stato/a5910e6115217a88ee86e2666e12983ef2ef2a42/releases/latest_release/stato.owl) | Preserved: the cited tutorial fits an active-inference model to empirical data; this is dataset-to-model fit assessment. |
| `molecular-biology` | [NCIT:C16872](http://purl.obolibrary.org/obo/NCIT_C16872) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16872?include=full) | Explicit hyphenated spelling in molecular and cellular sources; scope matches molecular_biology. |
| `molecular_biology` | [NCIT:C16872](http://purl.obolibrary.org/obo/NCIT_C16872) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16872?include=full) | Molecular-scale cell and gene processes; no individual molecule or gene is asserted. |
| `monte_carlo` | [NCIT:C94936](http://purl.obolibrary.org/obo/NCIT_C94936) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C94936?include=full) | Randomized repeated computation in MCMC, importance sampling and particle methods; not biological specimen sampling. |
| `motivation` | [NCIT:C28495](http://purl.obolibrary.org/obo/NCIT_C28495) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C28495?include=full) | Psychological drives and incentives underlying behavior. |
| `natural_selection` | [MeSH:D012641](http://id.nlm.nih.gov/mesh/D012641) | [definition/status](https://id.nlm.nih.gov/mesh/D012641.json) [scope](https://id.nlm.nih.gov/mesh/M0019598.json) | Differential reproductive success associated with heritable traits in biological evolution. |
| `neural_plasticity` | [MeSH:D009473](http://id.nlm.nih.gov/mesh/D009473) | [definition/status](https://id.nlm.nih.gov/mesh/D009473.json) [scope](https://id.nlm.nih.gov/mesh/M0014735.json) | Capacity of nervous systems to change in response to activation in sprouting and map-plasticity sources; ambiguous plasticity remains local. |
| `neuroanatomy` | [NCIT:C16905](http://purl.obolibrary.org/obo/NCIT_C16905) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16905?include=full) | Anatomy of the nervous system across named brain structures and cortical circuits. |
| `neuroendocrinology` | [NCIT:C16907](http://purl.obolibrary.org/obo/NCIT_C16907) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16907?include=full) | Interactions between nervous and endocrine systems. |
| `neuroimaging` | [NCIT:C173635](http://purl.obolibrary.org/obo/NCIT_C173635) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C173635?include=full) | Imaging nervous-system structure or function; does not imply a particular modality. |
| `neurology` | [NCIT:C16908](http://purl.obolibrary.org/obo/NCIT_C16908) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16908?include=full) | The medical discipline concerned with the nervous system; not a service billing category. |
| `neurophysiology` | [NCIT:C16910](http://purl.obolibrary.org/obo/NCIT_C16910) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16910?include=full) | Physiological functions of the nervous system. |
| `neuropsychology` | [NCIT:C16911](http://purl.obolibrary.org/obo/NCIT_C16911) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16911?include=full) | Brain and nervous-system relations to behavior and cognitive function. |
| `neuroscience` | [MeSH:D009488](http://id.nlm.nih.gov/mesh/D009488) | [definition/status](https://id.nlm.nih.gov/mesh/D009488.json) [scope](https://id.nlm.nih.gov/mesh/M0014763.json) | The scientific disciplines studying the nervous system; the source uses a singular discipline label for MeSH Neurosciences. |
| `optimization` | [NCIT:C78561](http://purl.obolibrary.org/obo/NCIT_C78561) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C78561?include=full) | Finding an optimum solution across numerical, variational, biological and cognitive optimization; no algorithm-specific replacement. |
| `perception` | [MeSH:D010465](http://id.nlm.nih.gov/mesh/D010465) | [definition/status](https://id.nlm.nih.gov/mesh/D010465.json) [scope](https://id.nlm.nih.gov/mesh/M0016252.json) | Recognition and interpretation of sensory stimuli; prefer MeSH over the conscious-image-only NCI definition for computational and biological accounts. |
| `personality` | [NCIT:C38002](http://purl.obolibrary.org/obo/NCIT_C38002) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C38002?include=full) | Patterns of individual psychological characteristics, as discussed in cognitive and clinical sources. |
| `pharmacology` | [NCIT:C16974](http://purl.obolibrary.org/obo/NCIT_C16974) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16974?include=full) | Study of drug properties and effects, not a particular therapeutic product. |
| `pheromones` | [MeSH:D010675](http://id.nlm.nih.gov/mesh/D010675) | [definition/status](https://id.nlm.nih.gov/mesh/D010675.json) [scope](https://id.nlm.nih.gov/mesh/M0016588.json) | Chemical signals between members of the same species in social-insect communication; not all chemical signaling. |
| `philosophy` | [NCIT:C39794](http://purl.obolibrary.org/obo/NCIT_C39794) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C39794?include=full) | Investigation of existence, knowledge and ethics, including philosophical discussion of the free-energy principle. |
| `physics` | [NCIT:C16989](http://purl.obolibrary.org/obo/NCIT_C16989) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16989?include=full) | Laws of matter and energy across physical and mathematical treatments. |
| `physiology` | [NCIT:C16990](http://purl.obolibrary.org/obo/NCIT_C16990) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16990?include=full) | Mechanical, physical and biochemical functions of living organisms. |
| `population_genetics` | [NCIT:C16628](http://purl.obolibrary.org/obo/NCIT_C16628) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16628?include=full) | Gene distributions and transmission across populations, not individual genotypes. |
| `prefrontal_cortex` | [NCIT:C154779](http://purl.obolibrary.org/obo/NCIT_C154779) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C154779?include=full) | The prefrontal brain region and its cognitive functions; not all frontal cortex. |
| `probability` | [NCIT:C54154](http://purl.obolibrary.org/obo/NCIT_C54154) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C54154?include=full) | Event likelihood and its mathematical calculus in distributions, conditional independence and Markov blankets; not a specific probability distribution. |
| `problem_solving` | [MeSH:D011340](http://id.nlm.nih.gov/mesh/D011340) | [definition/status](https://id.nlm.nih.gov/mesh/D011340.json) [scope](https://id.nlm.nih.gov/mesh/M0017603.json) | Selecting alternatives to reach a goal in cognitive and computational problem-solving sources. |
| `psychiatry` | [NCIT:C17026](http://purl.obolibrary.org/obo/NCIT_C17026) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C17026?include=full) | Medical management and computational investigation of mental disorders. |
| `psychometrics` | [NCIT:C17034](http://purl.obolibrary.org/obo/NCIT_C17034) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C17034?include=full) | Measurement of psychological characteristics and cognitive performance. |
| `quantum_mechanics` | [NCIT:C19650](http://purl.obolibrary.org/obo/NCIT_C19650) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C19650?include=full) | Quantum physical principles, including mathematical and computational applications. |
| `reasoning` | [NCIT:C86584](http://purl.obolibrary.org/obo/NCIT_C86584) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C86584?include=full) | Coherent logical thinking, including cognitive and philosophical accounts of it. |
| `regulation` | [NCIT:C48805](http://purl.obolibrary.org/obo/NCIT_C48805) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C48805?include=full) | Biological and cognitive processes controlling other processes; explicitly reject the legal-rule sense of C68821. |
| `rehabilitation` | [NCIT:C15315](http://purl.obolibrary.org/obo/NCIT_C15315) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C15315?include=full) | Restoration of function in clinical and cognitive sources; not a service-type identifier. |
| `research` | [NCIT:C15429](http://purl.obolibrary.org/obo/NCIT_C15429) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C15429?include=full) | Systematic investigation across research applications; use the active Research Activity concept, never retired C15319. |
| `retina` | [NCIT:C12343](http://purl.obolibrary.org/obo/NCIT_C12343) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C12343?include=full) | The sensory retinal tissue in visual processing; not a murine-specific concept. |
| `reward` | [NCIT:C99754](http://purl.obolibrary.org/obo/NCIT_C99754) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C99754?include=full) | Positive reinforcement and reward-related motivation; no numeric utility value or pharmacological treatment is asserted. |
| `semantics` | [NCIT:C54194](http://purl.obolibrary.org/obo/NCIT_C54194) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C54194?include=full) | Meaning in language and formal representations, including compositional semantics and semantic information. |
| `sleep` | [NCIT:C73425](http://purl.obolibrary.org/obo/NCIT_C73425) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C73425?include=full) | The biological periodic resting state, including sleep-related cognitive processes. |
| `social-cognition` | [MeSH:D000083282](http://id.nlm.nih.gov/mesh/D000083282) | [definition/status](https://id.nlm.nih.gov/mesh/D000083282.json) [scope](https://id.nlm.nih.gov/mesh/M000678997.json) | Explicit hyphenated spelling in social perception, trust and multi-agent cognitive accounts. |
| `social_cognition` | [MeSH:D000083282](http://id.nlm.nih.gov/mesh/D000083282) | [definition/status](https://id.nlm.nih.gov/mesh/D000083282.json) [scope](https://id.nlm.nih.gov/mesh/M000678997.json) | Perception and interpretation of one's own and others' social behavior, including computational accounts. |
| `social_psychology` | [NCIT:C17033](http://purl.obolibrary.org/obo/NCIT_C17033) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C17033?include=full) | Psychological processes in social interaction, not all social science. |
| `speech_perception` | [MeSH:D013067](http://id.nlm.nih.gov/mesh/D013067) | [definition/status](https://id.nlm.nih.gov/mesh/D013067.json) [scope](https://id.nlm.nih.gov/mesh/M0020254.json) | Recognition and interpretation of spoken language sounds in cognitive auditory sources. |
| `statistics` | [NCIT:C17167](http://purl.obolibrary.org/obo/NCIT_C17167) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C17167?include=full) | Quantitative data interpretation and probabilistic estimation; not the measured output of a particular study. |
| `stress` | [NCIT:C35041](http://purl.obolibrary.org/obo/NCIT_C35041) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C35041?include=full) | Psychological and physiological stress responses in appraisal and neuroendocrine contexts; not mechanical stress. |
| `synaptic_transmission` | [MeSH:D009435](http://id.nlm.nih.gov/mesh/D009435) | [definition/status](https://id.nlm.nih.gov/mesh/D009435.json) [scope](https://id.nlm.nih.gov/mesh/M0014683.json) | Communication across synapses in nervous-system sources; not generic communication. |
| `systems-biology` | [MeSH:D049490](http://id.nlm.nih.gov/mesh/D049490) | [definition/status](https://id.nlm.nih.gov/mesh/D049490.json) [scope](https://id.nlm.nih.gov/mesh/M0464791.json) | Explicit hyphenated spelling in biological-system modeling contexts. |
| `systems-theory` | [MeSH:D013598](http://id.nlm.nih.gov/mesh/D013598) | [definition/status](https://id.nlm.nih.gov/mesh/D013598.json) [scope](https://id.nlm.nih.gov/mesh/M0020984.json) | Explicit hyphenated spelling in complex, adaptive and control-system sources; same systems-theory subject. |
| `systems_biology` | [MeSH:D049490](http://id.nlm.nih.gov/mesh/D049490) | [definition/status](https://id.nlm.nih.gov/mesh/D049490.json) [scope](https://id.nlm.nih.gov/mesh/M0464791.json) | Systematic analysis and modeling of interacting biological processes; not all systems theory. |
| `systems_theory` | [MeSH:D013598](http://id.nlm.nih.gov/mesh/D013598) | [definition/status](https://id.nlm.nih.gov/mesh/D013598.json) [scope](https://id.nlm.nih.gov/mesh/M0020984.json) | Principles and models of interdependent components and functioning wholes; not every tag called systems. |
| `temporal_lobe` | [NCIT:C12353](http://purl.obolibrary.org/obo/NCIT_C12353) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C12353?include=full) | The temporal cerebral lobe in language, memory and perception sources. |
| `theory_of_mind` | [MeSH:D056345](http://id.nlm.nih.gov/mesh/D056345) | [definition/status](https://id.nlm.nih.gov/mesh/D056345.json) [scope](https://id.nlm.nih.gov/mesh/M0526999.json) | Attributing mental states to oneself and others in social-cognitive sources. |
| `thermodynamics` | [NCIT:C17193](http://purl.obolibrary.org/obo/NCIT_C17193) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C17193?include=full) | Energy conversion in physical systems; this does not equate thermodynamic free energy with variational free energy. |
| `time` | [NCIT:C25207](http://purl.obolibrary.org/obo/NCIT_C25207) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C25207?include=full) | Temporal experience and ordering discussed by mathematical and philosophical sources; not a unit or duration measurement. |
| `vision` | [NCIT:C38533](http://purl.obolibrary.org/obo/NCIT_C38533) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C38533?include=full) | All observed contexts concern visual perception, binocular vision or visual processing, not future plans or an organizational vision. |
| `visualization` | [NCIT:C85439](http://purl.obolibrary.org/obo/NCIT_C85439) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C85439?include=full) | Computer-generated diagrams, plots and visual environments in the observed sources; not mental imagery. |
| `working-memory` | [NCIT:C88418](http://purl.obolibrary.org/obo/NCIT_C88418) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C88418?include=full) | Explicit hyphenated spelling in cognitive working-memory sources; same immediate-task scope. |
| `working_memory` | [NCIT:C88418](http://purl.obolibrary.org/obo/NCIT_C88418) | [definition/status](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C88418?include=full) | Short-term storage, processing and recall needed for immediate cognitive tasks; does not encompass immune memory. |

## Deliberately retained local keywords

A similar label is a discovery lead, not a scope comparison. The following
families were examined and retained locally. Other unadmitted tags also retain
their exact local identifiers; absence from the registry does not mean that no
external term exists.

| Exact tags or family | Reason for retaining local resources |
| --- | --- |
| `active_inference`, `active-inference`, `free_energy_principle`, `free-energy-principle` | Specialized framework usage needs its own reviewed concept comparison; general inference or a neuroscience-only framing is insufficient. |
| `inference`, `bayesian`, `variational_inference` | None is an exact spelling or scope alias for the admitted Bayesian-inference tags. General inference must not become Bayesian inference. |
| `free_energy`, `free-energy`, `variational_free_energy` | Physical and variational meanings differ. A thermodynamic Free Energy label cannot substitute for variational free energy. |
| `precision`, `precision_weighting` | [NCI C48045](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C48045?include=full) defines a quantitative representation of accuracy associated with an action; STATO precision denotes positive predictive value. Neither justifies precision weighting or inverse-variance usage throughout this source. |
| `memory` | The source includes `biology/immune_memory.md` as well as mental memory. A cognition-only term would silently narrow the global spelling. The separately reviewed working- and episodic-memory tags are admitted. |
| `plasticity`, `neural_networks`, `neural-networks` | General plasticity and biological versus artificial networks require separate scope decisions. The explicit `neural_plasticity` spelling has a narrower reviewed use. |
| `self`, `adaptation`, `development` | The source mixes prereflective and reflective selfhood, cognitive and evolutionary adaptation, and several developmental meanings. Narrower NCI definitions do not cover every use. |
| `ontology`, `knowledge_base` | Philosophical ontology differs from a formal representation ontology; Obsidian knowledge-base navigation does not imply a formal-language database. Document roles stay explicit. |
| `topology`, `reading`, `integration` | [NCI C54101](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C54101?include=full) is regional anatomy, [C48356](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C48356?include=full) is an instrument datum, and [C48038](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C48038?include=full) is calculus integration. Those are false matches for the global source spellings. |
| `sampling`, `uncertainty`, `prediction` | Physical specimen sampling, measurement uncertainty and future extrapolation are too narrow for all observed computational and cognitive uses. |
| `systems`, `control`, `stability`, `conservation`, `organization`, `representation`, `theory` | Broad, mixed uses do not justify substituting an available narrow or disciplinary sense. The explicit systems-theory spellings are independently admitted. |
| `graphical_models`, `statistical_independence` | Their Markov-blanket usages are clear, but this review located no suitable active exact term in its bounded NCI, MeSH and STATO comparison. Do not replace graphical models with Bayesian networks or independence with a statistical test. |
| `epistemology`, `philosophy_of_mind`, `geometry`, `differential_geometry`, `information_geometry` | These specialized subjects were not replaced with broader philosophy, mathematics or an unrelated same-label term. |
| `citation`, `index`, `navigation`, `tutorial`, `documentation` | These may express document or navigation roles. The keyword registry does not turn them into topic-concept type assertions. |

The retired [NCI Research C15319](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C15319?include=full)
is rejected. The admitted `research` tag uses active C15429 after checking its
definition. For attention, perception and consciousness, MeSH supplies scopes
that fit the source better than the narrower NCI concentration, conscious-image
and clinical-alertness definitions. Regulation uses a process sense (C48805),
not the legal-rule sense (C68821).

[NCI Inference C75591](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C75591?include=full)
specifies conclusions derived from stated facts by logical rules. The source's
general `inference` spelling also covers probabilistic and variational models,
so this logical-rule definition is not admitted globally.
[NCI Free Energy C18218](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C18218?include=full)
specifies a thermodynamic sense; it is not used for variational free energy.

In `knowledge_base/mathematics/markov_blanket.md`, the exact `mathematics` and
`probability` subjects use NCI C16825 and C54154. `graphical_models` and
`statistical_independence` retain local keywords. This does not change any page
body or its document/topic distinction.

## Verification

[Focused keyword tests](../../code/tests/test_rdf_keywords.py) exercise exact lookup,
explicit spelling variants, accepting controls, rejected malformed and inactive
records, duplicate and conflicting source tags, and an isolated standard-library
import with a rejecting network audit hook. The same controls exist in `daf-okn`
at `tests/core/test_keyword_terms.py`. Real imported modules are compared for
metadata and source parity by the dated evidence checker. The cognitive runtime
source manifest includes `rdf_keywords.py` so a changed registry invalidates an
older capture. Root-owned full suites, regeneration and fresh cross-graph
captures remain separate checks.
