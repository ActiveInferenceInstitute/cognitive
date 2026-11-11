---
title: Documentation hub
type: documentation
status: stable
updated: 2026-11-11
semantic_relations:
  - type: organizes
    links:
      - "[[../README|Repository README]]"
      - "[[api/README|API]]"
      - "[[cli/README|Command line]]"
      - "[[visualization/README|Visualization]]"
      - "[[architecture/README|Architecture]]"
      - "[[config/README|Configuration]]"
      - "[[config/settings|Settings reference]]"
      - "[[development/README|Development]]"
      - "[[examples/README|Examples]]"
      - "[[guides/README|Guides]]"
      - "[[implementation/README|Implementation]]"
      - "[[research/README|Research]]"
      - "[[tools/README|Tools]]"
      - "[[templates/README|Templates]]"
      - "[[agents/README|Agent documentation]]"
      - "[[manuscript/README|Manuscript]]"
      - "[[policy/README|Documentation term policy]]"
      - "[[okn/README|Open Knowledge Network export]]"
      - "[[repo_docs/README|Repository documentation]]"
      - "[[gnn|Model specification]]"
---

# Documentation hub

This directory contains documentation for the implemented Python package, the
executable manuscript, conceptual research notes, and the repository's Obsidian
knowledge-vault material. The root README and the manuscript are the canonical
working entry points.

## Start here

1. [`../README.md`](../README.md): install, a minimal discrete model, commands,
   and quality gates.
2. [`repo_docs/getting_started.md`](repo_docs/getting_started.md): fresh-checkout
   setup and verification.
3. [`manuscript/README.md`](manuscript/README.md): end-to-end build and
   reproducibility contract.
4. [`api/README.md`](api/README.md): API documentation policy.
5. [`examples/README.md`](examples/README.md): example pages, with a pointer to
   the runnable scripts in [`../examples/README.md`](../examples/README.md).
6. [`cli/README.md`](cli/README.md): run a simulation or a sweep from the shell.

## Documentation domains

- [`api/`](api/README.md): public API reference, executed examples, versioning.
- [`cli/`](cli/README.md): the `cognitive` command and the report and export commands.
- [`visualization/`](visualization/README.md): the figure gallery and what each plot shows.
- [`architecture/`](architecture/README.md): how settings, simulation, plots and the command line fit together.
- [`config/`](config/README.md): the root `config.yaml` and which keys code reads;
  [`config/settings.md`](config/settings.md) is the reference for every setting.
- [`development/`](development/README.md): development loop and CI.
- [`agents/`](agents/README.md): navigation for agent documentation.
- [`manuscript/`](manuscript/README.md): the executable manuscript.
- [`gnn.md`](gnn.md): the GNN declaration of the canonical discrete model and how
  it is bound to the runtime configuration.
- [`policy/`](policy/README.md): the term policy the documentation gate enforces.
- [`okn/`](okn/README.md): the knowledge base as RDF for the Open Knowledge Network.
- [`examples/`](examples/README.md): validated example entry points.
- [`guides/`](guides/README.md): application notes and learning paths.
- [`implementation/`](implementation/README.md): implementation patterns and the
  RxInfer documentation subtree.
- [`research/`](research/README.md): research methods and application notes.
- [`repo_docs/`](repo_docs/README.md): repository documentation standards and
  maintenance material.
- [`templates/`](templates/README.md): document templates.
- [`tools/`](tools/README.md): repository tool documentation.

## Knowledge base

The conceptual knowledge base is at [`../knowledge_base/README.md`](../knowledge_base/README.md).
Its source files use Obsidian wiki links and may contain theoretical or
illustrative material beyond the implemented runtime. For code behavior, prefer
the installed package, its tests, and the manuscript's reproducibility section.

The [project structure](../project_structure.md) maps these folders; open work is
in the backlog file at the repository root.

## Validation

From the repository root:

```bash
python code/scripts/validate_docs.py --json
python code/scripts/verify_links.py . --json
python code/scripts/check_markdown_links.py . --json
python code/scripts/normalize_frontmatter.py --check .
python code/scripts/kb_index.py --check .
```

`validate_docs.py` is the aggregate gate: frontmatter, quoted
`semantic_relations` links, compile-checking of Python fences, package exports,
the term policy, wiki and Markdown links, knowledge-base catalogs, and
manuscript references. `verify_links.py` checks wiki links that name a file;
unresolved extensionless concept links are reported separately unless
`--strict-wiki-links` is passed. `normalize_frontmatter.py` and `kb_index.py` are
also run inside `validate_docs.py`; run them directly with `--write` to fix what
they report.

## Complete page index

Every page in this tree, grouped by folder. The earlier sections introduce the
same pages by topic; this list is exhaustive and points at the manuscript and
tools sub-trees as well.

### `./`

- [AGENTS](AGENTS.md)
- [gnn](gnn.md)

### `agents/`

- [AGENTS](agents/AGENTS.md)
- [README](agents/README.md)
- [agent_docs_readme](agents/agent_docs_readme.md)

### `api/`

- [AGENTS](api/AGENTS.md)
- [README](api/README.md)
- [agent_api](api/agent_api.md)
- [ant_colony_api](api/ant_colony_api.md)
- [api_documentation](api/api_documentation.md)
- [api_documentation_index](api/api_documentation_index.md)
- [api_examples](api/api_examples.md)
- [api_reference](api/api_reference.md)
- [api_versioning](api/api_versioning.md)
- [research_api](api/research_api.md)

### `architecture/`

- [AGENTS](architecture/AGENTS.md)
- [README](architecture/README.md)
- [episodes](architecture/episodes.md)

### `cli/`

- [AGENTS](cli/AGENTS.md)
- [README](cli/README.md)
- [reference](cli/reference.md)

### `config/`

- [AGENTS](config/AGENTS.md)
- [README](config/README.md)
- [settings](config/settings.md)

### `development/`

- [AGENTS](development/AGENTS.md)
- [README](development/README.md)
- [contribution_guide](development/contribution_guide.md)

### `examples/`

- [AGENTS](examples/AGENTS.md)
- [README](examples/README.md)
- [advanced_api_patterns](examples/advanced_api_patterns.md)
- [agent_examples](examples/agent_examples.md)
- [ant_colony_example](examples/ant_colony_example.md)
- [api_integration_examples](examples/api_integration_examples.md)
- [basic_api_usage](examples/basic_api_usage.md)
- [index](examples/index.md)
- [research_examples](examples/research_examples.md)
- [usage_examples](examples/usage_examples.md)
- [usage_examples_index](examples/usage_examples_index.md)

### `guides/`

- [AGENTS](guides/AGENTS.md)
- [README](guides/README.md)
- [agent_development](guides/agent_development.md)
- [ant_colony_guide](guides/ant_colony_guide.md)
- [api_implementation](guides/api_implementation.md)
- [best_practices](guides/best_practices.md)
- [research_guides](guides/research_guides.md)

### `guides/application/`

- [AGENTS](guides/application/AGENTS.md)
- [README](guides/application/README.md)
- [active_inference_spatial_applications](guides/application/active_inference_spatial_applications.md)
- [guide_for_cognitive_modeling](guides/application/guide_for_cognitive_modeling.md)
- [parr_2022_chapter_6](guides/application/parr_2022_chapter_6.md)

### `guides/learning_paths/`

- [AGENTS](guides/learning_paths/AGENTS.md)
- [README](guides/learning_paths/README.md)
- [active_inference_agi_learning_path](guides/learning_paths/active_inference_agi_learning_path.md)
- [active_inference_bio_inspired_cognitive_systems_path](guides/learning_paths/active_inference_bio_inspired_cognitive_systems_path.md)
- [active_inference_biological_learning_path](guides/learning_paths/active_inference_biological_learning_path.md)
- [active_inference_cognitive_learning_path](guides/learning_paths/active_inference_cognitive_learning_path.md)
- [active_inference_cognitive_systems_integration_path](guides/learning_paths/active_inference_cognitive_systems_integration_path.md)
- [active_inference_computational_learning_path](guides/learning_paths/active_inference_computational_learning_path.md)
- [active_inference_ecological_learning_path](guides/learning_paths/active_inference_ecological_learning_path.md)
- [active_inference_economic_learning_path](guides/learning_paths/active_inference_economic_learning_path.md)
- [active_inference_edge_cognitive_systems_path](guides/learning_paths/active_inference_edge_cognitive_systems_path.md)
- [active_inference_educational_programs_path](guides/learning_paths/active_inference_educational_programs_path.md)
- [active_inference_environmental_applications_path](guides/learning_paths/active_inference_environmental_applications_path.md)
- [active_inference_ethics_learning_path](guides/learning_paths/active_inference_ethics_learning_path.md)
- [active_inference_experience_method_path](guides/learning_paths/active_inference_experience_method_path.md)
- [active_inference_fuller_learning_path](guides/learning_paths/active_inference_fuller_learning_path.md)
- [active_inference_information_theory_learning_path](guides/learning_paths/active_inference_information_theory_learning_path.md)
- [active_inference_learning_path](guides/learning_paths/active_inference_learning_path.md)
- [active_inference_mathematical_learning_path](guides/learning_paths/active_inference_mathematical_learning_path.md)
- [active_inference_myrmecology_learning_path](guides/learning_paths/active_inference_myrmecology_learning_path.md)
- [active_inference_neuroscience_learning_path](guides/learning_paths/active_inference_neuroscience_learning_path.md)
- [active_inference_observation_learning_path](guides/learning_paths/active_inference_observation_learning_path.md)
- [active_inference_pragmatism_learning_path](guides/learning_paths/active_inference_pragmatism_learning_path.md)
- [active_inference_quantum_cognitive_systems_path](guides/learning_paths/active_inference_quantum_cognitive_systems_path.md)
- [active_inference_quantum_learning_path](guides/learning_paths/active_inference_quantum_learning_path.md)
- [active_inference_resilient_systems_path](guides/learning_paths/active_inference_resilient_systems_path.md)
- [active_inference_robotics_learning_path](guides/learning_paths/active_inference_robotics_learning_path.md)
- [active_inference_security_cognitive_systems_path](guides/learning_paths/active_inference_security_cognitive_systems_path.md)
- [active_inference_security_learning_path](guides/learning_paths/active_inference_security_learning_path.md)
- [active_inference_simulation_learning_path](guides/learning_paths/active_inference_simulation_learning_path.md)
- [active_inference_social_learning_path](guides/learning_paths/active_inference_social_learning_path.md)
- [active_inference_social_science_abm_learning_path](guides/learning_paths/active_inference_social_science_abm_learning_path.md)
- [active_inference_social_systems_learning_path](guides/learning_paths/active_inference_social_systems_learning_path.md)
- [active_inference_spatial_web_path](guides/learning_paths/active_inference_spatial_web_path.md)
- [active_inference_underwriting_learning_path](guides/learning_paths/active_inference_underwriting_learning_path.md)
- [active_inference_whose_on_first_learning_path](guides/learning_paths/active_inference_whose_on_first_learning_path.md)
- [catalog_of_learning_paths](guides/learning_paths/catalog_of_learning_paths.md)
- [index](guides/learning_paths/index.md)
- [navigation_framework_implementation](guides/learning_paths/navigation_framework_implementation.md)

### `implementation/`

- [AGENTS](implementation/AGENTS.md)
- [README](implementation/README.md)
- [implementation_guides](implementation/implementation_guides.md)
- [implementation_guides_index](implementation/implementation_guides_index.md)
- [implementation_patterns](implementation/implementation_patterns.md)

### `implementation/rxinfer/`

- [AGENTS](implementation/rxinfer/AGENTS.md)
- [README](implementation/rxinfer/README.md)
- [active_inference_examples](implementation/rxinfer/active_inference_examples.md)
- [advanced_systems_engineering_guidebook](implementation/rxinfer/advanced_systems_engineering_guidebook.md)
- [ast_processing](implementation/rxinfer/ast_processing.md)
- [compiler_pipeline](implementation/rxinfer/compiler_pipeline.md)
- [execution_engine](implementation/rxinfer/execution_engine.md)
- [factor_graphs](implementation/rxinfer/factor_graphs.md)
- [free_energy](implementation/rxinfer/free_energy.md)
- [free_energy_message_passing_active_inference](implementation/rxinfer/free_energy_message_passing_active_inference.md)
- [free_energy_message_passing_details](implementation/rxinfer/free_energy_message_passing_details.md)
- [getting_started](implementation/rxinfer/getting_started.md)
- [message_passing](implementation/rxinfer/message_passing.md)
- [model_macro_paradigm](implementation/rxinfer/model_macro_paradigm.md)
- [model_specification](implementation/rxinfer/model_specification.md)
- [reactive_programming](implementation/rxinfer/reactive_programming.md)
- [rxinfer_mountain_car_readme](implementation/rxinfer/rxinfer_mountain_car_readme.md)
- [rxinfer_stack](implementation/rxinfer/rxinfer_stack.md)
- [rxinfer_variational_inference](implementation/rxinfer/rxinfer_variational_inference.md)
- [streaming_inference](implementation/rxinfer/streaming_inference.md)
- [systems_engineering_guidebook](implementation/rxinfer/systems_engineering_guidebook.md)
- [variational_inference](implementation/rxinfer/variational_inference.md)

### `manuscript/`

- [00_abstract](manuscript/00_abstract.md)
- [01_introduction](manuscript/01_introduction.md)
- [02_methodology](manuscript/02_methodology.md)
- [03_results](manuscript/03_results.md)
- [04_conclusion](manuscript/04_conclusion.md)
- [05_experimental_setup](manuscript/05_experimental_setup.md)
- [06_reproducibility](manuscript/06_reproducibility.md)
- [07_scope_and_related_work](manuscript/07_scope_and_related_work.md)
- [08_appendix_formalism](manuscript/08_appendix_formalism.md)
- [09_appendix_validation](manuscript/09_appendix_validation.md)
- [99_references](manuscript/99_references.md)
- [AGENTS](manuscript/AGENTS.md)
- [README](manuscript/README.md)
- [SYNTAX](manuscript/SYNTAX.md)
- [preamble](manuscript/preamble.md)

### `okn/`

- [AGENTS](okn/AGENTS.md)
- [README](okn/README.md)
- [alignment-review](okn/alignment-review.md)
- [curation-workflow](okn/curation-workflow.md)
- [entity-model](okn/entity-model.md)
- [graph-quality](okn/graph-quality.md)
- [graph_description](okn/graph_description.md)
- [interoperability](okn/interoperability.md)
- [join-support](okn/join-support.md)
- [keyword-curation-2026-10-23](okn/keyword-curation-2026-10-23.md)
- [landing_zone](okn/landing_zone.md)
- [mapping-curation-2026-10-20](okn/mapping-curation-2026-10-20.md)
- [registry_entry](okn/registry_entry.md)
- [semantic-review](okn/semantic-review.md)
- [statistics](okn/statistics.md)
- [submission_criteria](okn/submission_criteria.md)
- [vocabulary](okn/vocabulary.md)

### `okn/evidence/`

- [AGENTS](okn/evidence/AGENTS.md)
- [README](okn/evidence/README.md)

### `okn/evidence/2026-10-13/`

- [join-results](okn/evidence/2026-10-13/join-results.md)

### `okn/evidence/2026-10-20/`

- [join-results](okn/evidence/2026-10-20/join-results.md)

### `okn/evidence/2026-10-20-graph-methods/`

- [join-results](okn/evidence/2026-10-20-graph-methods/join-results.md)

### `okn/evidence/2026-10-23-entity-model/`

- [join-results](okn/evidence/2026-10-23-entity-model/join-results.md)

### `okn/explanations/`

- [AGENTS](okn/explanations/AGENTS.md)
- [README](okn/explanations/README.md)

### `okn/explanations/2026-10-23-entity-model/`

- [join-support](okn/explanations/2026-10-23-entity-model/join-support.md)

### `okn/queries/`

- [AGENTS](okn/queries/AGENTS.md)
- [README](okn/queries/README.md)

### `policy/`

- [AGENTS](policy/AGENTS.md)
- [README](policy/README.md)

### `repo_docs/`

- [AGENTS](repo_docs/AGENTS.md)
- [README](repo_docs/README.md)
- [ai_documentation_style](repo_docs/ai_documentation_style.md)
- [ai_file_organization](repo_docs/ai_file_organization.md)
- [ai_folder_structure](repo_docs/ai_folder_structure.md)
- [ai_semantic_processing](repo_docs/ai_semantic_processing.md)
- [ai_validation_framework](repo_docs/ai_validation_framework.md)
- [api_development](repo_docs/api_development.md)
- [content_management](repo_docs/content_management.md)
- [documentation_standards](repo_docs/documentation_standards.md)
- [folder_structure](repo_docs/folder_structure.md)
- [getting_started](repo_docs/getting_started.md)
- [linking_analysis](repo_docs/linking_analysis.md)
- [linking_completeness](repo_docs/linking_completeness.md)
- [linking_patterns](repo_docs/linking_patterns.md)
- [linking_validation](repo_docs/linking_validation.md)
- [machine_learning](repo_docs/machine_learning.md)
- [model_implementation](repo_docs/model_implementation.md)
- [naming_conventions](repo_docs/naming_conventions.md)
- [obsidian_linking](repo_docs/obsidian_linking.md)
- [obsidian_usage](repo_docs/obsidian_usage.md)
- [package_documentation](repo_docs/package_documentation.md)
- [research](repo_docs/research.md)
- [simulation](repo_docs/simulation.md)
- [testing_guidelines](repo_docs/testing_guidelines.md)
- [unit_testing](repo_docs/unit_testing.md)
- [validation](repo_docs/validation.md)

### `repo_docs/repo_scripts/`

- [AGENTS](repo_docs/repo_scripts/AGENTS.md)
- [README](repo_docs/repo_scripts/README.md)
- [link_fix_prompt](repo_docs/repo_scripts/link_fix_prompt.md)

### `research/`

- [AGENTS](research/AGENTS.md)
- [README](research/README.md)
- [ant_colony_active_inference](research/ant_colony_active_inference.md)
- [index](research/index.md)
- [research_documentation](research/research_documentation.md)
- [research_documentation_index](research/research_documentation_index.md)

### `research/active_inference/`

- [AGENTS](research/active_inference/AGENTS.md)
- [README](research/active_inference/README.md)
- [applications](research/active_inference/applications.md)
- [hierarchical](research/active_inference/hierarchical.md)
- [scaling](research/active_inference/scaling.md)
- [theory](research/active_inference/theory.md)

### `research/architectures/`

- [AGENTS](research/architectures/AGENTS.md)
- [README](research/architectures/README.md)
- [continuous](research/architectures/continuous.md)
- [hierarchical](research/architectures/hierarchical.md)
- [multi_agent](research/architectures/multi_agent.md)
- [pomdp](research/architectures/pomdp.md)

### `research/complex_systems/`

- [AGENTS](research/complex_systems/AGENTS.md)
- [README](research/complex_systems/README.md)
- [adaptation](research/complex_systems/adaptation.md)
- [collective](research/complex_systems/collective.md)
- [emergence](research/complex_systems/emergence.md)
- [self_organization](research/complex_systems/self_organization.md)

### `research/examples/`

- [AGENTS](research/examples/AGENTS.md)
- [README](research/examples/README.md)

### `research/guidelines/`

- [AGENTS](research/guidelines/AGENTS.md)
- [README](research/guidelines/README.md)

### `research/implementations/`

- [AGENTS](research/implementations/AGENTS.md)
- [README](research/implementations/README.md)

### `research/methodology/`

- [AGENTS](research/methodology/AGENTS.md)
- [README](research/methodology/README.md)
- [hypothesis](research/methodology/hypothesis.md)
- [sampling](research/methodology/sampling.md)
- [validation](research/methodology/validation.md)
- [variables](research/methodology/variables.md)

### `research/standards/`

- [AGENTS](research/standards/AGENTS.md)
- [README](research/standards/README.md)

### `research/templates/`

- [AGENTS](research/templates/AGENTS.md)
- [README](research/templates/README.md)

### `research/tools/`

- [AGENTS](research/tools/AGENTS.md)
- [README](research/tools/README.md)
- [analysis_tools](research/tools/analysis_tools.md)
- [documentation_tools](research/tools/documentation_tools.md)
- [implementation_tools](research/tools/implementation_tools.md)

### `research/validation/`

- [AGENTS](research/validation/AGENTS.md)
- [README](research/validation/README.md)

### `templates/`

- [AGENTS](templates/AGENTS.md)
- [README](templates/README.md)
- [action_template](templates/action_template.md)
- [agent_template](templates/agent_template.md)
- [ai_concept_template](templates/ai_concept_template.md)
- [analysis_template](templates/analysis_template.md)
- [belief_template](templates/belief_template.md)
- [cognitive_architecture_template](templates/cognitive_architecture_template.md)
- [cognitive_concept](templates/cognitive_concept.md)
- [documentation_templates](templates/documentation_templates.md)
- [documentation_templates_index](templates/documentation_templates_index.md)
- [environment_template](templates/environment_template.md)
- [experiment_template](templates/experiment_template.md)
- [goal_template](templates/goal_template.md)
- [guide_template](templates/guide_template.md)
- [implementation_example](templates/implementation_example.md)
- [learning_path_template](templates/learning_path_template.md)
- [linking_template](templates/linking_template.md)
- [memory_system_template](templates/memory_system_template.md)
- [observation_template](templates/observation_template.md)
- [package_component](templates/package_component.md)
- [reasoning_system_template](templates/reasoning_system_template.md)
- [research_document](templates/research_document.md)
- [template_guide](templates/template_guide.md)
- [template_index](templates/template_index.md)

### `tools/`

- [AGENTS](tools/AGENTS.md)
- [README](tools/README.md)
- [advanced_linking](tools/advanced_linking.md)
- [api_doc_tools](tools/api_doc_tools.md)
- [automation_scripts](tools/automation_scripts.md)
- [benchmark_tools](tools/benchmark_tools.md)
- [build_system](tools/build_system.md)
- [ci_tools](tools/ci_tools.md)
- [config_tools](tools/config_tools.md)
- [coverage_tools](tools/coverage_tools.md)
- [dependency_analyzer](tools/dependency_analyzer.md)
- [dev_scripts](tools/dev_scripts.md)
- [doc_validator](tools/doc_validator.md)
- [git_tools](tools/git_tools.md)
- [git_workflow](tools/git_workflow.md)
- [graph_generator](tools/graph_generator.md)
- [link_checker](tools/link_checker.md)
- [linting_tools](tools/linting_tools.md)
- [model_context_protocol](tools/model_context_protocol.md)
- [mutation_testing](tools/mutation_testing.md)
- [network_analysis](tools/network_analysis.md)
- [obsidian_usage](tools/obsidian_usage.md)
- [packaging_tools](tools/packaging_tools.md)
- [plotting_tools](tools/plotting_tools.md)
- [setup_tools](tools/setup_tools.md)
- [static_analyzer](tools/static_analyzer.md)
- [style_checker](tools/style_checker.md)
- [test_generators](tools/test_generators.md)

### `visualization/`

- [AGENTS](visualization/AGENTS.md)
- [README](visualization/README.md)
- [gallery](visualization/gallery.md)

## Licensing

Code is MIT licensed under [`../LICENSE`](../LICENSE). Repository-owned knowledge-base
content and graph exports are CC BY 4.0 under [`../LICENSE-KNOWLEDGE-BASE.md`](../LICENSE-KNOWLEDGE-BASE.md).
Other documentation remains CC BY-NC-SA 4.0.
