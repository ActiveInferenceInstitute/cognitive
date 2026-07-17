# Manuscript syntax and registries

This manuscript follows the syntax used by the Institute template's
`template_code_project` exemplar.

## Citations

Use Pandoc citations such as `[@friston2010free]` or
`[@bogacz2017; @parr2019active]`. Every key must be defined in
`references.bib`; raw citation commands are not used in source Markdown.

## Formalisms

Numbered equations use Pandoc-crossref labels of the form `{#eq:name}` and
are referenced with `[@eq:name]`. The current equation registry is:

| Label | Source |
| --- | --- |
| `eq:generative_model` | `01_introduction.md` |
| `eq:posterior` | `01_introduction.md` |
| `eq:policy_distribution` | `01_introduction.md` |
| `eq:efe_decomposition` | `02_methodology.md` |
| `eq:continuous_prediction_error` | `02_methodology.md` |
| `eq:continuous_update` | `02_methodology.md` |
| `eq:state_round_trip` | `06_reproducibility.md` |
| `eq:normalization_invariant` | `08_appendix_formalism.md` |
| `eq:policy_enumeration` | `08_appendix_formalism.md` |
| `eq:coverage_contract` | `09_appendix_validation.md` |
| `eq:configured_matrices` | `05_experimental_setup.md` |

## Figures and tables

Figures are emitted into the build output by `build_manuscript.py`; source
references use paths relative to `combined.md`:

| Label | File |
| --- | --- |
| `fig:belief_updates` | `figures/belief_updates.png` |
| `fig:policy_distributions` | `figures/policy_distributions.png` |
| `fig:efe_decomposition` | `figures/efe_decomposition.png` |
| `fig:continuous_trajectory` | `figures/continuous_trajectory.png` |
| `fig:model_matrices` | `figures/model_matrices.png` |

Table labels are declared next to the Markdown table. The builder writes a
machine-readable `figure_registry.json` and `build_manifest.json` beside the
rendered files.

## Hydrated variables

Tokens are uppercase names enclosed by double braces. The builder must emit a
value for every token before rendering. Current tokens are
`MODEL_STATES`, `MODEL_OBSERVATIONS`, `MODEL_ACTIONS`, `MODEL_A_SHAPE`,
`MODEL_B_SHAPE`, `HORIZON`, `SEED`, `OBS_SEQUENCE`, `POLICY_COUNT`,
`METHOD_COUNT`, `FIGURE_COUNT`, and `PACKAGE_VERSION`.
