# InstituteOS integration — cognitive

## Identity

| Field | Value |
| --- | --- |
| Repository | `ActiveInferenceInstitute/cognitive` |
| Affiliation | `institute` |
| Category | `research-software` |
| Status | `active` |
| Maturity | `released` |
| Steward | Active Inference Institute |
| Default branch | `main` |
| Current release | — |
| License | `MIT` |
| Citation | Active Inference Institute — cognitive. |

## Federation

| Setting | Value |
| --- | --- |
| Dashboard mode | `repos` |
| Analytics | `true` |
| Dashboard | `true` |
| Sync policy | `pull` |
| Registries | — |
| Tags | cognitive, active-inference, modeling, python |

## Ecosystem relations

| Repository | Relation | Note |
| --- | --- | --- |
| `ActiveInferenceInstitute/InstituteOS` | `referenced-by` | Federated via the InstituteOS .aii manifest. |
| `ActiveInferenceInstitute/Active_Inference_Ontology` | `references` | Uses Active Inference Ontology terms. |
| `ActiveInferenceInstitute/Generalized_Notation_Notation` | `references` | Canonical discrete model declared in GNN; the parser revision is pinned in specs/gnn-pin.json. |

## Key artifacts

| Path | Kind | Description |
| --- | --- | --- |
| `README.md` | `doc` | Repository overview. |
| `LICENSE` | `doc` | License. |
| `specs/discrete_variational.gnn.md` | `doc` | GNN declaration of the canonical discrete model. |
| `specs/gnn-pin.json` | `data` | Pinned upstream GNN parser revision with parser-file digests. |
| `docs/gnn.md` | `doc` | Model specification - the declaration, its binding, and the validation commands. |
| `code/tools/src/models/active_inference/config_examples/discrete_variational.yaml` | `data` | Canonical runtime configuration bound to the declaration. |
| `code/tools/src/models/active_inference/gnn_spec.py` | `code` | Reader for the committed GNN declaration (standard library plus NumPy). |
| `code/scripts/validate_gnn_spec.py` | `tool` | GNN conformance validator - semantics plus the pinned parser. |
| `code/tests/test_gnn_conformance.py` | `test` | Binds the declaration to the canonical configuration and negative controls. |
| `code/tests/gnn_parser_probe.py` | `tool` | The only module that imports the upstream GNN parser; prints JSON. |
| `AGENTS.md` | `doc` | Agent guidance. |
| `pyproject.toml` | `code` | Python package configuration. |
| `docs` | `doc` | Documentation tree. |
| `CHANGELOG.md` | `doc` | Changelog. |

## Portable tasks

| Task | Command | Working directory | Description |
| --- | --- | --- | --- |
| `docs` | `ls docs` | `.` | List documentation. |
| `inventory` | `git ls-files` | `.` | List tracked files. |

## Validation

```bash
python -m aii_platform.aii_sidecar.validate <repo> --completeness
python -m aii_platform.aii_sidecar docs <repo> --overwrite
python -m aii_platform.aii_sidecar.run <repo> --list
```

The sidecar is valid when `validate` prints `OK` and reports no missing artifact,
manifest, link, or declared-doc path. Completeness is a curation target, not a
runtime requirement.

## Data boundary

This sidecar carries metadata only. It does not grant execution permission, does not
replace the repository's own documentation, and must not be used to move private
content into a public repository. Task execution is limited to trusted checkouts.
