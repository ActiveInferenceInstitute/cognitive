"""Conformance tests for the committed GNN declaration.

The declaration in ``specs/discrete_variational.gnn.md`` is the canonical
statement of the model and the configuration in
``config_examples/discrete_variational.yaml`` is what the factory builds. These
tests bind the two together, so editing one side alone fails the suite, and they
run the declaration through the revision pinned in ``specs/gnn-pin.json`` when a
runtime for it is available. The required CI job provisions that runtime; a
skipped parser check in a local run is a reported *not run*, never a pass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from cognitive.models.active_inference import gnn_spec
from cognitive.models.active_inference.dispatcher import ActiveInferenceFactory
from scripts.validate_gnn_spec import (
    digest_mismatches,
    main,
    parser_conformance,
    resolve_runtime,
    semantic_conformance,
    validate,
)

SPEC_PATH = gnn_spec.SPEC_PATH
RUNTIME = resolve_runtime()


def _canonical_configuration() -> Path:
    parameters = gnn_spec.parse_model_parameters()
    return gnn_spec.ROOT / parameters["canonical_config"]


def _dispatcher():
    return ActiveInferenceFactory.create_from_yaml(_canonical_configuration())


def test_committed_declaration_declares_every_required_section():
    text = gnn_spec.read_spec_text()
    assert gnn_spec.missing_sections(text) == []
    assert gnn_spec.parse_sections(text)["ModelName"].strip() == gnn_spec.MODEL_NAME


def test_declared_variables_declare_numeric_matrix_dimensions():
    variables = gnn_spec.parse_variables()
    for key in gnn_spec.MATRIX_KEYS:
        assert key in variables
    assert gnn_spec.matrix_dimensions() == {
        "A": (2, 2),
        "B": (2, 2, 2),
        "C": (2,),
        "D": (2,),
        "E": (2,),
    }
    for name in ("s", "s_prime", "o", "u", "G"):
        assert name in variables


def test_canonical_configuration_loads_through_the_factory():
    dispatcher = _dispatcher()
    model = dispatcher.model
    assert (model.num_observations, model.num_states, model.num_actions) == (2, 2, 2)
    assert dispatcher.config.temporal_horizon == 2
    assert np.array_equal(model.A, np.array([[0.9, 0.1], [0.1, 0.9]]))
    assert np.array_equal(model.D, np.array([0.5, 0.5]))


def test_declared_matrices_match_the_runtime_model():
    declared = gnn_spec.declared_matrices()
    model = _dispatcher().model
    expected = {
        "A": model.A,
        "B": np.stack([model.B[:, :, action] for action in range(model.num_actions)]),
        "C": model.C,
        "D": model.D,
        "E": model.E,
    }
    for key, value in expected.items():
        assert declared[key].shape == value.shape, key
        assert np.array_equal(declared[key], value), key


def test_declared_matrices_are_valid_probability_arrays():
    declared = gnn_spec.declared_matrices()
    assert np.allclose(declared["A"].sum(axis=0), 1.0)
    for action in range(declared["B"].shape[0]):
        assert np.allclose(declared["B"][action].sum(axis=0), 1.0)
    for key in ("D", "E"):
        assert np.allclose(declared[key].sum(), 1.0)
    assert np.all(np.isfinite(declared["C"]))


def test_declared_parameters_match_the_runtime_configuration():
    dispatcher = _dispatcher()
    model = dispatcher.model
    parameters = gnn_spec.parse_model_parameters()
    assert parameters["num_hidden_states"] == str(model.num_states)
    assert parameters["num_obs"] == str(model.num_observations)
    assert parameters["num_actions"] == str(model.num_actions)
    assert parameters["num_timesteps"] == str(dispatcher.config.temporal_horizon)
    assert gnn_spec.parse_time()["ModelTimeHorizon"] == str(dispatcher.config.temporal_horizon)
    assert parameters["b_tensor_order"] == "next_state_previous_state_action"


def test_ontology_annotations_cover_every_declared_variable():
    variables = gnn_spec.parse_variables()
    annotations = gnn_spec.parse_ontology()
    assert set(annotations) == set(variables)
    assert all(term.strip() for term in annotations.values())


def test_reader_rejects_a_document_without_a_required_section():
    text = gnn_spec.read_spec_text().replace("## Equations", "## Remarks")
    assert "Equations" in gnn_spec.missing_sections(text)


def test_reader_rejects_unbalanced_braces_and_wrong_shapes():
    text = gnn_spec.read_spec_text()
    with pytest.raises(ValueError, match="unbalanced braces"):
        gnn_spec.declared_matrices(text.replace("E={(0.50, 0.50)}", "E={(0.50, 0.50)"))
    with pytest.raises(ValueError, match="declares 3 values for shape"):
        gnn_spec.declared_matrices(text.replace("C={(0.00, 1.00)}", "C={(0.00, 1.00, 0.00)}"))
    with pytest.raises(ValueError, match="numeric dimensions"):
        gnn_spec.declared_matrices(text.replace("A[2,2,type=float]", "A[states,type=float]"))


def test_validator_reports_a_document_that_disagrees_with_the_configuration(tmp_path: Path):
    text = gnn_spec.read_spec_text().replace("(0.90, 0.10)", "(0.80, 0.20)")
    drifted = tmp_path / "drifted.gnn.md"
    drifted.write_text(text, encoding="utf-8")
    report = validate(drifted)
    assert report["semantics"]["ok"] is False
    assert any(
        "differs from the canonical configuration" in item
        for item in report["semantics"]["differences"]
    )
    assert report["exit_code"] == 1


def test_validator_reports_agreement_and_the_runtime_state():
    report = validate(SPEC_PATH)
    assert report["semantics"]["ok"] is True
    assert semantic_conformance()["ok"] is True
    assert report["semantics"]["configuration"].endswith(
        "config_examples/discrete_variational.yaml"
    )
    if RUNTIME is None:
        assert report["parser"]["status"] == "not run"
        assert report["exit_code"] == 2
    else:
        assert report["parser"]["status"] == "pass", report["parser"]
        assert report["exit_code"] == 0


def test_validator_main_prints_the_report(capsys):
    expected = 0 if RUNTIME is not None else 2
    assert main(["--json"]) == expected
    printed = capsys.readouterr().out
    assert '"status"' in printed and '"semantics"' in printed


def test_digest_check_reports_missing_and_changed_files(tmp_path: Path):
    pin = gnn_spec.load_pin()
    package = tmp_path / "gnn"
    package.mkdir()
    mismatches = digest_mismatches(package, pin)
    assert len(mismatches) == len(pin["parser_files"])
    assert {item["reason"] for item in mismatches} == {"missing"}
    (package / "__init__.py").write_text("not the pinned module\n", encoding="utf-8")
    mismatches = digest_mismatches(package, pin)
    assert any(item["reason"] == "digest mismatch" for item in mismatches)


def test_parser_conformance_rejects_a_disagreeing_payload():
    payload = {
        "gnn_version": "5.0.3",
        "gnn_module": "/tmp/gnn/__init__.py",
        "parser": {
            "success": True,
            "errors": [],
            "model_name": gnn_spec.MODEL_NAME,
            "variables": [{"name": "A", "dimensions": [3, 3]}],
            "ontology_mappings": [],
        },
        "semantic": {"is_valid": False, "errors": ["bad"]},
        "orientation": {"valid": False, "notes": ["rotated"]},
        "consistency": {"consistency_score": 0.5},
    }
    result = parser_conformance(payload)
    assert result["ok"] is False
    assert any("semantic validation failed" in item for item in result["differences"])
    assert any("orientation" in item for item in result["differences"])
    assert any("consistency score" in item for item in result["differences"])
    assert any("A dimensions" in item for item in result["differences"])


def test_runtime_resolution_reads_the_environment(monkeypatch):
    monkeypatch.delenv("GNN_RUNTIME_PYTHON", raising=False)
    monkeypatch.delenv("GNN_ROOT", raising=False)
    monkeypatch.delenv("GNN_PYTHON", raising=False)
    assert resolve_runtime() is None
    assert resolve_runtime("python3") == {
        "kind": "installed",
        "path": "python3",
        "command": ["python3"],
        "env": {},
    }
    monkeypatch.setenv("GNN_ROOT", "/tmp/gnn-checkout")
    monkeypatch.setenv("GNN_PYTHON", "python3.12")
    resolved = resolve_runtime()
    assert resolved["kind"] == "checkout"
    assert resolved["command"] == ["python3.12"]
    assert resolved["env"]["PYTHONPATH"].endswith("gnn-checkout/src")
    assert resolved["package_dir"].endswith("src/gnn")


@pytest.mark.skipif(RUNTIME is None, reason="no pinned GNN runtime named for this run")
def test_pinned_parser_agrees_with_the_reader():
    report = validate(SPEC_PATH)
    assert report["parser"]["status"] == "pass", report["parser"]
    assert report["parser"]["digest_mismatches"] == []
    assert report["parser"]["receipt"]["consistency_score"] == 1.0
    assert report["parser"]["receipt"]["orientation_valid"] is True
    assert report["parser"]["receipt"]["gnn_version"] == gnn_spec.load_pin()["tag"].lstrip("v")
