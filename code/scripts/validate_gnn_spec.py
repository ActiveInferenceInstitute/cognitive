"""Validate the committed GNN declaration with the pinned upstream parser.

Two checks run, in this order:

1. **Semantics** (always): every required section is declared, the declared
   ``A``/``B``/``C``/``D``/``E`` arrays and dimensions equal the canonical
   configuration in
   ``code/tools/src/models/active_inference/config_examples/discrete_variational.yaml``
   loaded through ``ActiveInferenceFactory.create_from_yaml``, and the declared
   model parameters equal the runtime model's.
2. **Pinned parser** (needs a runtime): the revision recorded in
   ``specs/gnn-pin.json`` parses the document, agrees with
   ``cognitive.models.active_inference.gnn_spec`` on every variable, and reports
   no error, no semantic error, no B-orientation warning, and full consistency.
   The parser files are digest-verified before their output is trusted.

Exit codes: ``0`` pass, ``1`` conformance failure, ``2`` the pinned runtime is
unavailable, so the parser check was not run — which is never a pass.

Usage::

    python -m scripts.validate_gnn_spec --gnn-runtime <python> --json
    GNN_RUNTIME_PYTHON=<python> python -m scripts.validate_gnn_spec
    GNN_ROOT=<gnn-checkout> GNN_PYTHON=<python> python -m scripts.validate_gnn_spec
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from cognitive.models.active_inference import gnn_spec
from cognitive.models.active_inference.dispatcher import ActiveInferenceFactory

PROBE_PATH = gnn_spec.ROOT / "code" / "tests" / "gnn_parser_probe.py"


def digest_mismatches(package_dir: Path, pin: dict[str, Any]) -> list[dict[str, str]]:
    """Return the recorded parser files that are missing or have changed."""
    mismatches: list[dict[str, str]] = []
    for recorded, expected in pin.get("parser_files", {}).items():
        relative = recorded.split("src/gnn/", 1)[-1]
        path = package_dir / relative
        if not path.is_file():
            mismatches.append({"path": recorded, "reason": "missing"})
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            mismatches.append({"path": recorded, "reason": "digest mismatch"})
    return mismatches


def resolve_runtime(
    runtime: str | None = None, root: str | None = None, python: str | None = None
) -> dict[str, Any] | None:
    """Return how to run the probe, or ``None`` when no runtime was named."""
    runtime_python = runtime or os.environ.get("GNN_RUNTIME_PYTHON")
    if runtime_python:
        return {"kind": "installed", "path": runtime_python, "command": [runtime_python], "env": {}}
    gnn_root = root or os.environ.get("GNN_ROOT")
    if gnn_root:
        interpreter = python or os.environ.get("GNN_PYTHON") or sys.executable
        checkout = Path(gnn_root).expanduser().resolve()
        return {
            "kind": "checkout",
            "path": str(checkout),
            "command": [interpreter],
            "env": {"PYTHONPATH": str(checkout / "src")},
            "package_dir": str(checkout / "src" / "gnn"),
        }
    return None


def run_probe(runtime: dict[str, Any], spec_path: Path) -> dict[str, Any]:
    """Run the probe under ``runtime`` and return its JSON payload."""
    env = dict(os.environ)
    env.update(runtime.get("env", {}))
    completed = subprocess.run(
        [*runtime["command"], str(PROBE_PATH), "--spec", str(spec_path)],
        capture_output=True,
        check=False,
        env=env,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip() or "no stderr"
        raise RuntimeError(f"probe exited {completed.returncode}: {detail[-600:]}")
    return json.loads(completed.stdout.decode("utf-8"))


def semantic_conformance(text: str | None = None) -> dict[str, Any]:
    """Compare the declaration with the canonical runtime configuration."""
    body = gnn_spec.read_spec_text() if text is None else text
    differences = [f"missing section: {name}" for name in gnn_spec.missing_sections(body)]
    declared = gnn_spec.declared_matrices(body)
    declared_dims = gnn_spec.matrix_dimensions(body)
    parameters = gnn_spec.parse_model_parameters(body)
    configuration = gnn_spec.ROOT / parameters.get("canonical_config", "")

    if not configuration.is_file():
        recorded = parameters.get("canonical_config")
        differences.append(f"canonical configuration is missing: {recorded}")
        return {
            "ok": False,
            "configuration": parameters.get("canonical_config", ""),
            "differences": differences,
        }

    dispatcher = ActiveInferenceFactory.create_from_yaml(configuration)
    model = dispatcher.model
    expected = {
        "A": model.A,
        "B": np.stack([model.B[:, :, action] for action in range(model.num_actions)]),
        "C": model.C,
        "D": model.D,
        "E": model.E,
    }
    expected_dims = {
        "A": (model.num_observations, model.num_states),
        "B": tuple(model.B.shape),
        "C": (model.num_observations,),
        "D": (model.num_states,),
        "E": (model.num_actions,),
    }
    for key in gnn_spec.MATRIX_KEYS:
        if declared_dims[key] != expected_dims[key]:
            differences.append(
                f"{key} declares dimensions {declared_dims[key]}, "
                f"the model has {expected_dims[key]}"
            )
        if declared[key].shape != expected[key].shape:
            differences.append(
                f"{key} declares shape {declared[key].shape}, the model has {expected[key].shape}"
            )
        elif not np.array_equal(declared[key], expected[key]):
            differences.append(f"{key} differs from the canonical configuration")

    declared_parameters = {
        "num_hidden_states": model.num_states,
        "num_obs": model.num_observations,
        "num_actions": model.num_actions,
        "num_timesteps": dispatcher.config.temporal_horizon,
    }
    for name, value in declared_parameters.items():
        if parameters.get(name) != str(value):
            differences.append(f"{name} declares {parameters.get(name)!r}, the runtime has {value}")
    horizon = gnn_spec.parse_time(body).get("ModelTimeHorizon")
    if horizon != str(dispatcher.config.temporal_horizon):
        differences.append(
            f"ModelTimeHorizon declares {horizon!r}, "
            f"the runtime has {dispatcher.config.temporal_horizon}"
        )

    return {
        "ok": not differences,
        "configuration": parameters.get("canonical_config", ""),
        "differences": differences,
    }


def parser_conformance(payload: dict[str, Any], text: str | None = None) -> dict[str, Any]:
    """Compare the pinned parser's reading of the document with the reader's."""
    body = gnn_spec.read_spec_text() if text is None else text
    parser = payload.get("parser", {})
    differences: list[str] = []
    if not parser.get("success"):
        differences.append(f"the pinned parser reported errors: {parser.get('errors')}")
    if not payload.get("semantic", {}).get("is_valid"):
        semantic_errors = payload.get("semantic", {}).get("errors")
        differences.append(f"semantic validation failed: {semantic_errors}")
    orientation = payload.get("orientation", {})
    if not orientation.get("valid", False):
        differences.append(f"B orientation is not canonical: {orientation.get('notes')}")
    score = payload.get("consistency", {}).get("consistency_score")
    if score != 1.0:
        differences.append(f"consistency score is {score}, expected 1.0")

    declared_variables = gnn_spec.parse_variables(body)
    parser_variables = {
        str(variable["name"]): tuple(int(dim) for dim in variable["dimensions"])
        for variable in parser.get("variables", [])
    }
    for key in gnn_spec.MATRIX_KEYS:
        declared = tuple(int(token) for token in declared_variables[key])
        if parser_variables.get(key) != declared:
            differences.append(
                f"{key} dimensions: parser {parser_variables.get(key)}, reader {declared}"
            )
    model_name = gnn_spec.parse_sections(body).get("ModelName", "").strip()
    if parser.get("model_name") != model_name:
        differences.append(
            f"model name: parser {parser.get('model_name')!r}, reader {model_name!r}"
        )
    parser_ontology = {
        str(mapping["variable"]): str(mapping["term"])
        for mapping in parser.get("ontology_mappings", [])
    }
    declared_ontology = gnn_spec.parse_ontology(body)
    if parser_ontology != declared_ontology:
        differences.append(
            f"ontology annotations: parser {parser_ontology}, reader {declared_ontology}"
        )
    return {
        "ok": not differences,
        "differences": differences,
        "receipt": {
            "gnn_version": payload.get("gnn_version"),
            "gnn_module": payload.get("gnn_module"),
            "consistency_score": score,
            "orientation_valid": orientation.get("valid", False),
        },
    }


def validate(
    spec_path: Path | None = None,
    runtime: str | None = None,
    root: str | None = None,
    python: str | None = None,
) -> dict[str, Any]:
    """Run both checks and return the report with its exit code."""
    path = spec_path or gnn_spec.SPEC_PATH
    text = path.read_text(encoding="utf-8")
    pin = gnn_spec.load_pin()
    try:
        document = str(path.relative_to(gnn_spec.ROOT))
    except ValueError:
        document = str(path)
    report: dict[str, Any] = {
        "document": document,
        "pin": {"tag": pin.get("tag"), "commit": pin.get("commit")},
        "semantics": semantic_conformance(text),
        "parser": {"status": "not run"},
    }
    resolved = resolve_runtime(runtime, root, python)
    if resolved is None:
        report["parser"] = {
            "status": "not run",
            "reason": (
                "no pinned runtime named (--gnn-runtime/GNN_RUNTIME_PYTHON or --gnn-root/GNN_ROOT)"
            ),
        }
        report["runtime"] = None
    else:
        try:
            payload = run_probe(resolved, path)
        except (OSError, RuntimeError, ValueError) as exc:
            report["runtime"] = {"kind": resolved["kind"], "path": resolved["path"]}
            report["parser"] = {"status": "not run", "reason": str(exc)}
        else:
            package_dir = Path(payload.get("gnn_module", "")).parent
            if resolved["kind"] == "checkout":
                package_dir = Path(resolved["package_dir"])
            mismatches = digest_mismatches(package_dir, pin)
            conformance = parser_conformance(payload, text)
            report["runtime"] = {"kind": resolved["kind"], "path": resolved["path"]}
            report["parser"] = {
                "status": "pass" if conformance["ok"] and not mismatches else "fail",
                "digest_mismatches": mismatches,
                **conformance,
            }

    parser_status = report["parser"]["status"]
    if not report["semantics"]["ok"] or parser_status == "fail":
        report["status"], report["exit_code"] = "fail", 1
    elif parser_status == "pass":
        report["status"], report["exit_code"] = "pass", 0
    else:
        report["status"], report["exit_code"] = "not run", 2
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the validation and print the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=gnn_spec.SPEC_PATH)
    parser.add_argument("--gnn-runtime", help="python with the pinned GNN distribution installed")
    parser.add_argument("--gnn-root", help="checkout of the pinned GNN revision")
    parser.add_argument("--gnn-python", help="interpreter to run the probe for --gnn-root")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)
    report = validate(args.spec, args.gnn_runtime, args.gnn_root, args.gnn_python)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"{report['status'].upper()}: {report['document']}")
        for difference in report["semantics"]["differences"]:
            print(f"  semantics: {difference}")
        if report["parser"]["status"] == "not run":
            print(f"  parser: not run ({report['parser']['reason']})")
        else:
            for difference in report["parser"]["differences"]:
                print(f"  parser: {difference}")
            for mismatch in report["parser"]["digest_mismatches"]:
                print(f"  parser: {mismatch['path']} {mismatch['reason']}")
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
