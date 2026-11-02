"""Read the committed Generalized Notation Notation declaration of the model.

The canonical discrete model is declared in ``specs/discrete_variational.gnn.md``
and configured for the runtime in
``config_examples/discrete_variational.yaml``. This module reads the declaration
with the standard library and NumPy only, so the tests and the validation script
can compare the two without importing the upstream parser. The upstream parser
itself runs only in ``code/tests/gnn_parser_probe.py``, under the revision pinned
in ``specs/gnn-pin.json``.

See ``docs/gnn.md`` for the commands and for what each check proves.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

MODEL_NAME = "Cognitive Discrete Variational Model"
SPEC_RELATIVE_PATH = Path("specs/discrete_variational.gnn.md")
PIN_RELATIVE_PATH = Path("specs/gnn-pin.json")

#: Sections the committed document must declare, in the order the format uses.
REQUIRED_SECTIONS = (
    "GNNSection",
    "GNNVersionAndFlags",
    "ModelName",
    "ModelAnnotation",
    "StateSpaceBlock",
    "Connections",
    "InitialParameterization",
    "Equations",
    "Time",
    "ActInfOntologyAnnotation",
    "ModelParameters",
    "Footer",
)

#: Variables whose declarations are validated as arrays.
MATRIX_KEYS = ("A", "B", "C", "D", "E")


def _find_root(start: Path) -> Path:
    """Return the checkout root holding ``start`` or the working directory."""
    for base in (start, Path.cwd().resolve()):
        for candidate in (base, *base.parents):
            if (candidate / SPEC_RELATIVE_PATH).is_file():
                return candidate
    raise FileNotFoundError(
        f"no checkout containing {SPEC_RELATIVE_PATH.as_posix()} above {start} or {Path.cwd()}"
    )


ROOT = _find_root(Path(__file__).resolve())
SPEC_PATH = ROOT / SPEC_RELATIVE_PATH
PIN_PATH = ROOT / PIN_RELATIVE_PATH


def read_spec_text(path: Path | None = None) -> str:
    """Return the text of the declaration (the committed one by default)."""
    return (path or SPEC_PATH).read_text(encoding="utf-8")


def parse_sections(text: str | None = None) -> dict[str, str]:
    """Split the document into its ``## Name`` sections."""
    body = read_spec_text() if text is None else text
    sections: dict[str, str] = {}
    current: str | None = None
    lines: list[str] = []
    for line in body.splitlines():
        if line.startswith("## "):
            if current is not None:
                sections[current] = "\n".join(lines).strip()
            current = line[3:].strip()
            lines = []
        elif current is not None:
            lines.append(line)
    if current is not None:
        sections[current] = "\n".join(lines).strip()
    return sections


def missing_sections(text: str | None = None) -> list[str]:
    """Return the required sections the document does not declare."""
    declared = parse_sections(text)
    return [name for name in REQUIRED_SECTIONS if name not in declared]


def parse_variables(text: str | None = None) -> dict[str, tuple[str, ...]]:
    """Return the declared dimensions of every ``StateSpaceBlock`` variable."""
    sections = parse_sections(text)
    block = sections.get("StateSpaceBlock", "")
    variables: dict[str, tuple[str, ...]] = {}
    for line in block.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped or "[" not in stripped or not stripped.endswith("]"):
            continue
        name, _, rest = stripped.partition("[")
        tokens = [token.strip() for token in rest[:-1].split(",")]
        variables[name.strip()] = tuple(token for token in tokens if not token.startswith("type="))
    return variables


def matrix_dimensions(text: str | None = None) -> dict[str, tuple[int, ...]]:
    """Return the declared dimensions of the model's matrix variables."""
    variables = parse_variables(text)
    dimensions: dict[str, tuple[int, ...]] = {}
    for key in MATRIX_KEYS:
        if key not in variables:
            raise ValueError(f"the declaration does not declare {key}")
        try:
            dimensions[key] = tuple(int(token) for token in variables[key])
        except ValueError as exc:
            raise ValueError(
                f"{key} does not declare numeric dimensions: {variables[key]}"
            ) from exc
    return dimensions


def _braced_body(source: str, start: int) -> str:
    """Return the text between the brace at ``start`` and its match."""
    depth = 0
    for index in range(start, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start + 1 : index]
    raise ValueError("unbalanced braces in the parameterization")


def parse_parameterization(text: str | None = None) -> dict[str, Any]:
    """Return the ``InitialParameterization`` entries as nested lists."""
    sections = parse_sections(text)
    block = sections.get("InitialParameterization", "")
    values: dict[str, Any] = {}
    for match in re.finditer(r"^([A-Za-z_]\w*)=\{", block, flags=re.MULTILINE):
        body = _braced_body(block, match.end() - 1)
        values[match.group(1)] = ast.literal_eval("[" + body + "]")
    return values


def declared_matrices(text: str | None = None) -> dict[str, np.ndarray]:
    """Return the declared matrices, reshaped to their declared dimensions."""
    values = parse_parameterization(text)
    dimensions = matrix_dimensions(text)
    matrices: dict[str, np.ndarray] = {}
    for key in MATRIX_KEYS:
        if key not in values:
            raise ValueError(f"the parameterization does not declare {key}")
        array = np.asarray(values[key], dtype=float)
        try:
            matrices[key] = array.reshape(dimensions[key])
        except ValueError as exc:
            raise ValueError(
                f"{key} declares {array.size} values for shape {dimensions[key]}"
            ) from exc
    return matrices


def parse_model_parameters(text: str | None = None) -> dict[str, str]:
    """Return the ``ModelParameters`` entries as a mapping of strings."""
    sections = parse_sections(text)
    block = sections.get("ModelParameters", "")
    parameters: dict[str, str] = {}
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        name, _, value = stripped.partition(":")
        parameters[name.strip()] = value.strip()
    return parameters


def parse_ontology(text: str | None = None) -> dict[str, str]:
    """Return the ``ActInfOntologyAnnotation`` mapping of variables to terms."""
    sections = parse_sections(text)
    block = sections.get("ActInfOntologyAnnotation", "")
    annotations: dict[str, str] = {}
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, term = stripped.partition("=")
        annotations[name.strip()] = term.strip()
    return annotations


def parse_time(text: str | None = None) -> dict[str, str]:
    """Return the ``Time`` entries; a bare line maps its own name."""
    sections = parse_sections(text)
    block = sections.get("Time", "")
    entries: dict[str, str] = {}
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name, sep, value = stripped.partition("=")
        entries[name.strip()] = value.strip() if sep else stripped
    return entries


def load_pin(path: Path | None = None) -> dict[str, Any]:
    """Return the recorded upstream parser revision."""
    return json.loads((path or PIN_PATH).read_text(encoding="utf-8"))


def spec_summary(text: str | None = None) -> dict[str, Any]:
    """Return one dictionary describing the declaration."""
    body = read_spec_text() if text is None else text
    sections = parse_sections(body)
    matrices = declared_matrices(body)
    return {
        "model_name": sections.get("ModelName", "").strip(),
        "version": sections.get("GNNVersionAndFlags", "").strip(),
        "sections": list(sections),
        "missing_sections": missing_sections(body),
        "variables": parse_variables(body),
        "matrix_dimensions": {key: list(value) for key, value in matrix_dimensions(body).items()},
        "matrix_shapes": {key: list(value.shape) for key, value in matrices.items()},
        "ontology": parse_ontology(body),
        "parameters": parse_model_parameters(body),
        "time": parse_time(body),
    }
