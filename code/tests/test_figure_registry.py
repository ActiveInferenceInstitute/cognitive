"""Validate generated figure assets and their manuscript associations."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.build_manuscript import _validate_figure_registry

SOURCE = (
    "![First generated figure.](figures/first.png){#fig:first width=85%}\n\n"
    "![Second generated figure.](figures/second.png){#fig:second}\n\n"
    "See [@fig:first] and [@fig:second].\n"
)


def _figure_build(tmp_path: Path) -> tuple[Path, list[dict[str, str]], Path]:
    output = tmp_path / "manuscript"
    (output / "figures").mkdir(parents=True)
    registry = [
        {"label": f"fig:{name}", "file": f"figures/{name}.png", "source": "test"}
        for name in ("first", "second")
    ]
    for entry in registry:
        (output / entry["file"]).write_bytes(b"nonempty generated asset\n")
    combined = output / "combined.md"
    combined.write_text(SOURCE, encoding="utf-8")
    return output, registry, combined


def test_valid_registry_matches_manuscript_image_declarations(tmp_path: Path) -> None:
    output, registry, combined = _figure_build(tmp_path)
    _validate_figure_registry(output, registry, combined)


def test_swapped_label_to_image_associations_are_rejected(tmp_path: Path) -> None:
    output, registry, combined = _figure_build(tmp_path)
    registry[0]["file"], registry[1]["file"] = registry[1]["file"], registry[0]["file"]
    with pytest.raises(ValueError, match="figure label/path pairs do not match registry"):
        _validate_figure_registry(output, registry, combined)


@pytest.mark.parametrize("asset_state", ["missing", "empty", "symlink", "directory"])
def test_registered_assets_must_be_regular_nonempty_files(tmp_path: Path, asset_state: str) -> None:
    output, registry, combined = _figure_build(tmp_path)
    asset = output / registry[0]["file"]
    asset.unlink()
    if asset_state == "empty":
        asset.touch()
    elif asset_state == "symlink":
        asset.symlink_to(output / registry[1]["file"])
    elif asset_state == "directory":
        asset.mkdir()
    with pytest.raises(ValueError, match="regular nonempty file"):
        _validate_figure_registry(output, registry, combined)


@pytest.mark.parametrize("field", ["label", "file"])
def test_duplicate_registry_labels_or_paths_are_rejected(tmp_path: Path, field: str) -> None:
    output, registry, combined = _figure_build(tmp_path)
    registry[1][field] = registry[0][field]
    with pytest.raises(ValueError, match="Duplicate figure registry entry"):
        _validate_figure_registry(output, registry, combined)


def test_duplicate_manuscript_figure_labels_are_rejected(tmp_path: Path) -> None:
    output, registry, combined = _figure_build(tmp_path)
    combined.write_text(SOURCE.replace("#fig:second", "#fig:first"), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate manuscript figure labels"):
        _validate_figure_registry(output, registry, combined)


def test_unregistered_manuscript_figure_is_rejected(tmp_path: Path) -> None:
    output, registry, combined = _figure_build(tmp_path)
    (output / "figures" / "third.png").write_bytes(b"unregistered asset\n")
    combined.write_text(
        SOURCE + "\n![Third figure.](figures/third.png){#fig:third}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="figure labels do not match registry"):
        _validate_figure_registry(output, registry, combined)


def test_unregistered_manuscript_image_path_is_rejected(tmp_path: Path) -> None:
    output, registry, combined = _figure_build(tmp_path)
    combined.write_text(
        SOURCE.replace("figures/first.png", "figures/unknown.png"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="figure paths do not match registry"):
        _validate_figure_registry(output, registry, combined)


def test_registered_figure_without_an_image_declaration_is_rejected(tmp_path: Path) -> None:
    output, registry, combined = _figure_build(tmp_path)
    combined.write_text(
        SOURCE.replace(
            "![First generated figure.](figures/first.png){#fig:first width=85%}",
            "[First figure link.](figures/first.png){#fig:first width=85%}",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="figure label/path pairs do not match registry"):
        _validate_figure_registry(output, registry, combined)


def test_unresolved_manuscript_figure_reference_is_rejected(tmp_path: Path) -> None:
    output, registry, combined = _figure_build(tmp_path)
    combined.write_text(SOURCE + "See [@fig:missing].\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unresolved manuscript figure references"):
        _validate_figure_registry(output, registry, combined)
