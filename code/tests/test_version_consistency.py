"""The declared version must agree everywhere it is published.

A package whose metadata version and runtime ``__version__`` disagree reports
one number in the build manifest and another to ``pip``, and neither is
obviously the wrong one.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import cognitive

if sys.version_info >= (3, 11):
    import tomllib
else:  # the project supports 3.10, which has no tomllib
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_runtime_version_matches_project_metadata() -> None:
    metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert cognitive.__version__ == metadata["project"]["version"]


def test_changelog_documents_the_current_version() -> None:
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    headings = re.findall(r"^## \[([^\]]+)\]", changelog, flags=re.MULTILINE)
    assert headings, "the changelog must contain at least one version heading"
    assert cognitive.__version__ in headings, (
        f"version {cognitive.__version__} has no CHANGELOG entry; found {headings}"
    )
