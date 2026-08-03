"""Validate standard Markdown links and anchors.

Obsidian wiki links (``[[target]]``) are checked by ``verify_links.py``.
This script checks the other Markdown link family: ``[text](target)`` and
``![alt](target)`` references, including ``file.md#anchor`` fragments.

Rules
-----
- File targets must resolve relative to the document, or to the repository
  root.
- ``#anchor`` fragments are validated against GitHub-flavored heading slugs of
  the target file (reported as warnings: slug rules are renderer-dependent).
- The vendored RxInfer documentation tree
  (``docs/implementation/rxinfer/docs/``) is a Documenter.jl snapshot and is
  skipped; its ``@ref``/``@id`` directives are not Markdown links.
- Figure references under ``docs/manuscript/`` (``figures/*``) are generated
  by ``code/scripts/build_manuscript.py`` at build time and are skipped.

The report is also produced by ``validate_docs.py``, which treats broken file
links as documentation errors.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~")
INLINE_CODE_PATTERN = re.compile(r"`[^`\n]+`")
LATEX_PATTERN = re.compile(r"\$[^$\n]*\$")
LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+[\"'][^\"']*[\"'])?\)")

IGNORED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "node_modules",
    "Output",
    "output",
    "outputs",
    "venv",
}

# Documenter.jl snapshot: not scanned for standard Markdown links.
VENDORED_DOCS = ("docs", "implementation", "rxinfer", "docs")
# Manuscript figure references are produced by the build script.
MANUSCRIPT_DIR = Path("docs") / "manuscript"


@dataclass
class MarkdownLinkReport:
    checked_links: int = 0
    broken_links: list[dict[str, str]] = field(default_factory=list)
    anchor_warnings: list[dict[str, str]] = field(default_factory=list)
    indexed_markdown_files: int = 0


def _iter_markdown_files(root_dir: Path):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [name for name in dirnames if name not in IGNORED_DIRECTORIES]
        current_dir = Path(dirpath)
        if tuple(current_dir.relative_to(root_dir).parts) == VENDORED_DOCS:
            dirnames[:] = []
            continue
        for filename in filenames:
            if filename.lower().endswith(".md"):
                yield current_dir / filename


def _strip_code(content: str) -> str:
    without_blocks = CODE_BLOCK_PATTERN.sub("", content)
    return INLINE_CODE_PATTERN.sub("", without_blocks)


def gfm_slug(heading: str) -> str:
    """GitHub-flavored heading slug (lowercase; keep word chars and hyphens)."""
    slug = heading.strip().lower()
    slug = re.sub(r"[^\w\s-]", "", slug, flags=re.UNICODE)
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")


def _heading_slugs(path: Path) -> set[str]:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return set()
    slugs: set[str] = set()
    for line in content.splitlines():
        match = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if match:
            slugs.add(gfm_slug(match.group(2)))
    return slugs


def _is_external(target: str) -> bool:
    return target.startswith(
        ("http://", "https://", "mailto:", "tel:", "ftp://", "//")
    ) or target.startswith(("<", "@ref", "@id"))


def verify_markdown_link_report(root_dir: str | Path) -> MarkdownLinkReport:
    root_path = Path(root_dir).resolve()
    report = MarkdownLinkReport()
    files = list(_iter_markdown_files(root_path))
    report.indexed_markdown_files = len(files)

    for filepath in files:
        rel_source = filepath.relative_to(root_path).as_posix()
        try:
            content = filepath.read_text(encoding="utf-8")
        except OSError:
            continue
        content = LATEX_PATTERN.sub("", _strip_code(content))
        own_slugs = _heading_slugs(filepath)

        for match in LINK_PATTERN.finditer(content):
            target = match.group(1).strip()
            if not target or target.startswith("#"):
                if target and target != "#":
                    slug = target[1:]
                    if slug not in own_slugs:
                        report.anchor_warnings.append(
                            {
                                "source": rel_source,
                                "link": target,
                                "kind": "same-file anchor",
                                "detail": f"no heading with slug '{slug}'",
                            }
                        )
                continue
            if _is_external(target):
                continue
            if "\\" in target:
                continue  # LaTeX or escaped syntax, not a link
            if target.startswith("{{"):
                continue  # template syntax

            report.checked_links += 1

            # Allow generated manuscript figures.
            if rel_source.startswith(MANUSCRIPT_DIR.as_posix()) and target.startswith(
                "figures/"
            ):
                continue

            anchor = None
            if "#" in target:
                path_part, _, anchor = target.partition("#")
            else:
                path_part = target

            resolved = _resolve(path_part, filepath.parent, root_path)
            if resolved is None:
                report.broken_links.append(
                    {
                        "source": rel_source,
                        "link": match.group(0),
                        "target": target,
                        "kind": "file not found",
                        "detail": f"no such file '{path_part}'",
                    }
                )
                continue

            if anchor:
                target_slugs = _heading_slugs(resolved)
                if anchor not in target_slugs:
                    report.anchor_warnings.append(
                        {
                            "source": rel_source,
                            "link": match.group(0),
                            "target": target,
                            "kind": "anchor",
                            "detail": (
                                f"'{anchor}' is not a heading slug of "
                                f"{resolved.relative_to(root_path).as_posix()}"
                            ),
                        }
                    )

    return report


def _resolve(path_part: str, source_dir: Path, root_path: Path) -> Path | None:
    candidates = [source_dir / path_part, root_path / path_part]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate standard Markdown links and anchors.",
    )
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the complete machine-readable report as JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = verify_markdown_link_report(args.root)
    if args.json:
        print(
            json.dumps(
                {
                    "checked_links": report.checked_links,
                    "broken_links": report.broken_links,
                    "anchor_warnings": report.anchor_warnings,
                    "indexed_markdown_files": report.indexed_markdown_files,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return int(bool(report.broken_links))

    print(f"Indexed {report.indexed_markdown_files} markdown files.")
    print(f"Checked {report.checked_links} standard Markdown links.")
    if report.broken_links:
        print(f"Found {len(report.broken_links)} broken links:")
        for broken in report.broken_links:
            print(f"  Source: {broken['source']}")
            print(f"  Link: {broken['link']}")
            print(f"  Detail: {broken['detail']}")
            print("-" * 20)
        return 1
    if report.anchor_warnings:
        print(
            f"Found {len(report.anchor_warnings)} anchor warnings "
            "(heading-slug mismatches; renderer-dependent)."
        )
    print("All standard Markdown file links verified successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
