from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

try:
    from scripts.markdown_fences import strip_fenced_blocks
except ImportError:  # run as a file: the script directory is on sys.path
    from markdown_fences import strip_fenced_blocks  # type: ignore[no-redef]

WIKI_LINK_PATTERN = re.compile(r"\[\[([^\]]+)\]\]")
INLINE_CODE_PATTERN = re.compile(r"`[^`\n]+`")
MATRIX_LITERAL_PATTERN = re.compile(r"^[\d,\s\[\]\.]+$")

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

FILE_LIKE_SUFFIXES = {
    ".bib",
    ".csv",
    ".css",
    ".gif",
    ".htm",
    ".html",
    ".ipynb",
    ".jpeg",
    ".jpg",
    ".js",
    ".json",
    ".md",
    ".markdown",
    ".mp3",
    ".mp4",
    ".npy",
    ".npz",
    ".pdf",
    ".png",
    ".py",
    ".rst",
    ".sh",
    ".svg",
    ".tex",
    ".toml",
    ".ts",
    ".tsv",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}


@dataclass
class LinkReport:
    broken_links: list[dict[str, str]] = field(default_factory=list)
    checked_links: int = 0
    resolved_links: int = 0
    skipped_concept_links: int = 0
    indexed_markdown_files: int = 0
    anchor_warnings: list[dict[str, str]] = field(default_factory=list)
    # Markdown files that could not be read. Their links were not checked, so a
    # report with entries here is not a clean report.
    unreadable_files: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class LinkIndex:
    markdown_by_stem: dict[str, list[Path]]
    markdown_by_rel_no_suffix: dict[str, Path]
    all_files_by_rel: set[str]
    # Frontmatter ``aliases:`` of each page, so ``[[plasticity]]`` opens the page
    # that lists "plasticity" as an alias, as Obsidian does.
    markdown_by_alias: dict[str, Path] = field(default_factory=dict)


def _iter_files(root_dir: Path):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [name for name in dirnames if name not in IGNORED_DIRECTORIES]
        current_dir = Path(dirpath)
        for filename in filenames:
            yield current_dir / filename


def _strip_markdown_code(content: str) -> str:
    return INLINE_CODE_PATTERN.sub("", strip_fenced_blocks(content))


def _normalise_key(value: str) -> str:
    return value.replace("\\", "/").strip().lower()


def _alias_key(value: str) -> str:
    """Normalise a name so ``Free Energy``, ``free_energy`` and ``free-energy`` agree."""
    return " ".join(_normalise_key(value).replace("_", " ").replace("-", " ").split())


def _frontmatter_aliases(path: Path) -> list[str]:
    """Return the ``aliases:`` listed in a page's frontmatter, or an empty list."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []  # the link scan reports an unreadable file
    if not text.startswith("---"):
        return []
    parts = text.split("---", 2)
    if len(parts) < 3 or "aliases" not in parts[1]:
        return []
    try:
        data = yaml.safe_load(parts[1])
    except yaml.YAMLError:
        return []  # validate_docs reports malformed frontmatter
    aliases = data.get("aliases") if isinstance(data, dict) else None
    if isinstance(aliases, str):
        aliases = [aliases]
    if not isinstance(aliases, list):
        return []
    return [alias for alias in aliases if isinstance(alias, str)]


def _build_index(root_dir: Path) -> LinkIndex:
    markdown_by_stem: dict[str, list[Path]] = {}
    markdown_by_rel_no_suffix: dict[str, Path] = {}
    all_files_by_rel: set[str] = set()
    markdown_by_alias: dict[str, Path] = {}

    for path in _iter_files(root_dir):
        rel_path = path.relative_to(root_dir)
        rel_key = _normalise_key(rel_path.as_posix())
        all_files_by_rel.add(rel_key)

        if path.suffix.lower() != ".md":
            continue

        stem_key = _normalise_key(path.stem)
        markdown_by_stem.setdefault(stem_key, []).append(rel_path)
        markdown_by_rel_no_suffix[_normalise_key(rel_path.with_suffix("").as_posix())] = rel_path
        for alias in _frontmatter_aliases(path):
            markdown_by_alias.setdefault(_alias_key(alias), rel_path)

    return LinkIndex(
        markdown_by_stem=markdown_by_stem,
        markdown_by_rel_no_suffix=markdown_by_rel_no_suffix,
        all_files_by_rel=all_files_by_rel,
        markdown_by_alias=markdown_by_alias,
    )


def _extract_wiki_target(match: str) -> str:
    target = match.split("|", 1)[0].split("#", 1)[0]
    return target.strip()


def _extract_wiki_fragment(match: str) -> str | None:
    """Return the ``#fragment`` anchor of a wiki link, if present."""
    before_pipe = match.split("|", 1)[0]
    if "#" not in before_pipe:
        return None
    fragment = before_pipe.split("#", 1)[1].strip()
    return fragment or None


def _slugify(value: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", value).strip().lower()
    return re.sub(r"[\s_]+", "-", slug)


def _heading_slugs(filepath: Path) -> set[str]:
    """Slugs for Markdown headings in a file, GFM/Obsidian style."""
    slugs: set[str] = set()
    try:
        content = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return slugs
    # A "# comment" inside a fenced block is not a heading and cannot be an anchor.
    for line in strip_fenced_blocks(content).splitlines():
        stripped = line.strip()
        if not stripped.startswith("#") or stripped.startswith("#!"):
            continue
        heading = stripped.lstrip("#").strip()
        if not heading:
            continue
        slug = _slugify(heading)
        if slug:
            slugs.add(slug)
    return slugs


def _has_file_suffix(target: str) -> bool:
    suffix = Path(target).suffix.lower()
    return suffix in FILE_LIKE_SUFFIXES


def _is_explicit_file_reference(target: str) -> bool:
    if _has_file_suffix(target):
        return True
    # A slash is an explicit path signal even when the author omitted the
    # Markdown suffix, for example ``tools/README``.
    return target.endswith("/") or "/" in target or "\\" in target


def _candidate_paths(root_dir: Path, source_path: Path, target: str) -> list[Path]:
    current_dir = source_path.parent
    target = target.replace("\\", "/")

    base_candidates = [
        current_dir / target,
        root_dir / target,
    ]

    candidates: list[Path] = []
    for base in base_candidates:
        candidates.append(base)
        candidates.append(base / "README.md")
        candidates.append(base / "index.md")
        if not _has_file_suffix(target):
            candidates.append(base.with_suffix(".md"))

    return candidates


def _target_exists(
    root_dir: Path,
    source_path: Path,
    target: str,
    index: LinkIndex,
) -> bool:
    for candidate in _candidate_paths(root_dir, source_path, target):
        if candidate.is_file():
            return True

    rel_key = _normalise_key(target)
    if rel_key in index.all_files_by_rel:
        return True

    if _has_file_suffix(target):
        return False

    if rel_key in index.markdown_by_rel_no_suffix:
        return True

    basename_key = _normalise_key(Path(target).stem)
    return basename_key in index.markdown_by_stem or _alias_key(target) in index.markdown_by_alias


def _resolve_target_path(
    root_dir: Path,
    source_path: Path,
    target: str,
    index: LinkIndex,
) -> Path | None:
    """Resolve a wiki target to a file path, or ``None``.

    Mirrors :func:`_target_exists` but returns the resolved file so
    callers can inspect its headings for anchor validation.
    """
    seen: set[str] = set()
    for candidate in _candidate_paths(root_dir, source_path, target):
        key = _normalise_key(str(candidate))
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate

    rel_key = _normalise_key(target)
    if rel_key in index.all_files_by_rel:
        return root_dir / target

    # The index holds paths relative to the root; callers read the file, so join
    # them to the root rather than depend on the working directory.
    if rel_key in index.markdown_by_rel_no_suffix:
        return root_dir / index.markdown_by_rel_no_suffix[rel_key]

    if _has_file_suffix(target):
        return None

    basename_key = _normalise_key(Path(target).stem)
    matches = index.markdown_by_stem.get(basename_key, [])
    if matches:
        return root_dir / matches[0]
    aliased = index.markdown_by_alias.get(_alias_key(target))
    return root_dir / aliased if aliased is not None else None


def verify_link_report(
    root_dir: str | Path,
    *,
    strict_wiki_links: bool = False,
) -> LinkReport:
    root_path = Path(root_dir).resolve()
    index = _build_index(root_path)
    report = LinkReport(indexed_markdown_files=len(index.markdown_by_rel_no_suffix))

    for filepath in _iter_files(root_path):
        if filepath.suffix.lower() != ".md":
            continue

        try:
            content = filepath.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            report.unreadable_files.append(
                {"source": filepath.relative_to(root_path).as_posix(), "error": str(exc)}
            )
            continue

        content_no_code = _strip_markdown_code(content)
        for match in WIKI_LINK_PATTERN.findall(content_no_code):
            if MATRIX_LITERAL_PATTERN.match(match):
                continue

            target = _extract_wiki_target(match)
            if not target:
                continue

            report.checked_links += 1
            resolved = _resolve_target_path(root_path, filepath, target, index)
            if resolved is not None:
                report.resolved_links += 1
                fragment = _extract_wiki_fragment(match)
                if fragment:
                    slugs = _heading_slugs(resolved)
                    if _slugify(fragment) not in slugs:
                        report.anchor_warnings.append(
                            {
                                "source": filepath.relative_to(root_path).as_posix(),
                                "link": match,
                                "target": target,
                                "fragment": fragment,
                            }
                        )
                continue

            if not strict_wiki_links and not _is_explicit_file_reference(target):
                report.skipped_concept_links += 1
                continue

            report.broken_links.append(
                {
                    "source": filepath.relative_to(root_path).as_posix(),
                    "link": match,
                    "target": target,
                }
            )

    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Obsidian wiki links in Markdown files.",
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Repository or documentation root to scan.",
    )
    parser.add_argument(
        "--strict-wiki-links",
        action="store_true",
        help=(
            "Require every wiki link to resolve to a file. By default, "
            "unresolved extensionless wiki targets are treated as concept links."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the complete machine-readable report as JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = verify_link_report(
        args.root,
        strict_wiki_links=args.strict_wiki_links,
    )

    if args.json:
        import json

        print(
            json.dumps(
                {
                    "broken_links": report.broken_links,
                    "checked_links": report.checked_links,
                    "resolved_links": report.resolved_links,
                    "skipped_concept_links": report.skipped_concept_links,
                    "indexed_markdown_files": report.indexed_markdown_files,
                    "anchor_warnings": report.anchor_warnings,
                    "unreadable_files": report.unreadable_files,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return int(bool(report.broken_links or report.unreadable_files))

    print(f"Index built with {report.indexed_markdown_files} markdown files.")
    print(f"Checked {report.checked_links} wiki links.")
    print(f"Resolved {report.resolved_links} links.")
    if report.skipped_concept_links:
        print(
            "Skipped "
            f"{report.skipped_concept_links} unresolved concept/wiki links "
            "(use --strict-wiki-links to audit them)."
        )

    if report.anchor_warnings:
        print(f"Found {len(report.anchor_warnings)} anchor warnings:")
        for warning in report.anchor_warnings:
            print(f"  Source: {warning['source']}")
            print(f"  Link: [[{warning['link']}]]")
            print(f"  Fragment: #{warning['fragment']}")
            print("-" * 20)

    if report.unreadable_files:
        print(f"Could not read {len(report.unreadable_files)} files; their links were not checked:")
        for unreadable in report.unreadable_files:
            print(f"  Source: {unreadable['source']}")
            print(f"  Error: {unreadable['error']}")
            print("-" * 20)

    if report.broken_links:
        print(f"Found {len(report.broken_links)} broken file links:")
        for broken in report.broken_links:
            print(f"  Source: {broken['source']}")
            print(f"  Link: [[{broken['link']}]]")
            print(f"  Target: {broken['target']}")
            print("-" * 20)
        return 1
    if report.unreadable_files:
        return 1

    print("All file links verified successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
