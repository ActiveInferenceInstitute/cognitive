"""Validate tracked documentation, examples, links, and public exports."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from scripts.check_markdown_links import verify_markdown_link_report
from scripts.export_rdf import stale_docs as stale_rdf_docs
from scripts.kb_index import stale_catalogs
from scripts.markdown_fences import iter_fenced_blocks
from scripts.normalize_frontmatter import unquoted_links
from scripts.verify_links import verify_link_report

TERM_POLICY_PATH = Path("docs/policy/documentation_terms.yaml")


def load_term_policy(root: Path) -> tuple[list[str], set[str]]:
    """Return the forbidden terms and the paths exempt from the scan.

    The list lives in ``docs/policy/documentation_terms.yaml`` rather than in
    this module so that the policy is readable without reading Python, and so
    that the files which must discuss the terms can be exempted by name instead
    of spelling each term as two concatenated string literals to dodge the scan
    they implement.
    """
    import yaml

    policy_path = root / TERM_POLICY_PATH
    if not policy_path.is_file():
        raise FileNotFoundError(f"Missing documentation term policy: {policy_path}")
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}
    if not isinstance(policy, dict):
        raise ValueError(f"{TERM_POLICY_PATH} must contain a mapping")
    terms = policy.get("terms")
    exempt = policy.get("exempt_paths", [])
    if not isinstance(terms, list) or not terms or not all(isinstance(t, str) for t in terms):
        raise ValueError(f"{TERM_POLICY_PATH} must define a non-empty list of terms")
    if not isinstance(exempt, list) or not all(isinstance(path, str) for path in exempt):
        raise ValueError(f"{TERM_POLICY_PATH} exempt_paths must be a list of paths")
    return [term.strip() for term in terms], set(exempt)


def tracked_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return [
        root / item
        for item in result.stdout.decode().split("\0")
        if item and (root / item).is_file()
    ]


def _text_files(paths: list[Path]) -> list[Path]:
    text_extensions = {".md", ".py", ".yaml", ".yml", ".toml", ".txt", ".json", ".sh"}
    return [path for path in paths if path.suffix.lower() in text_extensions]


def _validate_frontmatter(path: Path, content: str) -> list[str]:
    if path.suffix.lower() != ".md" or not content.startswith("---"):
        return []
    parts = content.split("---", 2)
    if len(parts) != 3 or not parts[1].strip():
        return [f"{path}: malformed YAML frontmatter"]
    try:
        import yaml

        parsed = yaml.safe_load(parts[1])
    except Exception as exc:  # pragma: no cover - reports parser details
        return [f"{path}: invalid YAML frontmatter: {exc}"]
    if not isinstance(parsed, dict):
        return [f"{path}: frontmatter must be a mapping"]
    template_values = {"unique_identifier", "timestamp", "lorem_ipsum"}
    for key, value in parsed.items():
        if isinstance(value, str) and value.strip().lower() in template_values:
            return [f"{path}: template frontmatter value for '{key}': '{value.strip()}'"]
    # An unquoted [[link]] parses as a nested list, so the relation is data but
    # no longer a link Obsidian can see. normalize_frontmatter.py fixes it.
    unquoted = unquoted_links(parsed)
    if unquoted:
        return [
            f"{path}: semantic_relations links must be quoted wiki links "
            f"(run normalize_frontmatter.py --write): {unquoted[0]}"
        ]
    return []


def _validate_python_blocks(path: Path, content: str) -> list[str]:
    """Compile each unindented, closed ``python`` fence.

    Fences nested in list items are routinely schematic signatures, and a fence
    that is never closed has no known end, so neither is compiled. A fence that
    opens inside a longer fence is example text, not a block, which is why the
    blocks come from the fence tracker and not a pattern.
    """
    errors: list[str] = []
    for block in iter_fenced_blocks(content):
        language = block.info.lower().split()[0:1]
        if language not in [["python"], ["py"]] or block.indent or not block.terminated:
            continue
        try:
            compile(block.body, str(path), "exec")
        except SyntaxError as exc:
            errors.append(f"{path}: Python example line {exc.lineno}: {exc.msg}")
    return errors


def unterminated_fences(path: Path, content: str) -> list[str]:
    """Describe each code fence that is never closed.

    CommonMark renders everything after such a fence as code, so the rest of
    the page is lost to readers and to the link and anchor checks. ``validate``
    reports each one both here and as a documentation error.
    """
    return [
        f"{path}: code fence opened at line {block.start_line} is never closed"
        for block in iter_fenced_blocks(content)
        if not block.terminated
    ]


def _validate_exports() -> list[str]:
    try:
        import cognitive
    except ImportError as exc:
        return [f"public package import failed: {exc}"]
    exported = getattr(cognitive, "__all__", ())
    return [f"public export missing: {name}" for name in exported if not hasattr(cognitive, name)]


def _validate_manuscript(root: Path) -> dict[str, Any]:
    """Validate manuscript structure, labels, citations, and figure references."""
    manuscript = root / "docs" / "manuscript"
    required = {
        "00_abstract.md",
        "01_introduction.md",
        "02_methodology.md",
        "03_results.md",
        "04_conclusion.md",
        "05_experimental_setup.md",
        "06_reproducibility.md",
        "07_scope_and_related_work.md",
        "08_appendix_formalism.md",
        "09_appendix_validation.md",
        "99_references.md",
    }
    errors: list[str] = []
    if not manuscript.is_dir():
        return {"ok": True, "sections": 0, "labels": 0, "citations": 0, "figures": 0, "errors": []}
    actual = {path.name for path in manuscript.glob("[0-9][0-9]_*.md")}
    missing = required.difference(actual)
    if missing:
        errors.append(f"Manuscript sections missing: {sorted(missing)}")
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(manuscript.glob("[0-9][0-9]_*.md"))
    )
    labels = re.findall(r"\{#((?:eq|fig|tbl|sec):[A-Za-z0-9_-]+)(?:\s[^}]*)?\}", source)
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        errors.append(f"Duplicate manuscript labels: {duplicates}")
    references = re.findall(r"\[@((?:eq|fig|tbl|sec):[A-Za-z0-9_-]+)\]", source)
    missing_references = sorted(set(references).difference(labels))
    if missing_references:
        errors.append(f"Unresolved manuscript cross-references: {missing_references}")
    citation_keys = set(re.findall(r"(?<![:\w])@([a-z][a-z0-9_]+)(?![a-z0-9_]*:)", source))
    bib = (
        (manuscript / "references.bib").read_text(encoding="utf-8")
        if (manuscript / "references.bib").is_file()
        else ""
    )
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    missing_citations = sorted(citation_keys.difference(bib_keys))
    if missing_citations:
        errors.append(f"Citation keys missing from references.bib: {missing_citations}")
    figure_paths = sorted(set(re.findall(r"figures/([A-Za-z0-9_-]+\.png)", source)))
    builder = (root / "code" / "scripts" / "build_manuscript.py").read_text(encoding="utf-8")
    missing_figures = [path for path in figure_paths if path not in builder]
    if missing_figures:
        errors.append(f"Figure paths are not generated by the builder: {missing_figures}")
    try:
        import yaml

        config = yaml.safe_load((manuscript / "config.yaml").read_text(encoding="utf-8"))
        if not isinstance(config, dict) or not isinstance(config.get("experiment"), dict):
            errors.append("Manuscript config.yaml must contain an experiment mapping")
    except Exception as exc:  # pragma: no cover - reports parser details
        errors.append(f"Invalid manuscript config.yaml: {exc}")
    return {
        "ok": not errors,
        "sections": len(actual),
        "labels": len(labels),
        "citations": len(citation_keys),
        "figures": len(figure_paths),
        "errors": errors,
    }


def validate(root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root).resolve()
    files = tracked_files(root_path)
    errors: list[str] = []
    forbidden: list[dict[str, str]] = []
    unclosed: list[str] = []
    terms, exempt_paths = load_term_policy(root_path)
    for path in _text_files(files):
        relative = path.relative_to(root_path).as_posix()
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            # Skipping would exempt the file from every check below.
            errors.append(f"{relative}: not valid UTF-8, so it was not checked: {exc.reason}")
            continue
        if relative in exempt_paths:
            continue
        for term in terms:
            if re.search(rf"\b{re.escape(term)}\b", content, flags=re.IGNORECASE):
                forbidden.append({"path": relative, "term": term})
        errors.extend(_validate_frontmatter(path, content))
        errors.extend(_validate_python_blocks(path, content))
        if path.suffix.lower() == ".md":
            unclosed.extend(unterminated_fences(path.relative_to(root_path), content))
    errors.extend(unclosed)
    links = verify_link_report(root_path, strict_wiki_links=False)
    if links.broken_links:
        errors.append(f"{len(links.broken_links)} explicit wiki links do not resolve")
    for unreadable in links.unreadable_files:
        errors.append(f"{unreadable['source']}: wiki links not checked: {unreadable['error']}")
    markdown_links = verify_markdown_link_report(root_path)
    for unreadable in markdown_links.unreadable_files:
        errors.append(f"{unreadable['source']}: Markdown links not checked: {unreadable['error']}")
    if markdown_links.broken_links:
        errors.append(f"{len(markdown_links.broken_links)} standard Markdown links do not resolve")
    stale = stale_catalogs(root_path)
    if stale:
        errors.append(
            f"{len(stale)} knowledge-base catalog(s) are missing or stale "
            f"(run kb_index.py --write): {', '.join(stale[:3])}"
        )
    stale_rdf = stale_rdf_docs(root_path)
    if stale_rdf:
        errors.append(
            f"{len(stale_rdf)} OKN export file(s) are missing or stale "
            f"(run export_rdf.py --write-docs): {', '.join(stale_rdf)}"
        )
    errors.extend(_validate_exports())
    manuscript = _validate_manuscript(root_path)
    errors.extend(manuscript["errors"])
    return {
        "schema_version": 2,
        "tracked_text_files": len(_text_files(files)),
        "term_policy": {
            "path": TERM_POLICY_PATH.as_posix(),
            "terms": len(terms),
            "exempt_paths": sorted(exempt_paths),
        },
        "forbidden_terms": forbidden,
        "documentation_errors": errors,
        "unterminated_fences": unclosed,
        "links": {
            "checked": links.checked_links,
            "resolved": links.resolved_links,
            "skipped_concepts": links.skipped_concept_links,
            "broken": links.broken_links,
            "anchor_warnings": links.anchor_warnings,
        },
        "markdown_links": {
            "checked": markdown_links.checked_links,
            "broken": markdown_links.broken_links,
            "anchor_warnings": markdown_links.anchor_warnings,
        },
        "manuscript": manuscript,
        "ok": not forbidden and not errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate tracked documentation and runtime examples"
    )
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable report")
    args = parser.parse_args(argv)
    report = validate(args.root)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Checked {report['tracked_text_files']} tracked text files.")
        print(f"Checked {report['links']['checked']} wiki links.")
        print(f"Checked {report['markdown_links']['checked']} standard Markdown links.")
        print(f"Found {len(report['forbidden_terms'])} forbidden-term occurrences.")
        print(f"Found {len(report['documentation_errors'])} documentation errors.")
        print(f"Found {len(report['unterminated_fences'])} unterminated code fences.")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
