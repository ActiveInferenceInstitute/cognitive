"""Generate the knowledge-base catalogs that make every page easy to find.

The hub pages of each domain list only a fraction of its pages, so most pages
were reachable only by search or by following links from a neighbour. Each
domain under ``knowledge_base/`` now carries a generated ``catalog.md`` that
lists every page beneath it, grouped by sub-folder, with the page title and a
one-line description. ``knowledge_base/catalog.md`` lists the domains and the
pages that sit at the top level.

The catalogs are derived data. Edit a page's ``title``/``description``
frontmatter or its opening paragraph, then regenerate; never edit a catalog.
``--check`` fails when a catalog is missing or stale, and ``validate_docs`` runs
the same check so a new page cannot be added without appearing in its catalog.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

try:
    from scripts.markdown_fences import strip_fenced_blocks
except ImportError:  # run as a file: the script directory is on sys.path
    from markdown_fences import strip_fenced_blocks  # type: ignore[no-redef]

KB_NAME = "knowledge_base"
CATALOG = "catalog.md"
# Navigation and generated files are not catalogued: they are the catalogue.
_SKIP_STEMS = {"README", "AGENTS", "catalog"}
_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---[ \t]*\n?", re.DOTALL)
_BOILERPLATE = re.compile(r"This knowledge base entry is a foundation[^.]*\.\s*", re.IGNORECASE)
_MATH = re.compile(r"\\\(|\\\[|\$")
_MAX_DESCRIPTION = 180


@dataclass(frozen=True)
class Page:
    path: Path  # relative to the repository root
    title: str
    description: str

    @property
    def target(self) -> str:
        return self.path.with_suffix("").as_posix()


def _split(text: str) -> tuple[dict[str, object], str]:
    match = _FRONTMATTER.match(text)
    if not match:
        return {}, text
    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        data = None
    return (data if isinstance(data, dict) else {}), text[match.end() :]


def _plain(text: str) -> str:
    """Strip Markdown and wiki-link syntax down to readable prose."""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[\[([^\]|#]*)(?:#[^\]|]*)?\|([^\]]*)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]|#]*)(?:#[^\]]*)?\]\]", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*_`]+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _first_sentence(text: str) -> str:
    text = _BOILERPLATE.sub("", text).strip()
    match = re.search(r"(?<=[.!?])\s", text)
    sentence = text[: match.start()] if match else text
    if len(sentence) > _MAX_DESCRIPTION:
        sentence = sentence[: _MAX_DESCRIPTION - 1].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    return sentence


def _paragraphs(body: str) -> list[tuple[str | None, str]]:
    """Return ``(nearest heading, paragraph)`` pairs of plain prose paragraphs."""
    result: list[tuple[str | None, str]] = []
    heading: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            result.append((heading, " ".join(buffer)))
            buffer.clear()

    for line in strip_fenced_blocks(body, replacement="\n").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            flush()
            heading = stripped.lstrip("#").strip().lower()
            continue
        if not stripped:
            flush()
            continue
        if stripped.startswith(("-", "*", "+", "|", ">", "!", "<", "$", "1.", "---")):
            flush()
            continue
        buffer.append(stripped)
    flush()
    return result


def describe(front: dict[str, object], body: str) -> str:
    """One-line description: ``description`` frontmatter, else the opening prose."""
    declared = front.get("description")
    if isinstance(declared, str) and declared.strip():
        return _first_sentence(_plain(declared))
    paragraphs = _paragraphs(body)
    preferred = [p for h, p in paragraphs if h in {"overview", "summary", "introduction"}]
    # Only the page's opening can describe the page; later paragraphs describe
    # a subtopic and would mislabel it.
    openers = preferred[:1] + [p for _, p in paragraphs[:2]]
    sentences = [_first_sentence(_plain(c)) for c in openers]
    # A sentence that ends in a colon introduces a formula or list, and one that
    # carries inline math reads as noise in a one-line summary. A page with no
    # clean opening sentence is listed by title alone rather than misdescribed.
    for sentence in sentences:
        if len(sentence) >= 20 and not sentence.endswith(":") and not _MATH.search(sentence):
            return sentence
    return ""


def page_title(front: dict[str, object], body: str, stem: str) -> str:
    declared = front.get("title")
    if isinstance(declared, str) and declared.strip():
        title = declared.strip()
    else:
        heading = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
        title = _plain(heading.group(1)) if heading else stem.replace("_", " ").title()
    return re.sub(r"\s+", " ", re.sub(r"[|\[\]]", " ", title)).strip()


def load_page(root: Path, path: Path) -> Page:
    front, body = _split(path.read_text(encoding="utf-8"))
    return Page(path.relative_to(root), page_title(front, body, path.stem), describe(front, body))


def _pages_under(root: Path, folder: Path) -> list[Page]:
    pages = [
        load_page(root, path)
        for path in sorted(folder.rglob("*.md"))
        if path.stem not in _SKIP_STEMS and path.stat().st_size > 0  # empty file: not a page
    ]
    return pages


def _line(page: Page) -> str:
    entry = f"- [[{page.target}|{page.title}]]"
    return f"{entry} — {page.description}" if page.description else entry


def _frontmatter(title: str) -> str:
    return f'---\ntitle: "{title}"\ntype: catalog\nstatus: generated\n---\n'


def domain_catalog(root: Path, domain: Path) -> str:
    """Render the catalog of one domain folder."""
    pages = _pages_under(root, domain)
    groups: dict[str, list[Page]] = {}
    for page in pages:
        relative = page.path.relative_to(domain.relative_to(root)).parent
        groups.setdefault(relative.as_posix() if relative.parts else "", []).append(page)

    name = domain.name.replace("_", " ")
    lines = [
        _frontmatter(f"Catalog: {name}").rstrip("\n"),
        "",
        f"# Catalog: {name}",
        "",
        f"All {len(pages)} pages under `{domain.relative_to(root).as_posix()}/`, with the "
        "opening line of each. Generated by `code/scripts/kb_index.py` — regenerate, "
        "do not edit. Back to the [[" + f"{domain.relative_to(root).as_posix()}/README|"
        f"{name} overview]].",
    ]
    for group in sorted(groups, key=lambda g: (g != "", g.lower())):
        heading = f"`{group}/`" if group else "Top level"
        lines += ["", f"## {heading}", ""]
        entries = sorted(groups[group], key=lambda p: (p.title.casefold(), p.target))
        lines += [_line(page) for page in entries]
    return "\n".join(lines) + "\n"


def root_catalog(root: Path, domains: list[Path]) -> str:
    """Render ``knowledge_base/catalog.md``: the domains plus top-level pages."""
    kb = root / KB_NAME
    top = [
        load_page(root, path) for path in sorted(kb.glob("*.md")) if path.stem not in _SKIP_STEMS
    ]
    lines = [
        _frontmatter("Catalog: knowledge base").rstrip("\n"),
        "",
        "# Catalog: knowledge base",
        "",
        "Entry point to every domain catalog. Generated by `code/scripts/kb_index.py` — "
        "regenerate, do not edit.",
        "",
        "## Domains",
        "",
    ]
    for domain in domains:
        count = len(_pages_under(root, domain))
        rel = domain.relative_to(root).as_posix()
        lines.append(f"- [[{rel}/{CATALOG[:-3]}|{domain.name.replace('_', ' ')}]] — {count} pages")
    lines += ["", "## Top-level pages", ""]
    lines += [_line(page) for page in sorted(top, key=lambda p: (p.title.casefold(), p.target))]
    return "\n".join(lines) + "\n"


def expected_catalogs(root: Path) -> dict[Path, str]:
    """Map each catalog path to the content it should have."""
    kb = root / KB_NAME
    if not kb.is_dir():
        return {}
    domains = sorted(
        (d for d in kb.iterdir() if d.is_dir() and any(d.rglob("*.md"))),
        key=lambda d: d.name.lower(),
    )
    catalogs = {domain / CATALOG: domain_catalog(root, domain) for domain in domains}
    catalogs[kb / CATALOG] = root_catalog(root, domains)
    return catalogs


def stale_catalogs(root: Path) -> list[str]:
    """Return the repo-relative catalog paths that are missing or out of date."""
    stale: list[str] = []
    for path, content in expected_catalogs(root).items():
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            stale.append(path.relative_to(root).as_posix())
    return sorted(stale)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate knowledge-base catalogs.")
    parser.add_argument("root", nargs="?", default=".")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Exit 1 if any catalog is stale")
    mode.add_argument("--write", action="store_true", help="Regenerate stale catalogs")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    stale = stale_catalogs(root)
    if args.write:
        for path, content in expected_catalogs(root).items():
            if path.relative_to(root).as_posix() in stale:
                path.write_text(content, encoding="utf-8")
    for item in stale:
        print(item)
    verb = "regenerated" if args.write else "stale"
    print(f"{len(stale)} catalog(s) {verb}", file=sys.stderr)
    return 1 if (stale and not args.write) else 0


if __name__ == "__main__":
    raise SystemExit(main())
