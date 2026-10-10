"""Keep ``semantic_relations`` links in frontmatter as quoted wiki-link strings.

Obsidian reads a frontmatter link only when it is a quoted string such as
``"[[active_inference]]"``. An unquoted ``[[active_inference]]`` is parsed by
YAML as a nested list, so the relation survives as data but stops being a link:
it is invisible to the graph view, backlinks, and rename tracking. This module
rewrites every ``semantic_relations`` block into the quoted form and, in
``--check`` mode, reports the files that have drifted back.

Only the ``semantic_relations`` block is touched; every other frontmatter key
and the document body are preserved byte for byte.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

_KEY = "semantic_relations"
_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---(?=\n|\Z)", re.DOTALL)
_TOP_LEVEL_KEY = re.compile(r"^[^\s#-]")


def _flatten(value: Any) -> list[str]:
    """Flatten arbitrarily nested YAML lists into a flat list of link targets."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        flat: list[str] = []
        for item in value:
            flat.extend(_flatten(item))
        return flat
    return [str(value).strip()]


def wiki_link(target: str) -> str:
    """Return ``target`` as a wiki link, leaving an already-wrapped link alone."""
    if target.startswith("[[") and target.endswith("]]"):
        return target
    return f"[[{target}]]"


def normalized_relations(relations: Any) -> list[dict[str, Any]]:
    """Return the relation list with each ``links`` value as flat wiki-link strings."""
    if not isinstance(relations, list):
        return []
    result: list[dict[str, Any]] = []
    for entry in relations:
        if not isinstance(entry, dict):
            continue
        normalized: dict[str, Any] = {}
        for key, value in entry.items():
            if key == "links":
                normalized[key] = [wiki_link(t) for t in _flatten(value) if t]
            else:
                normalized[key] = value
        result.append(normalized)
    return result


def _scalar(value: Any) -> str:
    """Render a scalar so that ``yaml.safe_load`` reads back the same value."""
    return yaml.safe_dump(value, default_flow_style=True, allow_unicode=True).splitlines()[0]


def render_block(relations: list[dict[str, Any]]) -> str:
    """Render the ``semantic_relations`` block in the canonical indented style."""
    if not relations:
        return f"{_KEY}: []"
    lines = [f"{_KEY}:"]
    for entry in relations:
        first = True
        for key, value in entry.items():
            prefix = "  - " if first else "    "
            first = False
            if key == "links":
                if value:
                    lines.append(f"{prefix}links:")
                    lines.extend(
                        f"      - {json.dumps(link, ensure_ascii=False)}" for link in value
                    )
                else:
                    lines.append(f"{prefix}links: []")
            else:
                lines.append(f"{prefix}{key}: {_scalar(value)}")
    return "\n".join(lines)


def normalize_text(text: str) -> str:
    """Return ``text`` with its ``semantic_relations`` block normalized.

    Text without frontmatter, without the key, or whose frontmatter does not
    parse is returned unchanged; the validator reports the last case.
    """
    match = _FRONTMATTER.match(text)
    if not match:
        return text
    front = match.group(1)
    try:
        data = yaml.safe_load(front)
    except yaml.YAMLError:
        return text
    if not isinstance(data, dict) or _KEY not in data:
        return text
    lines = front.split("\n")
    start = next((i for i, line in enumerate(lines) if line.startswith(f"{_KEY}:")), None)
    if start is None:
        return text
    end = start + 1
    while end < len(lines) and not _TOP_LEVEL_KEY.match(lines[end]):
        end += 1
    block = render_block(normalized_relations(data[_KEY]))
    rebuilt = "\n".join(lines[:start] + block.split("\n") + lines[end:])
    return text[: match.start(1)] + rebuilt + text[match.end(1) :]


def unquoted_links(front: dict[str, Any]) -> list[str]:
    """Return the relation links in parsed frontmatter that are not quoted wiki links."""
    relations = front.get(_KEY)
    if relations is None:
        return []
    if not isinstance(relations, list):
        return [f"{_KEY} must be a list of relations"]
    problems: list[str] = []
    for entry in relations:
        if not isinstance(entry, dict):
            problems.append(f"relation is not a mapping: {entry!r}")
            continue
        # An empty ``links:`` key parses to None and means "no links"; any other
        # falsy value (0, "", false) is bad input, not an empty list.
        links = [] if entry.get("links") is None else entry["links"]
        if not isinstance(links, list):
            problems.append(f"links of '{entry.get('type')}' is not a list")
            continue
        for link in links:
            if not (isinstance(link, str) and link.startswith("[[") and link.endswith("]]")):
                problems.append(f"{entry.get('type')}: {link!r}")
    return problems


def tracked_markdown(root: Path) -> list[Path]:
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "*.md"],
        check=True,
        capture_output=True,
    ).stdout.decode()
    return [root / item for item in out.split("\0") if item and (root / item).is_file()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Normalize semantic_relations links to quoted wiki-link strings."
    )
    parser.add_argument("root", nargs="?", default=".")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="List drifted files; exit 1 if any")
    mode.add_argument("--write", action="store_true", help="Rewrite drifted files in place")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    drifted: list[Path] = []
    unreadable: list[Path] = []
    for path in tracked_markdown(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            unreadable.append(path)
            continue
        updated = normalize_text(text)
        if updated == text:
            continue
        drifted.append(path)
        if args.write:
            path.write_text(updated, encoding="utf-8")

    for path in drifted:
        print(path.relative_to(root).as_posix())
    verb = "rewrote" if args.write else "found"
    print(f"{verb} {len(drifted)} file(s) with unquoted semantic_relations links", file=sys.stderr)
    for path in unreadable:
        print(f"not valid UTF-8, not checked: {path.relative_to(root).as_posix()}", file=sys.stderr)
    return 1 if ((drifted and not args.write) or unreadable) else 0


if __name__ == "__main__":
    raise SystemExit(main())
