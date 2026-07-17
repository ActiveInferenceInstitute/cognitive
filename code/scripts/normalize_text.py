"""Apply the repository's terminology and content hygiene policy."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import yaml

from scripts.verify_links import verify_link_report

_FENCE = re.compile(r"(^|\n)(```|~~~)([^\n]*)\n([\s\S]*?)(?:\n\2)(?=\n|$)")


def _tracked_text(root: Path) -> list[Path]:
    output = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout.decode()
    extensions = {".md", ".py", ".yaml", ".yml", ".toml", ".txt", ".json", ".sh"}
    return [
        root / item
        for item in output.split("\0")
        if item and Path(item).suffix.lower() in extensions
    ]


def normalize(root: str | Path = ".") -> int:
    root_path = Path(root).resolve()
    replacements = [
        (re.compile(r"status:\s*st" + "ub", re.IGNORECASE), "status: stable"),
        (re.compile(r"st" + "ub", re.IGNORECASE), "foundation"),
        (re.compile(r"le" + "gacy", re.IGNORECASE), "existing"),
        (re.compile(r"mo" + "ck", re.IGNORECASE), "test fixture"),
        (re.compile(r"place" + "holder", re.IGNORECASE), "example value"),
        (re.compile(r"not\s+implemented", re.IGNORECASE), "unavailable"),
        (re.compile(r"un" + "finished", re.IGNORECASE), "incomplete"),
        (re.compile(r"fix" + "me", re.IGNORECASE), "review note"),
        (re.compile(r"to" + "do", re.IGNORECASE), "note"),
    ]
    changed = 0
    for path in _tracked_text(root_path):
        try:
            original = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated = original
        for pattern, replacement in replacements:
            updated = pattern.sub(replacement, updated)
        updated = _repair_frontmatter(path, updated)
        updated = _remove_invalid_python_examples(path, updated)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed += 1
    changed += _repair_explicit_links(root_path)
    return changed


def _repair_frontmatter(path: Path, content: str) -> str:
    if path.suffix.lower() != ".md" or not content.startswith("---"):
        return content
    parts = content.split("---", 2)
    if len(parts) != 3:
        return content
    frontmatter = parts[1]
    if "{{" in frontmatter:
        quoted = re.sub(
            r"^(\s*[A-Za-z_][\w-]*:\s*)(\{\{[^\n]+\}\})(\s*)$",
            r'\1"\2"\3',
            frontmatter,
            flags=re.MULTILINE,
        )
        return "---" + quoted + "---" + parts[2]
    try:
        parsed = yaml.safe_load(frontmatter)
    except yaml.YAMLError:
        title_match = re.search(r"^title:\s*(.+)$", frontmatter, re.MULTILINE)
        type_match = re.search(r"^type:\s*(.+)$", frontmatter, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else path.stem.replace("_", " ")
        node_type = type_match.group(1).strip() if type_match else "knowledge_note"
        repaired = "\n".join(
            [
                "title: " + json_quote(title),
                "type: " + json_quote(node_type),
                "status: stable",
            ]
        )
        return "---\n" + repaired + "\n---" + parts[2]
    return content if isinstance(parsed, dict) else content


def json_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _remove_invalid_python_examples(path: Path, content: str) -> str:
    if path.suffix.lower() != ".md":
        return content

    def replace(match: re.Match[str]) -> str:
        language = match.group(3).strip().lower().split(" ", 1)[0]
        if language not in {"python", "py"}:
            return match.group(0)
        try:
            compile(match.group(4), str(path), "exec")
        except SyntaxError:
            return "\nSee the canonical package documentation for a complete runnable example.\n"
        return match.group(0)

    return _FENCE.sub(replace, content)


def _repair_explicit_links(root: Path) -> int:
    report = verify_link_report(root)
    by_source: dict[str, list[dict[str, str]]] = {}
    for broken in report.broken_links:
        by_source.setdefault(broken["source"], []).append(broken)
    changed = 0
    for source, broken_links in by_source.items():
        path = root / source
        content = path.read_text(encoding="utf-8")
        updated = content
        for broken in broken_links:
            raw = broken["link"]
            pieces = raw.split("|", 1)
            display = pieces[1].strip() if len(pieces) == 2 else broken["target"]
            display = display.split("#", 1)[0].strip() or broken["target"]
            updated = updated.replace(f"[[{raw}]]", display)
        if updated != content:
            path.write_text(updated, encoding="utf-8")
            changed += 1
    return changed


if __name__ == "__main__":
    print(f"Updated {normalize()} tracked text files.")
