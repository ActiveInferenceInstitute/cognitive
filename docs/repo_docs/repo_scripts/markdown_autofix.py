#!/usr/bin/env python3
"""
markdown_autofix.py

Auto-fix low-risk Markdown formatting issues across a repository:
- Remove trailing spaces (outside fenced code blocks)
- Collapse multiple consecutive blank lines to a single blank line (outside fenced code blocks)
- Ensure blank lines around ATX headings (# ...)
- Ensure blank lines around fenced code blocks (``` or ~~~)
- Ensure blank lines around top-level lists (before first list item and after list block)
- Ensure a single trailing newline at EOF

Deliberately does NOT auto-fix riskier style rules (line length, heading style, duplicate headings, etc.).

Usage:
  python3 docs/repo_docs/repo_scripts/markdown_autofix.py --path . [--dry-run]

Notes:
- Skips common generated/virtual dirs by default.
- Only processes files ending with .md (case-insensitive).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import List


IGNORE_DIRS = {
    "node_modules",
    ".git",
    ".obsidian",
    "venv",
    ".venv",
    "env",
    "__pycache__",
    "Output",
    "output",
    "outputs",
}


HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+.+$")
LIST_PATTERN_BULLET = re.compile(r"^\s{0,3}([*+-])\s+.+$")
LIST_PATTERN_ORDERED = re.compile(r"^\s{0,3}\d+\.\s+.+$")


def is_markdown_file(path: str) -> bool:
    lower = path.lower()
    return lower.endswith(".md")


def is_code_fence_line(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith("```") or stripped.startswith("~~~")


def strip_trailing_spaces(line: str) -> str:
    # Preserve newline if present
    if line.endswith("\n"):
        return line.rstrip(" \t\r\x0b\x0c") + "\n"
    return line.rstrip(" \t\r\x0b\x0c")


def is_blank(line: str) -> bool:
    return len(line.strip()) == 0


def is_heading(line: str) -> bool:
    return bool(HEADING_PATTERN.match(line))


def is_list_item(line: str) -> bool:
    return bool(LIST_PATTERN_BULLET.match(line) or LIST_PATTERN_ORDERED.match(line))


def process_lines(lines: List[str]) -> List[str]:
    """Apply low-risk fixes to a list of markdown lines."""
    out: List[str] = []

    in_code_fence = False
    last_emitted_blank = False
    in_list_block = False

    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]

        # Code fence toggling (before any other transformation that depends on in_code_fence)
        fence_line = is_code_fence_line(line)
        if fence_line:
            stripped = line.lstrip()
            is_backticks = stripped.startswith("```")
            fence_token = "```" if is_backticks else "~~~"

            # Ensure blank line before opening fence (when not already at start or preceded by blank)
            if not in_code_fence:
                if len(out) > 0 and not last_emitted_blank:
                    out.append("\n")
                    last_emitted_blank = True

                # Opening fence: if no language specified, add a safe default 'text' to satisfy MD040
                info_string = stripped[len(fence_token):].strip()
                if info_string == "":
                    # Replace only the first occurrence to preserve indentation
                    line = line.replace(fence_token, fence_token + "text", 1)

            # Emit fence line (strip trailing spaces on the fence itself)
            out.append(strip_trailing_spaces(line))
            last_emitted_blank = False
            # Toggle fence state
            in_code_fence = not in_code_fence

            # If we just closed a fence, ensure a blank line after it (unless EOF or already blank)
            if not in_code_fence:
                # Look ahead to next line (original)
                if i + 1 < n and not is_blank(lines[i + 1]):
                    out.append("\n")
                    last_emitted_blank = True
            i += 1
            continue

        if in_code_fence:
            # Inside code blocks, do minimal changes: preserve content, but strip trailing spaces
            out.append(strip_trailing_spaces(line))
            last_emitted_blank = is_blank(line)
            i += 1
            continue

        # Outside code fences: strip trailing spaces
        line = strip_trailing_spaces(line)

        # Collapse multiple blank lines
        if is_blank(line):
            if not last_emitted_blank:
                out.append("\n")
                last_emitted_blank = True
            # Blank lines end list blocks
            in_list_block = False
            i += 1
            continue

        # Ensure blank line before headings (not at file start)
        if is_heading(line):
            if len(out) > 0 and not last_emitted_blank:
                out.append("\n")
            # Emit heading
            out.append(line)
            last_emitted_blank = False
            # Ensure blank line after heading (unless next line is already blank or EOF)
            if i + 1 < n and not is_blank(lines[i + 1]) and not is_code_fence_line(lines[i + 1]):
                out.append("\n")
                last_emitted_blank = True
            i += 1
            # A heading ends any list context
            in_list_block = False
            continue

        # Handle lists: ensure blank line before first list item of a block, and after the block
        if is_list_item(line):
            if not in_list_block:
                # Starting a new list block
                if len(out) > 0 and not last_emitted_blank:
                    out.append("\n")
                in_list_block = True
            out.append(line)
            last_emitted_blank = False

            # Look ahead: if next line ends the list block and is not blank, ensure a blank line after
            if i + 1 < n:
                next_line = lines[i + 1]
                if not is_blank(next_line) and not is_list_item(next_line):
                    # Insert blank line to separate list from following paragraph/section
                    out.append("\n")
                    last_emitted_blank = True
                    in_list_block = False
            i += 1
            continue

        # Any non-blank, non-heading, non-list, non-code line
        out.append(line)
        last_emitted_blank = False
        # If we were in a list and encounter normal text without a separating blank, ensure separation
        if in_list_block:
            # If previous emitted line was not blank, insert one before continuing content
            if not last_emitted_blank and len(out) > 0:
                # Ensure previous separation already added above; keep state consistent
                in_list_block = False
        i += 1

    # Ensure single trailing newline at EOF
    if len(out) == 0 or not out[-1].endswith("\n"):
        out.append("\n")

    # Collapse any accidental trailing multiple newlines at EOF to exactly one
    while len(out) >= 2 and out[-1] == "\n" and out[-2] == "\n":
        out.pop()

    return out


def process_file(path: str, dry_run: bool = False) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except (OSError, UnicodeDecodeError):
        return False

    new_lines = process_lines(lines)
    changed = new_lines != lines

    if changed and not dry_run:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.writelines(new_lines)

    return changed


def should_ignore_dir(dirname: str) -> bool:
    base = os.path.basename(dirname.rstrip(os.sep))
    return base in IGNORE_DIRS


def walk_and_process(root: str, dry_run: bool = False) -> int:
    changed_count = 0
    for current_root, dirs, files in os.walk(root):
        # Prune ignored directories in-place
        dirs[:] = [d for d in dirs if not should_ignore_dir(os.path.join(current_root, d))]
        for filename in files:
            path = os.path.join(current_root, filename)
            if not is_markdown_file(path):
                continue
            if process_file(path, dry_run=dry_run):
                changed_count += 1
                print(f"fixed: {os.path.relpath(path, root)}")
    return changed_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-fix low-risk Markdown formatting issues.")
    parser.add_argument("--path", default=".", help="Root directory to process (default: .)")
    parser.add_argument("--dry-run", action="store_true", help="Do not write changes; just report would-change files")
    args = parser.parse_args()

    root = os.path.abspath(args.path)
    changed = walk_and_process(root, dry_run=args.dry_run)
    if args.dry_run:
        print(f"Would fix {changed} files.")
    else:
        print(f"Fixed {changed} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


