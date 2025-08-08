#!/usr/bin/env python3
"""
markdown_autofix_low_risk.py

Purpose: Apply safe, low-risk Markdown normalizations repo-wide to reduce common lint issues.

Fixes included (outside fenced code blocks):
- MD009: remove trailing spaces
- MD012: collapse multiple blank lines
- MD022: ensure blank lines around ATX headings
- MD018/MD019: normalize ATX heading spacing (one space after #, no left indent)
- MD023: ensure ATX headings are left-aligned (remove leading spaces)
- MD031: ensure blank lines around fenced code blocks
- MD032: ensure blank lines around top-level lists (before and after)
- MD040: add default language to unlabeled fenced code blocks ("text")
- MD047: ensure a single trailing newline at EOF
- MD010: replace hard tabs with spaces (outside code fences)

Intentionally NOT fixing: MD013 (line length), MD003 (heading style conversions), MD024/MD025 (duplicate/single H1), MD026 (trailing punctuation), etc.

Usage:
  python3 docs/repo_docs/repo_scripts/markdown_autofix_low_risk.py --path . [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import re
from typing import List, Tuple


IGNORE_DIRS = {
    "node_modules",
    ".git",
    ".obsidian",
    "venv",
    ".venv",
}

IGNORE_GLOBS_SUBSTR = [
    "/Output/",
    "/output/",
    "/outputs/",
]


HEADER_RE = re.compile(r"^(?P<indent>\s*)(?P<hashes>#{1,6})(?P<sp>\s*)(?P<title>.*)$")
ULIST_RE = re.compile(r"^(?P<indent>\s*)([-*+])\s+.+$")
OLIST_RE = re.compile(r"^(?P<indent>\s*)(\d+)\.[\s\S].*$")
OLIST_PREFIX_RE = re.compile(r"^(?P<indent>\s*)(?P<num>\d+)\.(?P<space>\s+)(?P<rest>.*)$")
FENCE_START_RE = re.compile(r"^\s*(```|~~~)(.*)$")


def is_ignored_path(path: str) -> bool:
    parts = path.split(os.sep)
    for p in parts:
        if p in IGNORE_DIRS:
            return True
    norm = path.replace(os.sep, "/")
    return any(s in norm for s in IGNORE_GLOBS_SUBSTR)


def strip_trailing_spaces(line: str) -> str:
    return re.sub(r"[ \t]+$", "", line)


def normalize_atx_heading(line: str) -> Tuple[str, bool]:
    m = HEADER_RE.match(line)
    if not m:
        return line, False
    hashes = m.group("hashes")
    title = m.group("title")
    # remove leading spaces and enforce single space after hashes
    normalized = f"{hashes} {title.strip()}"
    return normalized, True


def fix_file(path: str, dry_run: bool = False) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            original = f.read()
    except Exception:
        return False

    lines = original.splitlines(True)  # keep newlines
    out: List[str] = []

    in_code = False
    current_fence_token = None  # ``` or ~~~
    last_emitted_blank = False
    in_list_block = False
    in_blockquote_block = False
    prev_was_blockquote = False

    def emit_blank_once():
        nonlocal last_emitted_blank
        if not last_emitted_blank:
            out.append("\n")
            last_emitted_blank = True

    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip("\n")

        # Detect fence line
        fence_match = FENCE_START_RE.match(line)
        if fence_match:
            token = "```" if line.lstrip().startswith("```") else "~~~"
            # Opening fence
            if not in_code:
                # Ensure blank line before opening fence
                if len(out) > 0 and not last_emitted_blank:
                    emit_blank_once()
                # Add default language if missing (MD040)
                head = line.strip()
                # if it's only markers with optional spaces, no language specified
                if head == token:
                    out.append(f"{token} text\n")
                else:
                    out.append(strip_trailing_spaces(line) + "\n")
                in_code = True
                current_fence_token = token
                last_emitted_blank = False
                i += 1
                continue
            # Closing fence
            else:
                out.append(strip_trailing_spaces(line) + "\n")
                in_code = False
                current_fence_token = None
                last_emitted_blank = False
                # Ensure blank line after closing fence, unless EOF or already blank
                if i + 1 < len(lines):
                    nxt = lines[i + 1].rstrip("\n")
                    if nxt.strip() != "":
                        emit_blank_once()
                i += 1
                continue

        if in_code:
            # Collapse multiple blank lines inside code fences as well (safe for MD012)
            if line.strip() == "":
                if not last_emitted_blank:
                    out.append("\n")
                # else skip additional blank lines
                last_emitted_blank = True
            else:
                out.append(raw)
                last_emitted_blank = False
            i += 1
            continue

        # Outside code blocks:
        # Tabs to spaces (MD010)
        line = line.replace("\t", "    ")
        # Remove trailing spaces (MD009)
        line = strip_trailing_spaces(line)

        stripped = line.strip()

        # Collapse multiple blank lines (MD012)
        if stripped == "":
            # Avoid blank lines inside a blockquote block (MD028)
            if prev_was_blockquote:
                # Look ahead: if next non-empty line is also a blockquote, skip this blank
                j = i + 1
                skip_blank = False
                while j < len(lines):
                    nxt = lines[j].rstrip("\n")
                    if nxt.strip() == "":
                        j += 1
                        continue
                    if nxt.lstrip().startswith(">"):
                        skip_blank = True
                    break
                if skip_blank:
                    i += 1
                    continue
            emit_blank_once()
            i += 1
            continue

        # Normalize ATX headings (MD018/MD019/MD023) and ensure blank lines around (MD022)
        normalized, is_heading = normalize_atx_heading(line)
        if is_heading:
            # Ensure blank line before heading
            if len(out) > 0 and not last_emitted_blank:
                emit_blank_once()
            out.append(normalized + "\n")
            last_emitted_blank = False
            # Ensure blank line after heading by looking ahead to next non-blank, non-fence line
            # We'll insert a blank here, unless next line is already blank or a fence
            j = i + 1
            saw_blank = False
            saw_fence = False
            while j < len(lines):
                nxt = lines[j].rstrip("\n")
                if nxt.strip() == "":
                    saw_blank = True
                    break
                if FENCE_START_RE.match(nxt):
                    saw_fence = True
                    break
                # Non-blank, non-fence
                break
            if not saw_blank and not saw_fence:
                emit_blank_once()
            i += 1
            continue

        # List handling: mark entering a list block if this line starts a top-level list
        u_match = ULIST_RE.match(line)
        o_match = OLIST_RE.match(line)
        is_list_item = bool(u_match or o_match)

        if is_list_item:
            # Ensure blank line before list block if previous emitted wasn't blank and we are not already in a list block
            if not in_list_block and not last_emitted_blank and len(out) > 0:
                emit_blank_once()
            # Normalize ordered list numbering to 1. style (MD029)
            if o_match:
                m = OLIST_PREFIX_RE.match(line)
                if m:
                    line = f"{m.group('indent')}1.{m.group('space')}{m.group('rest')}"
            in_list_block = True
            out.append(line + "\n")
            last_emitted_blank = False
        else:
            # If we were in a list block and encounter a non-blank, non-list, ensure a blank line after the list
            if in_list_block and not last_emitted_blank:
                emit_blank_once()
            in_list_block = False
            # Blockquote handling (MD028): track blockquote blocks and avoid injecting extra blanks inside
            if line.lstrip().startswith(">"):
                # Ensure blank line before starting a blockquote block, unless at start or already blank
                if not prev_was_blockquote and len(out) > 0 and not last_emitted_blank:
                    emit_blank_once()
                in_blockquote_block = True
                prev_was_blockquote = True
            else:
                # Leaving blockquote block: ensure a single blank after if next content is non-blank, non-heading, non-list
                if prev_was_blockquote and not last_emitted_blank:
                    # We will emit a blank now to separate from following paragraph
                    emit_blank_once()
                in_blockquote_block = False
                prev_was_blockquote = False

            out.append(line + "\n")
            last_emitted_blank = False

        i += 1

    # Ensure single trailing newline (MD047)
    while len(out) > 0 and out[-1].strip() == "":
        out.pop()
    out.append("\n")

    new_content = "".join(out)
    changed = new_content != original
    if changed and not dry_run:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=".", help="Root path to scan")
    parser.add_argument("--dry-run", action="store_true", help="Only report files that would change")
    args = parser.parse_args()

    changed_files = []
    for root, dirs, files in os.walk(args.path):
        # prune ignored dirs in-place
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for name in files:
            if not name.lower().endswith(".md"):
                continue
            full = os.path.join(root, name)
            if is_ignored_path(full):
                continue
            try:
                changed = fix_file(full, dry_run=args.dry_run)
            except Exception:
                changed = False
            if changed:
                changed_files.append(full)

    if args.dry_run:
        print(f"[dry-run] Would update {len(changed_files)} files")
        for f in changed_files:
            print(f"- {f}")
    else:
        print(f"Updated {len(changed_files)} files")


if __name__ == "__main__":
    main()


