#!/usr/bin/env python3
"""Validate all Obsidian wikilinks in the repo.

Checks that [[target]] links point to files that actually exist.
Reports broken links that need fixing.
"""

import re
from pathlib import Path
from collections import defaultdict

VAULT_ROOT = Path("/Users/4d/Documents/GitHub/cognitive")

def build_file_set():
    """Build a set of all file paths (with and without .md extension)."""
    exclude_dirs = {'.obsidian', 'node_modules', 'venv', '.git', '__pycache__'}
    files = set()

    for f in VAULT_ROOT.rglob('*'):
        if f.is_file() and not any(d in f.parts for d in exclude_dirs):
            rel = str(f.relative_to(VAULT_ROOT))
            files.add(rel)
            # Also add without extension for .md files
            if rel.endswith('.md'):
                files.add(rel[:-3])
            # Add the stem (filename without extension)
            files.add(f.stem)

    # Also add directory paths
    for d in VAULT_ROOT.rglob('*'):
        if d.is_dir() and not any(dd in d.parts for dd in exclude_dirs):
            rel = str(d.relative_to(VAULT_ROOT))
            files.add(rel)

    return files

def find_wikilinks(content: str):
    """Extract all wikilinks from content."""
    pattern = r'\[\[([^\]]+)\]\]'
    links = []
    for m in re.finditer(pattern, content):
        link_text = m.group(1)
        # Split on | to get target
        if '|' in link_text:
            target = link_text.split('|')[0].strip()
        else:
            target = link_text.strip()
        # Remove anchor
        if '#' in target:
            target = target.split('#')[0].strip()
        # Skip empty targets (anchor-only links)
        if target:
            links.append((target, m.start()))
    return links

def check_link(target: str, file_set: set) -> bool:
    """Check if a wikilink target resolves to an existing file."""
    # Direct match
    if target in file_set:
        return True
    # With .md extension
    if target + '.md' in file_set:
        return True
    # Try lowercased
    target_lower = target.lower()
    for f in file_set:
        if f.lower() == target_lower or f.lower() == target_lower + '.md':
            return True
    # Check if any file ends with the target
    for f in file_set:
        if f.endswith('/' + target) or f.endswith('/' + target + '.md'):
            return True
    return False

def main():
    print("Building file set...")
    file_set = build_file_set()
    print(f"Found {len(file_set)} path entries")

    exclude_dirs = {'.obsidian', 'node_modules', 'venv', '.git', '__pycache__'}
    exclude_paths = {'docs/implementation/rxinfer'}

    md_files = sorted(VAULT_ROOT.rglob('*.md'))
    md_files = [f for f in md_files if not any(d in f.parts for d in exclude_dirs)]
    md_files = [f for f in md_files if not any(
        str(f.relative_to(VAULT_ROOT)).startswith(ep) for ep in exclude_paths
    )]

    print(f"Scanning {len(md_files)} markdown files...")

    broken_links = defaultdict(list)
    total_links = 0
    valid_links = 0

    for file_path in md_files:
        try:
            content = file_path.read_text(encoding='utf-8')
        except (UnicodeDecodeError, FileNotFoundError):
            continue

        links = find_wikilinks(content)
        rel_path = str(file_path.relative_to(VAULT_ROOT))

        for target, pos in links:
            total_links += 1
            if check_link(target, file_set):
                valid_links += 1
            else:
                # Find line number
                line_num = content[:pos].count('\n') + 1
                broken_links[rel_path].append((target, line_num))

    print(f"\nTotal wikilinks found: {total_links}")
    print(f"Valid links: {valid_links}")
    print(f"Broken links: {total_links - valid_links}")

    if broken_links:
        print(f"\nBroken links by file ({len(broken_links)} files):")
        # Sort by number of broken links
        for file_path, links in sorted(broken_links.items(), key=lambda x: -len(x[1])):
            print(f"\n  {file_path} ({len(links)} broken):")
            for target, line in links:
                print(f"    L{line}: [[{target}]]")

if __name__ == '__main__':
    main()
