#!/usr/bin/env python3
"""Fix broken wikilinks where the path is wrong but the target file exists.

Only fixes links where:
1. The link contains a path separator (/)
2. No file exists at the linked path
3. A file with the same name exists at a UNIQUE location that matches the path context
"""

import re
from pathlib import Path
from collections import defaultdict

VAULT_ROOT = Path("/Users/4d/Documents/GitHub/cognitive")

def build_indices():
    """Build file path indices."""
    exclude_dirs = {'.obsidian', 'node_modules', 'venv', '.git', '__pycache__'}

    # Map from stem -> list of full paths (without .md)
    by_stem = defaultdict(list)
    # Set of all valid paths
    all_paths = set()
    # Map from partial path (last N segments) -> full path
    by_suffix = {}

    for f in VAULT_ROOT.rglob('*.md'):
        if any(d in f.parts for d in exclude_dirs):
            continue
        rel = str(f.relative_to(VAULT_ROOT))
        no_ext = rel[:-3]
        all_paths.add(no_ext)
        all_paths.add(rel)
        by_stem[f.stem].append(no_ext)

        # Build suffix index
        parts = Path(no_ext).parts
        for i in range(len(parts)):
            suffix = str(Path(*parts[i:]))
            if suffix not in by_suffix:
                by_suffix[suffix] = no_ext
            else:
                by_suffix[suffix] = None  # Ambiguous

    # Also add directory paths
    for d in VAULT_ROOT.rglob('*'):
        if d.is_dir() and not any(dd in d.parts for dd in exclude_dirs):
            all_paths.add(str(d.relative_to(VAULT_ROOT)))

    return all_paths, by_stem, by_suffix

def link_exists(target, all_paths):
    """Check if a link target resolves to an existing file."""
    if target in all_paths:
        return True
    if target + '.md' in all_paths:
        return True
    # Check as suffix
    for p in all_paths:
        if p.endswith('/' + target) or p.endswith('/' + target + '.md'):
            return True
    return False

def find_best_fix(broken_target, all_paths, by_stem, by_suffix):
    """Find the best fix for a broken link target."""
    parts = Path(broken_target).parts
    stem = parts[-1] if parts else broken_target

    # Strategy 1: Check if the suffix (last N path segments) uniquely matches
    for i in range(len(parts)):
        suffix = str(Path(*parts[i:]))
        if suffix in by_suffix and by_suffix[suffix] is not None:
            candidate = by_suffix[suffix]
            if candidate in all_paths:
                return candidate

    # Strategy 2: If stem is unique, use it
    if stem in by_stem and len(by_stem[stem]) == 1:
        return by_stem[stem][0]

    # Strategy 3: Find candidate with most path overlap
    if stem in by_stem:
        candidates = by_stem[stem]
        if len(candidates) == 1:
            return candidates[0]

        # Score by how many path segments match
        best_score = -1
        best_candidate = None
        for candidate in candidates:
            cand_parts = Path(candidate).parts
            score = 0
            for p in parts[:-1]:  # Check directory parts
                if p in cand_parts:
                    score += 1
            if score > best_score:
                best_score = score
                best_candidate = candidate

        if best_score > 0:
            return best_candidate

    return None

def main():
    print("Building indices...")
    all_paths, by_stem, by_suffix = build_indices()

    exclude_dirs = {'.obsidian', 'node_modules', 'venv', '.git', '__pycache__'}
    exclude_paths = {'docs/implementation/rxinfer'}

    md_files = sorted(VAULT_ROOT.rglob('*.md'))
    md_files = [f for f in md_files if not any(d in f.parts for d in exclude_dirs)]
    md_files = [f for f in md_files if not any(
        str(f.relative_to(VAULT_ROOT)).startswith(ep) for ep in exclude_paths
    )]

    print(f"Scanning {len(md_files)} files...")

    fixes_applied = 0
    files_modified = 0
    unfixable = []

    for file_path in md_files:
        try:
            content = file_path.read_text(encoding='utf-8')
        except (UnicodeDecodeError, FileNotFoundError):
            continue

        original = content
        rel_path = str(file_path.relative_to(VAULT_ROOT))

        def fix_link(m):
            nonlocal fixes_applied
            link_text = m.group(1)

            # Separate target from alias and anchor
            if '|' in link_text:
                target_part, alias = link_text.split('|', 1)
            else:
                target_part = link_text
                alias = None

            target = target_part.strip()
            anchor = ''
            if '#' in target:
                target, anchor = target.split('#', 1)

            # Skip empty targets
            if not target:
                return m.group(0)

            # Skip if link is valid
            if link_exists(target, all_paths):
                return m.group(0)

            # Only fix path-based links (contain /)
            if '/' not in target:
                return m.group(0)

            # Find fix
            fix = find_best_fix(target, all_paths, by_stem, by_suffix)
            if fix:
                fixes_applied += 1
                new_target = fix
                if anchor:
                    new_target += '#' + anchor
                if alias:
                    return f'[[{new_target}|{alias}]]'
                else:
                    return f'[[{new_target}]]'
            else:
                unfixable.append((rel_path, target))
                return m.group(0)

        content = re.sub(r'\[\[([^\]]+)\]\]', fix_link, content)

        if content != original:
            file_path.write_text(content, encoding='utf-8')
            files_modified += 1

    print(f"\nFixes applied: {fixes_applied}")
    print(f"Files modified: {files_modified}")
    print(f"Unfixable path-based links: {len(unfixable)}")

    if unfixable:
        print(f"\nUnfixable links (first 50):")
        seen = set()
        count = 0
        for fp, target in unfixable:
            key = (fp, target)
            if key in seen:
                continue
            seen.add(key)
            count += 1
            if count > 50:
                print(f"  ... and {len(set((f,t) for f,t in unfixable)) - 50} more")
                break
            print(f"  {fp}: [[{target}]]")

if __name__ == '__main__':
    main()
