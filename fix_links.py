#!/usr/bin/env python3
"""Fix all Obsidian links in the cognitive repo.

1. Convert [[../path]] wikilinks to correct vault-root-relative [[path]]
2. Convert [text](local-path.md) markdown links to [[path|text]] wikilinks

Smart resolution: when ../path resolves to a non-existent file,
searches the vault for the best matching file.
"""

import os
import re
from pathlib import Path
from functools import lru_cache

VAULT_ROOT = Path("/Users/4d/Documents/GitHub/cognitive")
DRY_RUN = False

# Build file index once
_file_index = None

def build_file_index():
    """Build an index of all files in the vault for smart resolution."""
    global _file_index
    if _file_index is not None:
        return _file_index

    _file_index = {}
    exclude_dirs = {'.obsidian', 'node_modules', 'venv', '.git', '__pycache__'}

    for f in VAULT_ROOT.rglob('*'):
        if f.is_file() and not any(d in f.parts for d in exclude_dirs):
            rel = str(f.relative_to(VAULT_ROOT))
            # Index by full relative path
            _file_index[rel] = rel
            # Index by path without extension
            no_ext = str(f.relative_to(VAULT_ROOT).with_suffix(''))
            _file_index[no_ext] = no_ext
            # Index by filename without extension
            stem = f.stem
            if stem not in _file_index:
                _file_index[stem] = no_ext
            # Index by last N path components
            parts = Path(no_ext).parts
            for i in range(1, len(parts)):
                partial = str(Path(*parts[i:]))
                if partial not in _file_index:
                    _file_index[partial] = no_ext

    return _file_index

def find_best_match(target: str) -> str | None:
    """Find the best matching file in the vault for a given target path."""
    idx = build_file_index()

    # Clean up the target
    clean = target.strip().rstrip('/')
    if clean.endswith('.md'):
        clean = clean[:-3]

    # Direct match
    if clean in idx:
        return idx[clean]

    # Try without leading path separators
    if clean.startswith('/'):
        clean = clean[1:]
        if clean in idx:
            return idx[clean]

    # Try matching just the tail segments
    parts = Path(clean).parts
    for i in range(len(parts)):
        partial = str(Path(*parts[i:]))
        if partial in idx:
            return idx[partial]

    # Try matching with common path prefixes
    common_prefixes = ['knowledge_base/', 'docs/', 'code/', 'code/Things/',
                       'code/tools/', 'code/tests/']
    for prefix in common_prefixes:
        candidate = prefix + clean
        if candidate in idx:
            return idx[candidate]

    # For paths that look like they should be under knowledge_base
    # (cognitive, mathematics, biology, systems, philosophy, ontology, agents, etc.)
    kb_dirs = {'cognitive', 'mathematics', 'biology', 'systems', 'philosophy',
               'ontology', 'agents', 'BioFirm', 'research', 'citations',
               'free_energy_principle'}
    if parts and parts[0] in kb_dirs:
        candidate = 'knowledge_base/' + clean
        if candidate in idx:
            return idx[candidate]

    # For paths that look like docs
    doc_dirs = {'guides', 'api', 'repo_docs', 'templates', 'implementation',
                'examples', 'development', 'config', 'research', 'tools'}
    if parts and parts[0] in doc_dirs:
        candidate = 'docs/' + clean
        if candidate in idx:
            return idx[candidate]

    # For paths starting with Things
    if parts and parts[0] == 'Things':
        candidate = 'code/' + clean
        if candidate in idx:
            return idx[candidate]

    return None


def resolve_link_target(file_path: Path, link_target: str) -> str:
    """Resolve a link target to a correct vault-root-relative path.

    Strategy:
    1. Try direct mechanical resolution of ../
    2. If result exists, use it
    3. If not, use smart matching to find the intended target
    """
    clean_target = link_target.strip().rstrip('/')

    # Remove trailing .md
    has_anchor = '#' in clean_target
    anchor = ''
    if has_anchor:
        clean_target, anchor = clean_target.split('#', 1)

    if clean_target.endswith('.md'):
        clean_target = clean_target[:-3]

    # Handle ./ prefix
    if clean_target.startswith('./'):
        clean_target = clean_target[2:]

    # Handle ../ prefix - resolve mechanically first
    if clean_target.startswith('../'):
        file_dir = file_path.parent
        # Mechanically resolve
        resolved = (file_dir / clean_target).resolve()
        try:
            vault_relative = str(resolved.relative_to(VAULT_ROOT))
        except ValueError:
            vault_relative = clean_target

        # Check if the resolved path actually exists
        check_path = VAULT_ROOT / vault_relative
        if check_path.exists() or (check_path.with_suffix('.md')).exists():
            result = vault_relative
            if result.endswith('.md'):
                result = result[:-3]
            return result + (f'#{anchor}' if anchor else '')

        # Mechanical resolution failed - use smart matching
        # Extract the meaningful part after resolving ../
        # e.g., from code/Things/AGENTS.md, ../knowledge_base/X resolves to code/knowledge_base/X
        # but the intended target is knowledge_base/X
        best = find_best_match(vault_relative)
        if best:
            return best + (f'#{anchor}' if anchor else '')

        # Try with just the original path minus ../
        stripped = re.sub(r'^(\.\./)+', '', clean_target)
        best = find_best_match(stripped)
        if best:
            return best + (f'#{anchor}' if anchor else '')

        # Last resort: return the stripped version (better than ../...)
        return stripped + (f'#{anchor}' if anchor else '')

    # Non-relative path - check if it resolves
    # For paths used in current-dir context (e.g., mathematics/core_principle)
    if not clean_target.startswith(('http', 'mailto')):
        # Resolve relative to the file's directory
        file_dir = file_path.parent
        check = file_dir / clean_target
        if check.exists() or check.with_suffix('.md').exists():
            try:
                result = str(check.resolve().relative_to(VAULT_ROOT))
                if result.endswith('.md'):
                    result = result[:-3]
                return result + (f'#{anchor}' if anchor else '')
            except ValueError:
                pass

        # Smart match
        best = find_best_match(clean_target)
        if best:
            return best + (f'#{anchor}' if anchor else '')

    return clean_target + (f'#{anchor}' if anchor else '')


def fix_relative_wikilinks(content: str, file_path: Path) -> str:
    """Fix [[../path]] and [[../path|alias]] wikilinks."""
    def replace_match(m):
        full_match = m.group(0)
        link_part = m.group(1)

        # Split on | to handle aliases
        if '|' in link_part:
            target, alias = link_part.split('|', 1)
        else:
            target = link_part
            alias = None

        # Only fix if target contains ../
        if '../' not in target:
            return full_match

        resolved = resolve_link_target(file_path, target)

        if alias:
            return f'[[{resolved}|{alias}]]'
        else:
            return f'[[{resolved}]]'

    # Match [[...]] patterns
    pattern = r'\[\[([^\]]+)\]\]'
    return re.sub(pattern, replace_match, content)


def fix_markdown_internal_links(content: str, file_path: Path) -> str:
    """Convert [text](local-path) to [[path|text]] wikilinks.

    Only converts internal links (not http/https/mailto/anchors/images).
    """
    def replace_match(m):
        full_match = m.group(0)
        text = m.group(1)
        target = m.group(2)

        # Skip external URLs
        if target.startswith(('http://', 'https://', 'mailto:')):
            return full_match

        # Skip anchor-only links
        if target.startswith('#'):
            return full_match

        resolved = resolve_link_target(file_path, target)

        # Build wikilink
        if text and text != resolved:
            return f'[[{resolved}|{text}]]'
        else:
            return f'[[{resolved}]]'

    # Match [text](path) but not ![text](path) (images)
    pattern = r'(?<!!)\[([^\]]*)\]\(([^)]+)\)'

    def smart_replace(m):
        target = m.group(2)
        if target.startswith(('http://', 'https://', 'mailto:')):
            return m.group(0)
        return replace_match(m)

    return re.sub(pattern, smart_replace, content)


def process_file(file_path: Path) -> tuple[bool, list[str]]:
    """Process a single file and return (changed, changes_list)."""
    changes = []

    try:
        content = file_path.read_text(encoding='utf-8')
    except (UnicodeDecodeError, FileNotFoundError):
        return False, []

    original = content

    # Fix relative wikilinks (../path)
    content = fix_relative_wikilinks(content, file_path)

    # Fix markdown internal links [text](path)
    content = fix_markdown_internal_links(content, file_path)

    if content != original:
        if not DRY_RUN:
            file_path.write_text(content, encoding='utf-8')

        # Find what changed
        orig_lines = original.splitlines()
        new_lines = content.splitlines()
        for i, (old, new) in enumerate(zip(orig_lines, new_lines)):
            if old != new:
                rel_path = file_path.relative_to(VAULT_ROOT)
                changes.append(f"  {rel_path}:{i+1}")
                changes.append(f"    - {old.strip()}")
                changes.append(f"    + {new.strip()}")

        return True, changes

    return False, []


def main():
    files_changed = 0
    total_changes = []
    unresolved = []

    # Build file index
    print("Building file index...")
    build_file_index()
    print(f"Indexed {len(_file_index)} path entries")

    # Find all markdown files
    md_files = sorted(VAULT_ROOT.rglob('*.md'))
    exclude_dirs = {'.obsidian', 'node_modules', 'venv', '.git', '__pycache__'}
    # Also exclude rxinfer docs (external docs with their own image link structure)
    exclude_paths = {'docs/implementation/rxinfer'}
    md_files = [f for f in md_files if not any(d in f.parts for d in exclude_dirs)]
    md_files = [f for f in md_files if not any(
        str(f.relative_to(VAULT_ROOT)).startswith(ep) for ep in exclude_paths
    )]

    print(f"Scanning {len(md_files)} markdown files...")

    for file_path in md_files:
        changed, changes = process_file(file_path)
        if changed:
            files_changed += 1
            total_changes.extend(changes)

    print(f"\n{'DRY RUN - ' if DRY_RUN else ''}Files modified: {files_changed}")
    print(f"Total line changes: {len([c for c in total_changes if c.startswith('    -')])}")

    if total_changes:
        print("\nChanges made (first 300 lines):")
        for change in total_changes[:300]:
            print(change)
        if len(total_changes) > 300:
            print(f"\n... and {len(total_changes) - 300} more lines")


if __name__ == '__main__':
    main()
