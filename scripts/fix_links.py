import json
import csv
from pathlib import Path
from typing import Dict, Set, List, Tuple
import re
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
import time

class LinkAnalyzer:
    def __init__(self, output_dir: Path):
        """Initialize link analyzer with output directory.
        
        Args:
            output_dir (Path): Path to directory containing analysis files
        """
        self.output_dir = output_dir
        
        # Increase CSV field size limit
        csv.field_size_limit(2**30)  # Set to a large value
        
        # Load data files
        try:
            self.structure = self._load_json('file_structure.json')
            self.files = self._load_csv('files.csv')
            self.wikilinks = self._load_csv('wikilinks.csv')
            self.existing_files = self._get_existing_files()
            self.link_graph = self._build_link_graph()
        except Exception as e:
            print(f"Error loading data files: {e}")
            raise

    def _load_json(self, filename: str) -> Dict:
        """Load JSON file from output directory."""
        try:
            with open(self.output_dir / filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            raise

    def _load_csv(self, filename: str) -> List[Dict]:
        """Load CSV file from output directory."""
        try:
            with open(self.output_dir / filename, 'r', encoding='utf-8') as f:
                return list(csv.DictReader(f))
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            raise

    def _get_existing_files(self) -> Set[str]:
        """Get set of existing file paths."""
        return {f['path'] for f in self.files}

    def _build_link_graph(self) -> Dict[str, Set[str]]:
        """Build graph of file links."""
        graph = defaultdict(set)
        for link in self.wikilinks:
            source = link['source']
            target = link['target']
            graph[source].add(target)
        return dict(graph)

    def _find_broken_links(self) -> Dict[str, Set[str]]:
        """Find broken links in the knowledge base."""
        broken_links = defaultdict(set)
        for source, targets in self.link_graph.items():
            for target in targets:
                target_path = f"knowledge_base/{target}.md"
                if target_path not in self.existing_files:
                    broken_links[source].add(target)
        return dict(broken_links)

    def _find_ambiguous_links(self) -> Dict[str, List[str]]:
        """Find ambiguous links that could refer to multiple files."""
        ambiguous = {}
        for source, targets in self.link_graph.items():
            for target in targets:
                matches = []
                pattern = f".*{target}.*\\.md$"
                for file in self.existing_files:
                    if re.match(pattern, file, re.IGNORECASE):
                        matches.append(file)
                if len(matches) > 1:
                    ambiguous[f"{source} -> {target}"] = matches
        return ambiguous

    def _find_missing_backlinks(self) -> Dict[str, Set[str]]:
        """Find files that should have reciprocal links."""
        missing_backlinks = defaultdict(set)
        for source, targets in self.link_graph.items():
            for target in targets:
                target_path = f"knowledge_base/{target}.md"
                if target_path in self.existing_files:
                    if source not in self.link_graph.get(target_path, set()):
                        missing_backlinks[target_path].add(source)
        return dict(missing_backlinks)

    def _similarity_score(self, a: str, b: str) -> float:
        """Compute similarity score between two strings."""
        a = a.lower()
        b = b.lower()
        if not a or not b:
            return 0.0
        intersection = len(set(a) & set(b))
        union = len(set(a) | set(b))
        return intersection / union if union > 0 else 0.0

    def suggest_fixes(self) -> Dict[str, List[Dict]]:
        """Suggest fixes for link issues."""
        fixes = defaultdict(list)
        
        # Check broken links
        broken = self._find_broken_links()
        for source, targets in broken.items():
            for target in targets:
                # Find similar existing files
                suggestions = []
                for existing in self.existing_files:
                    score = self._similarity_score(target, Path(existing).stem)
                    if score > 0.5:  # Threshold for similarity
                        suggestions.append((existing, score))
                suggestions.sort(key=lambda x: x[1], reverse=True)
                
                fix = {
                    'type': 'broken_link',
                    'source': source,
                    'target': target,
                    'suggestions': [s[0] for s in suggestions[:3]]
                }
                fixes[source].append(fix)
        
        # Check ambiguous links
        ambiguous = self._find_ambiguous_links()
        for link, matches in ambiguous.items():
            source, target = link.split(' -> ')
            fix = {
                'type': 'ambiguous_link',
                'source': source,
                'target': target,
                'matches': matches
            }
            fixes[source].append(fix)
        
        # Check missing backlinks
        missing = self._find_missing_backlinks()
        for target, sources in missing.items():
            for source in sources:
                fix = {
                    'type': 'missing_backlink',
                    'source': target,
                    'target': source
                }
                fixes[target].append(fix)
        
        return dict(fixes)

    def _get_template_for_link(self, link_name: str) -> str:
        """Generate template content for a new file."""
        return f"""---
title: {link_name.replace('_', ' ').title()}
type: knowledge_base
status: draft
created: {datetime.now().strftime('%Y-%m-%d')}
tags:
  - todo
semantic_relations:
  - type: related
    links: []
---

# {link_name.replace('_', ' ').title()}

[TODO: Add content]
"""

    def apply_fixes(self, fixes: Dict[str, List[Dict]], kb_root: Path) -> List[Dict]:
        """Apply suggested fixes to the knowledge base."""
        changes = []
        
        for source, file_fixes in fixes.items():
            source_path = kb_root / source.lstrip('/')
            if not source_path.exists():
                continue
                
            content = source_path.read_text(encoding='utf-8')
            modified = False
            
            for fix in file_fixes:
                if fix['type'] == 'broken_link':
                    if fix['suggestions']:
                        # Replace broken link with first suggestion
                        old_link = f"[[{fix['target']}]]"
                        new_link = f"[[{Path(fix['suggestions'][0]).stem}]]"
                        if old_link in content:
                            content = content.replace(old_link, new_link)
                            modified = True
                            changes.append({
                                'type': 'fixed_link',
                                'file': str(source),
                                'old': old_link,
                                'new': new_link
                            })
                    else:
                        # Create new file for broken link
                        new_file = kb_root / f"{fix['target']}.md"
                        if not new_file.exists():
                            new_file.parent.mkdir(parents=True, exist_ok=True)
                            new_file.write_text(
                                self._get_template_for_link(fix['target']),
                                encoding='utf-8'
                            )
                            changes.append({
                                'type': 'created_file',
                                'file': str(new_file)
                            })
                
                elif fix['type'] == 'missing_backlink':
                    target_path = kb_root / fix['target'].lstrip('/')
                    if target_path.exists():
                        target_content = target_path.read_text(encoding='utf-8')
                        # Add backlink in semantic_relations
                        if 'semantic_relations:' in target_content:
                            lines = target_content.splitlines()
                            for i, line in enumerate(lines):
                                if line.strip() == 'semantic_relations:':
                                    # Find the related section or create it
                                    related_found = False
                                    for j in range(i+1, len(lines)):
                                        if '  - type: related' in lines[j]:
                                            related_found = True
                                            # Find or create links section
                                            for k in range(j+1, len(lines)):
                                                if 'links:' in lines[k]:
                                                    lines[k] = lines[k].rstrip() + f" [[{Path(source).stem}]]"
                                                    break
                                            break
                                    if not related_found:
                                        lines.insert(i+1, '  - type: related\n    links: []')
                                    target_content = '\n'.join(lines)
                                    target_path.write_text(target_content, encoding='utf-8')
                                    changes.append({
                                        'type': 'added_backlink',
                                        'file': str(target_path),
                                        'link': str(source)
                                    })
                                    break
            
            if modified:
                source_path.write_text(content, encoding='utf-8')
        
        return changes

    def generate_report(self, fixes: Dict[str, List[Dict]], changes: List[Dict]) -> str:
        """Generate a detailed report of link analysis and changes."""
        report = ["# Link Analysis Report\n"]
        
        # Summarize issues
        total_issues = sum(len(fixes) for fixes in fixes.values())
        report.append(f"## Summary\nTotal issues found: {total_issues}\n")
        
        # Detail fixes by type
        report.append("## Issues by Type\n")
        
        # Collect all issues by type
        broken_links = []
        ambiguous_links = []
        missing_backlinks = []
        
        for source_file, file_fixes in fixes.items():
            for fix in file_fixes:
                if fix['type'] == 'broken_link':
                    broken_links.append({
                        'source': source_file,
                        'target': fix['target'],
                        'suggestions': fix.get('suggestions', [])
                    })
                elif fix['type'] == 'ambiguous_link':
                    ambiguous_links.append({
                        'source': source_file,
                        'target': fix['target'],
                        'matches': fix.get('matches', [])
                    })
                elif fix['type'] == 'missing_backlink':
                    missing_backlinks.append({
                        'source': fix['source'],
                        'target': fix['target']
                    })
        
        # Report broken links
        if broken_links:
            report.append(f"### Broken Links ({len(broken_links)})\n")
            report.append("Links that point to non-existent files:\n")
            for issue in broken_links:
                report.append(f"- In `{issue['source']}`: [[{issue['target']}]]")
                if issue['suggestions']:
                    report.append("  Suggestions:")
                    for suggestion in issue['suggestions']:
                        report.append(f"  - `{suggestion}`")
            report.append("")
        
        # Report ambiguous links
        if ambiguous_links:
            report.append(f"### Ambiguous Links ({len(ambiguous_links)})\n")
            report.append("Links that could refer to multiple files:\n")
            for issue in ambiguous_links:
                report.append(f"- In `{issue['source']}`: [[{issue['target']}]]")
                report.append("  Could refer to:")
                for match in issue['matches']:
                    report.append(f"  - `{match}`")
            report.append("")
        
        # Report missing backlinks
        if missing_backlinks:
            report.append(f"### Missing Backlinks ({len(missing_backlinks)})\n")
            report.append("Files that should have reciprocal links:\n")
            for issue in missing_backlinks:
                report.append(f"- `{issue['source']}` should link back to `{issue['target']}`")
            report.append("")
        
        # List changes made
        report.append("## Changes Applied")
        change_types = defaultdict(list)
        for change in changes:
            change_types[change['type']].append(change)
        
        for change_type, type_changes in change_types.items():
            report.append(f"\n### {change_type.replace('_', ' ').title()}")
            for change in type_changes:
                if change_type == 'fixed_link':
                    report.append(f"- In {change['file']}: {change['old']} → {change['new']}")
                elif change_type == 'created_file':
                    report.append(f"- Created {change['file']}")
                elif change_type == 'added_backlink':
                    report.append(f"- Added backlink in {change['file']} to {change['link']}")
        
        return '\n'.join(report)

    def generate_summary_report(self, fixes: Dict[str, List[Dict]], changes: List[Dict]) -> str:
        """Generate a concise summary report with samples of each issue type."""
        report = ["# Link Analysis Summary Report\n"]
        
        # Overall statistics
        total_issues = sum(len(fixes) for fixes in fixes.values())
        report.append(f"## Overall Statistics\n")
        report.append(f"- Total files analyzed: {len(self.files)}")
        report.append(f"- Total issues found: {total_issues}")
        report.append(f"- Total files with issues: {len(fixes)}\n")
        
        # Collect issues by type
        broken_links = []
        ambiguous_links = []
        missing_backlinks = []
        
        for source_file, file_fixes in fixes.items():
            for fix in file_fixes:
                if fix['type'] == 'broken_link':
                    broken_links.append({
                        'source': source_file,
                        'target': fix['target'],
                        'suggestions': fix.get('suggestions', [])
                    })
                elif fix['type'] == 'ambiguous_link':
                    ambiguous_links.append({
                        'source': source_file,
                        'target': fix['target'],
                        'matches': fix.get('matches', [])
                    })
                elif fix['type'] == 'missing_backlink':
                    missing_backlinks.append({
                        'source': fix['source'],
                        'target': fix['target']
                    })
        
        # Report issue type summaries with samples
        report.append("## Issue Type Summary\n")
        
        # Broken links summary
        if broken_links:
            report.append(f"### Broken Links: {len(broken_links)} issues\n")
            report.append("Sample issues (up to 5):")
            for issue in broken_links[:5]:
                report.append(f"- In `{issue['source']}`: [[{issue['target']}]]")
                if issue['suggestions']:
                    report.append("  Suggestions:")
                    for suggestion in issue['suggestions'][:3]:
                        report.append(f"  - `{suggestion}`")
            report.append("")
        
        # Ambiguous links summary
        if ambiguous_links:
            report.append(f"### Ambiguous Links: {len(ambiguous_links)} issues\n")
            report.append("Sample issues (up to 5):")
            for issue in ambiguous_links[:5]:
                report.append(f"- In `{issue['source']}`: [[{issue['target']}]]")
                report.append("  Could refer to:")
                for match in issue['matches'][:3]:
                    report.append(f"  - `{match}`")
            report.append("")
        
        # Missing backlinks summary
        if missing_backlinks:
            report.append(f"### Missing Backlinks: {len(missing_backlinks)} issues\n")
            report.append("Sample issues (up to 5):")
            for issue in missing_backlinks[:5]:
                report.append(f"- `{issue['source']}` should link back to `{issue['target']}`")
            report.append("")
        
        # Add recommendations
        report.append("## Recommendations\n")
        report.append("1. Start by fixing broken links, as they affect content accessibility")
        report.append("2. Resolve ambiguous links to ensure correct references")
        report.append("3. Add missing backlinks to improve navigation")
        report.append("\nNote: Use the full report for complete issue details.")
        
        return '\n'.join(report)

def main():
    """Main function to analyze and fix links."""
    script_dir = Path(__file__).parent
    output_dir = script_dir / 'output'
    kb_root = script_dir.parent / 'knowledge_base'
    
    start_time = time.time()
    
    print("\n🔍 Knowledge Base Link Analysis")
    print("=" * 40)
    
    print("\n⚡ Initializing link analyzer...")
    analyzer = LinkAnalyzer(output_dir)
    
    print("\n📊 Analyzing link issues...")
    with tqdm(total=3, desc="Progress", unit="step") as pbar:
        fixes = analyzer.suggest_fixes()
        pbar.update(1)
        
        print("\n🛠️  Applying fixes...")
        changes = analyzer.apply_fixes(fixes, kb_root)
        pbar.update(1)
        
        print("\n📝 Generating reports...")
        report = analyzer.generate_report(fixes, changes)
        summary = analyzer.generate_summary_report(fixes, changes)
        pbar.update(1)
    
    # Save reports
    report_file = output_dir / 'link_analysis_report.md'
    summary_file = output_dir / 'link_analysis_summary.md'
    report_file.write_text(report, encoding='utf-8')
    summary_file.write_text(summary, encoding='utf-8')
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n✨ Analysis complete in {duration:.2f}s")
    print(f"📄 Full report saved to {report_file}")
    print(f"📑 Summary report saved to {summary_file}")
    print("\nSummary:")
    print(f"🔗 Fixed links: {sum(1 for c in changes if c['type'] == 'fixed_link')}")
    print(f"📁 Created files: {sum(1 for c in changes if c['type'] == 'created_file')}")
    print(f"🔄 Added backlinks: {sum(1 for c in changes if c['type'] == 'added_backlink')}")
    print("=" * 40 + "\n")

if __name__ == "__main__":
    main() 