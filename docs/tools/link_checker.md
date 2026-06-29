# Link Checker

`code/scripts/verify_links.py` validates Obsidian wiki links without treating
every unresolved concept as a broken file reference.

Default mode resolves existing wiki targets and reports only explicit file
references that are missing, such as `[[missing.md]]`. Extensionless links such
as `[[active_inference]]` or `[[systems/network_theory]]` are intentional graph
edges in the knowledge base and are skipped when no matching file exists.

Run the default gate:

```bash
python code/scripts/verify_links.py .
```

Run a stricter audit when you want every wiki link to resolve to a file:

```bash
python code/scripts/verify_links.py . --strict-wiki-links
```
