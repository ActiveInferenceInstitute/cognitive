---
title: Repository Policy Files
type: reference
status: stable
updated: 2026-10-10
---

# Repository policy files

Machine-readable policy that the repository's gates enforce. Each file here is
the single source of one rule, so the rule can be read without reading the
script that applies it.

| File | Enforced by | Rule |
|---|---|---|
| [`documentation_terms.yaml`](documentation_terms.yaml) | `code/scripts/validate_docs.py` | Terms that may not appear in tracked text, and the files exempt from the scan. |

## Documentation terms

The term list rejects words that advertise content the repository does not
have: a page describing a section as incomplete work makes a claim the
repository cannot keep, and a reader has no way to tell a genuine gap from a
forgotten marker.

Two properties of the policy are deliberate.

**The list is data, not code.** The validator reads this YAML file
(`load_term_policy` in `code/scripts/validate_docs.py`). A gate that has to
hide its own vocabulary from itself is a gate nobody can audit, so the files
that must spell the terms literally are named as `exempt_paths` instead.

**The scan covers source files, not only documentation.** An in-code work
marker therefore fails the gate. The tradeoff is real: a marker in a comment is
an honest signal, and this policy pushes it out of the comment. It is pushed to
[`TODO.md`](../../TODO.md), the tracked backlog, where it is visible to a
reader who is not reading that file's source.

The policy must not become a rename. Substituting a synonym for a banned term
turns a gate about accuracy into a gate about vocabulary and leaves the
inaccurate content in place under a better name. Resolve a flagged term by
fixing what it describes or by moving it to the backlog.
