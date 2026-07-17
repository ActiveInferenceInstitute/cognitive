---
title: API documentation
type: documentation
status: stable
---

# API documentation

The canonical runtime API is the installed `cognitive` package. Start with
[`api_reference.md`](api_reference.md), the root
[`README.md`](../../README.md), and the executable manuscript in
[`../manuscript/README.md`](../manuscript/README.md).

Public exports are checked by `code/scripts/validate_docs.py` and integration
tests import the package after editable installation. The API documentation
does not describe classes that are absent from the package.
