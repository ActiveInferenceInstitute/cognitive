---
title: API documentation overview
type: index
status: stable
---

# API documentation overview

The runtime API is organized around validated probability models and explicit
agent families:

1. `cognitive.models.active_inference` contains discrete inference,
   homeostatic control, lifecycle state, and persistence.
2. `cognitive.models.matrices` contains matrix operations, loaders,
   initializers, and plotting data adapters.
3. `cognitive.utils` contains probability helpers, node creation, and
   knowledge-network construction.
4. `Things.Simple_POMDP` and `Things.Continuous_Generic` contain the two
   maintained concrete agent families.

Use [`api_reference.md`](api_reference.md) for signatures and
[`../../README.md`](../../README.md) for a runnable example. Theory remains in
`knowledge_base/`; it is not treated as a runtime import surface.
