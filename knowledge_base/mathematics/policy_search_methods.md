---

title: Policy Search Methods (Active Inference Lens)

type: mathematical_concept

status: draft

created: 2025-08-08

tags: [policy-optimization]

semantic_relations:

  - type: relates

    links: [policy_selection, expected_free_energy, natural_gradients]

---

## Policy Search Methods

### Families

- Enumerative / tree search (short horizon)

- Gradient-based (natural gradient on policy parameters)

- Sampling-based (CEM, evolution strategies)

### Notes

- Use EFE as objective surrogate; anneal [[precision_parameter]] during search

- Combine with [[natural_gradients]] for geometry-aware updates

