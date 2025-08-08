---

title: Natural Gradients

type: mathematical_concept

status: stable

created: 2025-08-08

tags: [optimization, information-geometry]

semantic_relations:

  - type: relates

    links: [information_geometry, fisher_information]

  - type: used_by

    links: [policy_selection, action_distribution]

---

## Natural Gradients

### Definition

Given objective \(F(\theta)\) on a statistical manifold with metric \(G(\theta)=I(\theta)\):

```math

\tilde{\nabla} F(\theta) = G(\theta)^{-1} \nabla F(\theta)

```

### Benefits

- Scale-invariant steps, faster convergence in probabilistic models

- Principled preconditioning via [[fisher_information]]

### Applications

- Variational inference updates, policy optimization, distributional control

### See also

- [[information_geometry]] · [[fisher_information]] · [[policy_selection]]

