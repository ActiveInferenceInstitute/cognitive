---

title: Annealing Schedules

type: concept

status: stable

created: 2025-08-08

tags: [optimization, schedules, stability]

semantic_relations:

  - type: relates

    links: [precision_parameter, convergence_control]

  - type: used_by

    links: [policy_selection, variational_inference]

---

## Annealing Schedules

### Common schedules

- Exponential: \(T_t = T_0 \alpha^t\)

- Inverse time: \(T_t = \tfrac{T_0}{1+kt}\)

- Cosine: \(T_t = T_{min} + \tfrac{1}{2}(T_{max}-T_{min})(1+\cos(\pi t / T))\)

### Uses

- Reduce temperature (increase \(\gamma\)) to transition from exploration to exploitation

- Smooth convergence of VI and policy optimization

### See also

- [[precision_parameter]] · [[convergence_control]] · [[policy_selection]]

