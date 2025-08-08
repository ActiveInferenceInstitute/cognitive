---

title: Kullback–Leibler (KL) Divergence

type: mathematical_concept

status: stable

created: 2025-08-08

tags: [information-theory, divergence, probability]

semantic_relations:

  - type: relates

    links: [information_theory, variational_inference, expected_free_energy]

  - type: underpins

    links: [variational_free_energy, epistemic_value]

---

## KL Divergence

### Definition

For distributions p and q on X (with p absolutely continuous w.r.t. q):

```math

D_{\mathrm{KL}}(p\,\|\,q) = \int_X p(x)\,\log\frac{p(x)}{q(x)}\,dx

```

### Properties

- Non-negativity: \(D_{\mathrm{KL}}(p\|q) \ge 0\), equality iff \(p=q\) a.e.

- Asymmetry: \(D_{\mathrm{KL}}(p\|q) \neq D_{\mathrm{KL}}(q\|p)\)

- Information projection: arises in variational optimization under constraints

### Roles in this repository

- [[variational_free_energy]]: complexity term \(D_{\mathrm{KL}}\big(q(s)\,\|\,p(s)\big)\)

- [[expected_free_energy]]: risk term \(D_{\mathrm{KL}}\big(q(o|\pi)\,\|\,p(o)\big)\); epistemic value as expected KL

- [[information_theory]]: connects to entropy and mutual information

### Implementation notes

- Use log-sum-exp stabilization in discrete sums

- Guard zeros with small epsilons when computing logs

### See also

- [[information_theory]] · [[variational_inference]] · [[variational_free_energy]] · [[expected_free_energy]]

