---

title: Worked Example — Expected Free Energy

type: example

status: draft

created: 2025-08-08

tags: [example, efe]

semantic_relations:

  - type: demonstrates

    links: [expected_free_energy]

---

## EFE on a toy POMDP

### Setup

Small POMDP with 2 states, 2 observations, 2 actions; define A, B, C.

### Compute

```math

G(\pi,\tau) = \mathbb{E}_{q(o_\tau|\pi)}\big[D_{\mathrm{KL}}(q(s_\tau\mid o_\tau,\pi)\,\|\,q(s_\tau\mid\pi))\big] + D_{\mathrm{KL}}(q(o_\tau\mid\pi)\,\|\,p(o_\tau))

```

Enumerate policies (length 1–2), show numerical values of terms and softmax selection.

