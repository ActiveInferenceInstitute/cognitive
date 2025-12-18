---

title: Communication

type: concept

status: stable

tags:
  - communication
  - information
  - pragmatics

semantic_relations:

  - type: relates

    links:
      - information_processing
      - bayesian_brain
      - active_inference

---

# Communication

Communication can be framed as coordinated inference between senders and receivers. Messages are actions that shape the receiver’s observations to reduce uncertainty (epistemic value) and align preferences (pragmatic value).

## Information-theoretic view

```math

I(M;R) = H(R) - H(R|M)\,,\quad G = \mathbb{E}[\text{information gain}] + \mathbb{E}[\text{preference fulfillment}]

```

Designing messages to maximize mutual information while respecting costs/prior preferences aligns with minimizing expected free energy.

## Components

- Encoding policy: maps beliefs and goals to signal forms

- Decoding policy: updates beliefs from received signals

- Shared priors: conventions that enable efficient codes and disambiguation

See: [[information_processing]], [[bayesian_inference]], [[active_inference]].

