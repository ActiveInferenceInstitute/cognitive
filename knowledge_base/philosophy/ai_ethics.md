---
title: AI Ethics
description: The study of moral questions raised by the design and use of artificial intelligence, including fairness, transparency, privacy, accountability and safety, and the principles and governance frameworks proposed for them.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - ethics
  - artificial_intelligence
  - governance
semantic_relations:
  - type: relates
    links:
      - "[[ai_safety]]"
      - "[[philosophy_of_ai]]"
      - "[[technology_ethics]]"
      - "[[applied_ethics]]"
      - "[[cognitive_security]]"
---

# AI Ethics

## Overview
AI ethics is the part of [[applied_ethics]] concerned with the moral issues raised by artificial intelligence systems and the people and institutions that build and deploy them. It covers both near-term questions about systems already in use and longer-term questions about advanced systems. It overlaps with [[ai_safety]], which focuses on preventing harmful behaviour, and with [[technology_ethics]] more generally.

## Main issue areas
- **Fairness and bias.** Systems trained on historical data can reproduce or amplify unequal treatment, and there are several competing formal definitions of fairness that cannot always be satisfied together.
- **Transparency and explainability.** People affected by automated decisions may need to understand how those decisions were reached, which is difficult for complex models.
- **Privacy and surveillance.** Data collection, inference of sensitive attributes and biometric identification.
- **Accountability.** Who is responsible when automated systems cause harm: designers, operators, deployers or users.
- **Autonomy and manipulation.** Persuasive and recommender systems may shape choices and beliefs, a concern shared with [[cognitive_security]].
- **Safety and alignment.** Ensuring that systems pursue the objectives their designers intend. Russell (2019) argues for building systems that are uncertain about human preferences and defer to them.
- **Labour, power and environment.** Effects on employment, concentration of power and the resource costs of large models.

## Principles and governance
Many organisations have published AI principles. Jobin, Ienca and Vayena (2019) reviewed a large set of such guidelines and found convergence on a small number of themes, namely transparency, justice and fairness, non-maleficence, responsibility and privacy, but divergence in interpretation and implementation. Floridi and colleagues (2018) proposed a framework that adds explicability to the four bioethical principles of beneficence, non-maleficence, autonomy and justice. Binding regulation has begun to appear in several jurisdictions, and intergovernmental bodies have issued recommendations.

## Relation to Active Inference
Agents built with explicit generative models represent beliefs and preferences in a form that can in principle be inspected, which some proponents suggest could help with transparency and with specifying objectives. These are design proposals and have not been demonstrated at scale. Active Inference also supplies a vocabulary for modelling how agents influence one another's beliefs, relevant to manipulation concerns. It does not supply a normative answer to any of the questions above.

## Related concepts
- [[ai_safety]] — preventing harmful behaviour of AI systems
- [[philosophy_of_ai]] — conceptual questions about machine intelligence
- [[technology_ethics]] — ethics of technology broadly
- [[applied_ethics]] — the parent field
- [[cognitive_security]] — protecting belief formation from manipulation
- [[artificial_intelligence]] — the technology at issue

## References
- Jobin, A., Ienca, M., & Vayena, E. (2019). The global landscape of AI ethics guidelines. Nature Machine Intelligence, 1, 389-399.
- Floridi, L., et al. (2018). AI4People: An ethical framework for a good AI society. Minds and Machines, 28, 689-707.
- Russell, S. (2019). Human Compatible: Artificial Intelligence and the Problem of Control. Viking.
