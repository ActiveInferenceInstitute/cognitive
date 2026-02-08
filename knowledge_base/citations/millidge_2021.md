---
title: "Whence the Expected Free Energy?"
authors:
  - "Beren Millidge"
  - "Alexander Tschantz"
  - "Christopher L. Buckley"
type: citation
status: verified
created: 2025-01-01
year: 2021
journal: "Neural Computation"
volume: 33
issue: 2
pages: 447-482
doi: "10.1162/neco_a_01354"
tags:
  - expected_free_energy
  - active_inference
  - derivation
  - objective_function
  - exploration_exploitation
semantic_relations:
  - type: foundational_for
    links:
      - [[knowledge_base/mathematics/expected_free_energy]]
  - type: extends
    links:
      - [[friston_2017_curiosity]]
      - [[da_costa_2020]]
  - type: cited_by
    links:
      - [[parr_pezzulo_friston_2022]]
---

# Whence the Expected Free Energy?

## Authors
- **Beren Millidge** (University of Edinburgh)
- **Alexander Tschantz** (University of Sussex)
- **Christopher L. Buckley** (University of Sussex)

## Publication Details
- **Journal**: Neural Computation
- **Year**: 2021
- **Volume**: 33
- **Issue**: 2
- **Pages**: 447-482
- **DOI**: [10.1162/neco_a_01354](https://doi.org/10.1162/neco_a_01354)

## Abstract
This paper provides a critical analysis of the expected free energy (EFE) -- the objective function used for policy selection in active inference. The authors examine multiple proposed derivations of the EFE from first principles and assess whether it can be derived uniquely from the free energy principle, or whether it requires additional assumptions beyond free energy minimization. They identify that the EFE requires specific assumptions about the form of the generative model (particularly about prior preferences) and that different decompositions of the EFE yield different behavioral properties. The paper clarifies the theoretical status of the EFE within the broader FEP framework.

## Key Contributions

### Critical Analysis of EFE Derivations
- **Multiple Derivations**: Reviews several proposed routes to the EFE
- **Assumptions Identified**: What assumptions each derivation requires
- **Uniqueness Question**: Is the EFE the only possible objective?
- **Relationship to VFE**: How EFE relates to variational free energy

### Decomposition Analysis
- **Risk + Ambiguity**: One decomposition (pragmatic + ambiguity)
- **Extrinsic + Intrinsic**: Another decomposition (reward + information gain)
- **Not Equivalent**: Different decompositions have different behavioral implications
- **Bound Properties**: Whether EFE is a proper bound on anything

### Theoretical Clarification
- **EFE is Not VFE**: Expected free energy is not simply variational free energy in the future
- **Additional Assumptions**: EFE requires assumptions about prior preferences
- **Generative Model Design**: How to specify generative models that yield the EFE
- **Alternative Objectives**: Other possible objectives for policy selection

## Core Concepts

### The Expected Free Energy
The EFE for policy pi at future time tau:
```
G(pi, tau) = E_{q(o,s|pi)} [ln q(s|pi) - ln p(o, s)]
```

### Proposed Derivations

**1. From variational free energy:**
Treating policies as parameters to optimize:
```
G(pi) = E_{q(o|pi)} [F(o, pi)]  # Expected VFE for predicted observations
```

Issue: This requires marginalizing over future observations, which changes the mathematical form.

**2. From KL divergence to preferred outcomes:**
```
G(pi) = KL[q(o|pi) || p(o)] + E_{q(o|pi)} [H[q(s|o, pi)]]
```

Issue: Requires specifying prior preferences p(o) separately.

**3. From Bayesian decision theory:**
```
G(pi) = -E_{q(o|pi)} [ln p(o|C)] + H[q(s|pi)] - H[q(s|o, pi)]
```

This decomposition yields risk (deviation from preferences) minus information gain.

### Key Insight: Prior Preferences
The EFE is not derivable from free energy minimization alone. It requires:
- **Prior preferences**: A distribution C over preferred observations
- **Generative model structure**: Specific factorization assumptions
- **Temporal assumptions**: How present inference relates to future outcomes

### Decompositions and Their Properties
| Decomposition | Components | Behavioral Property |
|---|---|---|
| Risk + Ambiguity | KL to prefs + conditional entropy | Exploitative + risk-averse |
| Info Gain + Utility | Mutual info + neg KL to prefs | Explorative + goal-seeking |
| Full EFE | All three terms | Balanced exploration-exploitation |

## Mathematical Formalism

### Full EFE Decomposition
```
G(pi) = E_q[-ln p(o|C)]        # Pragmatic value (negative)
       + E_q[H[p(o|s)]]         # Ambiguity
       - E_q[I(o; s|pi)]        # Epistemic value (negative)
```

### Alternative Decomposition
```
G(pi) = KL[q(o|pi) || p(o|C)]  # Risk
       + E_q[H[q(s|o, pi)]]    # Ambiguity (conditional entropy)
```

### Relationship to Variational Free Energy
VFE for current observations:
```
F = E_q[ln q(s) - ln p(o, s)]
```

EFE for future observations:
```
G(pi) = E_{q(o|pi)} E_{q(s|o,pi)} [ln q(s|pi) - ln p(o, s)]
```

The expectation over future observations is what distinguishes EFE from VFE and introduces the need for prior preferences.

## Impact and Applications

### Theoretical Understanding
- **Clarifies Foundations**: What the EFE is and is not
- **Guides Model Building**: How to specify generative models correctly
- **Identifies Limitations**: Where the EFE framework may need extension
- **Opens Alternatives**: Motivates exploration of alternative objectives

### For Practitioners
- **Model Design**: Practical guidance on specifying prior preferences
- **Decomposition Choice**: Which decomposition to use for which application
- **Parameter Sensitivity**: Understanding how model choices affect behavior

### For Theorists
- **Open Questions**: Fundamental issues requiring further work
- **Derivation Standards**: What counts as a valid derivation
- **Axiomatic Analysis**: Toward an axiomatic foundation for active inference

## Related Work

### Foundational Papers
- [[friston_2017_curiosity]] - Original EFE introduction
- [[da_costa_2020]] - Discrete active inference synthesis

### Comparisons
- [[sajid_2021]] - Active inference compared with RL
- [[tschantz_2020]] - RL through active inference
- [[buckley_2017]] - Mathematical review of continuous FEP

### Textbook
- [[parr_pezzulo_friston_2022]] - Comprehensive treatment

## Citations and Influence
This paper has been essential for clarifying the theoretical foundations of the expected free energy. It has been widely cited in discussions about the formal status of the EFE and has influenced how researchers specify and justify their active inference models. It represents important critical engagement with the foundations of the framework from within the active inference community.

## Reading Guide
1. **Introduction**: The role of EFE in active inference
2. **Background**: Variational free energy recap
3. **Derivations**: Multiple proposed routes to the EFE
4. **Decompositions**: Different ways to decompose the EFE
5. **Analysis**: What each derivation assumes and implies
6. **Discussion**: Open questions and future directions

---

> **Critical Foundation**: The definitive analysis of where the expected free energy comes from and what assumptions it requires.

---

> **Decomposition Analysis**: Shows that different EFE decompositions have different behavioral implications.

---

> **Honest Assessment**: Provides a rigorous, honest examination of the theoretical foundations from within the active inference community.
