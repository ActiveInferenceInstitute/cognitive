---
title: "Agency, Autonomy, and Free Will Under the FEP"
type: concept
status: active
created: 2025-01-01
updated: 2025-02-06
tags:
  - free_energy_principle
  - free_will
  - agency
  - autonomy
  - counterfactual_depth
  - compatibilism
semantic_relations:
  - type: foundation
    links:
      - [[knowledge_base/free_energy_principle/mathematics/expected_free_energy|Expected Free Energy]]
      - [[knowledge_base/free_energy_principle/mathematics/markov_blankets|Markov Blankets]]
  - type: relates
    links:
      - [[mind_body_problem|Mind-Body Problem]]
      - [[epistemology|Epistemology]]
      - [[knowledge_base/free_energy_principle/cognitive/decision_making|Decision Making]]
      - [[knowledge_base/free_energy_principle/cognitive/consciousness|Consciousness]]
---

# Agency, Autonomy, and Free Will Under the FEP

## Overview

The question of free will -- whether organisms (and potentially artificial agents) genuinely choose their actions or are determined by prior causes -- takes on a distinctive character under the Free Energy Principle. The FEP provides formal definitions of agency, autonomy, and counterfactual depth that illuminate what it means to be a "free" agent and how degrees of freedom emerge from deterministic dynamics.

## Agency Under the FEP

### What Makes a System an Agent?

Under the FEP, agency is not a mystical property but a structural one. A system is an agent if it has:

1. **A Markov blanket**: It is distinguishable from its environment
2. **Active states**: It can influence external states (not just passively receive)
3. **A generative model**: Its internal states parameterize beliefs about external states
4. **Policy selection**: Its actions are selected to minimize expected free energy

The degree of agency depends on the complexity of the generative model:

```
Minimal agency:     Thermostat -- single state, reactive, no planning
Simple agency:      Bacterium -- chemotaxis, simple preferences, no temporal depth
Complex agency:     Mammal -- hierarchical model, planning, counterfactual reasoning
Full agency:        Human -- deep temporal models, meta-cognition, narrative self
```

### The Agency Continuum

Agency is not binary but exists on a continuum defined by the **temporal depth** and **counterfactual richness** of the generative model:

```
Temporal depth: How far into the future the agent plans
Counterfactual richness: How many alternative scenarios the agent can entertain
```

```
Reactive:           depth = 0, alternatives = 0  (thermostat)
Anticipatory:       depth > 0, alternatives = 0  (simple prediction)
Deliberative:       depth > 0, alternatives > 0  (planning)
Reflective:         depth > 0, alternatives > 0, + self-model  (meta-cognition)
Narratively free:   depth >> 0, alternatives >> 0, + narrative self  (human agency)
```

## Autonomy and Self-Determination

### Formal Definition of Autonomy

An autonomous system, under the FEP, is one whose dynamics are primarily driven by its internal states rather than external perturbations:

```
Autonomy = I(a; mu) / I(a; psi)
         = mutual information between actions and internal states
           / mutual information between actions and external states
```

High autonomy: Actions are driven by internal beliefs and preferences
Low autonomy: Actions are driven by external forces

### Self-Determination Through Prior Preferences

Under active inference, an agent's behavior is determined by its prior preferences `p(o)`:

```
a* = argmin_a G(a) = argmin_a {-Epistemic_value(a) - E_q[ln p(o)]}
```

The agent selects actions that lead to preferred observations. In this sense, the agent is "self-determined" -- its behavior follows from its own prior preferences, not from externally imposed rewards.

But where do these preferences come from?
- **Genetically encoded**: Evolved through natural selection (homeostatic setpoints)
- **Developmentally shaped**: Acquired during critical periods
- **Culturally learned**: Internalized from social environment
- **Self-chosen**: Higher-order preferences about what preferences to have (meta-preferences)

The last category is crucial for genuine autonomy: an agent that can reflect on and modify its own preferences has a deeper form of self-determination.

## Counterfactual Depth and Freedom

### The Role of Counterfactuals

The capacity for counterfactual reasoning -- imagining what WOULD happen under different choices -- is central to the FEP account of free will:

```
Counterfactual depth = number of alternative policies the agent evaluates
                     = |{pi_1, pi_2, ..., pi_N}|  in EFE computation
```

An agent with deeper counterfactual reasoning:
1. Considers more alternative actions
2. Simulates longer consequences
3. Evaluates both epistemic and pragmatic implications
4. Can choose based on simulated rather than actual experience

### Freedom as Counterfactual Sensitivity

A "free" action, under this view, is one that is **sensitive to counterfactual evaluation** -- the agent genuinely considered alternatives and selected this action based on its expected consequences:

```
Free action: P(a) = sigma(-gamma * G(a)) where multiple alternatives had significant G
Unfree action: P(a) = 1 (no alternatives considered, habitual, or externally forced)
```

This definition captures the intuition that:
- Reflexive actions are less "free" (no counterfactual evaluation)
- Deliberated actions are more "free" (extensive counterfactual evaluation)
- Coerced actions are unfree (external constraints remove alternatives)

## Compatibilism and the FEP

### Determinism and Agency

The FEP operates within deterministic dynamics (modulo stochastic noise). The flow equations fully determine the system's trajectory:

```
dx/dt = f(x) + sigma * xi(t)
```

Yet the system exhibits what looks like "choice" -- it selects among policies based on expected free energy. How?

### The Compatibilist Resolution

The FEP naturally supports **compatibilism** -- the view that free will is compatible with determinism:

1. **The agent's actions are determined** by its internal states and dynamics
2. **The internal states encode beliefs and preferences** through the generative model
3. **Actions are selected through evaluation** of expected consequences (EFE)
4. **The agent could have done otherwise** in the counterfactual sense: if the internal states had been different (different beliefs, different preferences), different actions would have been selected

Free will is not the absence of causation but the presence of the RIGHT KIND of causation: actions caused by the agent's own beliefs, preferences, and evaluations, rather than by external forces.

```
Free: a = f(mu, q(s), p(o), G(pi))  -- action caused by internal model
Unfree: a = f(psi)                    -- action caused by external force
```

### Levels of Freedom

The FEP suggests multiple levels of freedom:

**Level 1: Freedom of action**
Can the agent select among different motor outputs?
```
a in A where |A| > 1  (multiple actions available)
```

**Level 2: Freedom of belief**
Can the agent update its beliefs in response to evidence?
```
dq(s)/dt = -nabla_q F != 0  (beliefs can change)
```

**Level 3: Freedom of preference**
Can the agent modify its own prior preferences?
```
dC/dt = learning_rule(experience)  (preferences can be updated)
```

**Level 4: Freedom of model**
Can the agent change the structure of its generative model?
```
m* = argmin_m F[q | m]  (model selection)
```

Higher levels represent deeper forms of freedom. A thermostat has level 1 freedom. A human has all four levels.

## The Will as Policy Precision

### Willpower and Gamma

The FEP offers a formal account of **willpower** through the precision parameter gamma:

```
P(pi) = sigma(-gamma * G(pi))
```

**Strong will** (high gamma): The agent commits firmly to its best policy, resisting temptation (alternative policies with short-term appeal but high long-term EFE).

**Weak will** (low gamma): The agent is easily swayed by alternative policies, even those with higher EFE.

**Akrasia** (weakness of will): Acting against one's own best judgment is modeled as:

```
G(pi_best) < G(pi_chosen)  but gamma is low enough that P(pi_chosen) > P(pi_best)
```

The agent "knows" the best policy but selects a suboptimal one because policy precision is insufficient.

### Dopamine and Volition

Since dopamine encodes policy precision (gamma), dopaminergic dysfunction directly affects volition:

- **Parkinson's disease** (low DA): Difficulty initiating action (low gamma -> flat policy distribution)
- **Addiction** (aberrant DA): Excessive commitment to drug-seeking policies (high gamma for one policy)
- **Depression** (low DA): Loss of motivation (low gamma -> nothing seems worth doing)

## The Illusion of Free Will?

### The FEP Position

The FEP does not claim that free will is an illusion. Instead, it provides a **deflationary** account: free will is real, but it is not what dualists think it is. It is not an uncaused cause or a supernatural intervention. It is the computational process of evaluating policies under a generative model and selecting based on expected consequences.

This is "real" free will because:
1. The agent genuinely evaluates alternatives (counterfactual reasoning)
2. The selection is based on the agent's own model (self-determination)
3. Different internal states would have produced different choices (modal freedom)
4. The process involves information integration across time (not mere reaction)

This is "deflationary" because:
1. It does not require libertarian free will (uncaused causation)
2. It is implemented by ordinary physical dynamics
3. It can be described mathematically (no mysterious "gap" in causation)
4. It admits of degrees (more or less free, not absolutely free or unfree)

## Moral Responsibility

### FEP-Informed Responsibility

If agency admits of degrees, so does moral responsibility:

```
Responsibility proportional to:
  - Temporal depth of deliberation
  - Counterfactual richness of evaluation
  - Accuracy of the generative model (did the agent know the consequences?)
  - Precision of policy selection (did the agent choose deliberately?)
  - Freedom from external coercion (were alternatives genuinely available?)
```

This provides a principled basis for gradations of responsibility:
- Children: Reduced temporal depth -> reduced responsibility
- Mental illness: Distorted generative model -> modified responsibility
- Coercion: Removed alternatives -> reduced responsibility
- Addiction: Aberrant precision -> modified responsibility
- Fully deliberative adult: Full counterfactual depth -> full responsibility

## Key References

1. Friston, K., Schwartenbeck, P., FitzGerald, T., Moutoussis, M., Behrens, T., & Dolan, R. J. (2014). The anatomy of choice: dopamine and decision-making. *Philosophical Transactions of the Royal Society B*, 369(1655), 20130481.
2. Seth, A. K. (2021). *Being You*. Dutton. Chapters on free will and agency.
3. Clark, A. (2016). *Surfing Uncertainty*. Oxford University Press. Chapter 9.
4. Kirchhoff, M., & Froese, T. (2017). Where there is life there is mind. *Entropy*, 19(4), 169.
5. Dennett, D. C. (2003). *Freedom Evolves*. Viking.
6. Friston, K. J., Da Costa, L., & Parr, T. (2023). Some interesting observations on the free energy principle. *Entropy*, 25(8), 1216.
