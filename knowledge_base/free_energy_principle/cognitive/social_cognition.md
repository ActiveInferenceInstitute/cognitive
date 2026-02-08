---
title: "Social Cognition as Mutual Active Inference"
type: concept
status: active
created: 2025-01-01
updated: 2025-02-06
tags:
  - free_energy_principle
  - social_cognition
  - theory_of_mind
  - communication
  - shared_narratives
  - active_inference
semantic_relations:
  - type: foundation
    links:
      - [[knowledge_base/free_energy_principle/mathematics/expected_free_energy|Expected Free Energy]]
      - [[knowledge_base/free_energy_principle/mathematics/markov_blankets|Markov Blankets]]
  - type: relates
    links:
      - [[perception|Perception]]
      - [[decision_making|Decision Making]]
      - [[consciousness|Consciousness]]
      - [[knowledge_base/free_energy_principle/systems/emergence|Emergence]]
---

# Social Cognition as Mutual Active Inference

## Overview

Social cognition -- understanding and interacting with other minds -- is perhaps the most challenging domain for cognitive science. Under the Free Energy Principle, social cognition is formalized as **mutual active inference**: each agent has a generative model that includes other agents, and social interaction emerges from the coupled minimization of free energy across agents.

This framework provides principled accounts of theory of mind, communication, cultural evolution, and the emergence of shared social realities.

## Theory of Mind as Generative Modeling

### Modeling Other Minds

Theory of mind (ToM) -- the ability to attribute mental states to others -- is, under the FEP, the process of including other agents within one's own generative model:

```
p(o_self, s_world, s_other) = p(o_self | s_world) * p(s_world | s_other) * p(s_other)
```

Where `s_other` represents the hidden states of the other agent, including their:
- **Beliefs**: `q_other(s_world)` -- what they think the world is like
- **Preferences**: `p_other(o)` -- what they want
- **Policies**: `pi_other` -- what they plan to do
- **Precision**: `gamma_other` -- how confident they are

**Inference about others** is perception applied to social signals:
```
q(s_other) = argmin_q F[q, o_social]
```

Where `o_social` includes facial expressions, body language, speech, and behavior.

### Recursive Modeling

True social cognition involves recursive modeling -- not just "what does the other think?" but "what does the other think I think?":

```
Level 0: My beliefs about the world: q(s_world)
Level 1: My beliefs about your beliefs: q(q_other(s_world))
Level 2: My beliefs about your beliefs about my beliefs: q(q_other(q_self(s_world)))
...
```

This recursion is bounded by computational resources, and the depth of recursive modeling may vary across individuals and contexts. Autism spectrum conditions may involve reduced depth of recursive social modeling.

## Communication as Active Inference

### Speaking as Active Inference

Under the FEP, speech (and communication more broadly) is **active inference** -- the speaker acts to reduce the listener's uncertainty:

```
G_speaker(utterance) = -E_q[D_KL[q_listener(s | utterance) || q_listener(s)]]
                       - E_q[ln p_speaker(o_listener)]
```

The speaker selects utterances that:
1. **Maximize information transfer**: Reduce the listener's uncertainty about intended meaning (epistemic value)
2. **Achieve communicative goals**: Move the listener's beliefs toward desired states (pragmatic value)

### Listening as Perceptual Inference

The listener performs perceptual inference about the speaker's intended meaning:

```
q_listener(meaning) = argmin_q F[q, utterance]
```

The listener infers the hidden meaning behind the observable utterance using their generative model of the speaker.

### Pragmatic Communication

The Gricean maxims of conversation emerge naturally from EFE minimization:

| Gricean Maxim | FEP Derivation |
|--------------|----------------|
| **Quality** (be truthful) | Accurate generative model minimizes free energy |
| **Quantity** (be informative) | Maximize information gain (epistemic value) |
| **Relevance** (be relevant) | Target information that reduces listener's free energy |
| **Manner** (be clear) | Reduce ambiguity in the generative model |

## Shared Generative Models and Culture

### Generalized Synchrony

When two agents interact extensively, their generative models tend to **synchronize**:

```
Over time: q_A(s) -> q_B(s) and q_B(s) -> q_A(s)
```

This is because each agent acts to make the world conform to their predictions, and the other agent is part of that world. The result is a **shared generative model** -- agents come to perceive and understand the world in similar ways.

### Cultural Norms as Shared Priors

Culture, under the FEP, consists of shared prior beliefs and preferences that coordinate behavior across individuals:

```
Culture = {C_shared, D_shared, A_shared}
```

Where:
- `C_shared` = shared preferences (values, norms, aesthetics)
- `D_shared` = shared prior beliefs (cosmology, history, identity)
- `A_shared` = shared observation models (language, categories, concepts)

Cultural transmission is the process of aligning new members' generative models with the shared model through active inference (teaching, storytelling, ritual).

### Institutions as Collective Markov Blankets

Social institutions (governments, corporations, religions) can be understood as collective entities with their own Markov blankets:

```
Institution:
  Internal states: Organizational structure, procedures, knowledge
  Sensory states: Information intake (reports, surveys, feedback)
  Active states: Policies, outputs, communications
  External states: The broader social environment
```

Institutions minimize collective free energy by coordinating the behavior of their constituent agents. They persist because they maintain their Markov blanket -- their organizational identity -- through active inference at the collective level.

## Empathy and Emotional Inference

### Empathy as Inference About Interoceptive States

Empathy is the process of inferring another person's interoceptive (emotional, bodily) states:

```
q(s_other_intero) = inference about other's internal bodily states
```

This requires a generative model that maps observable social signals to hidden interoceptive states:

```
p(facial_expression | emotional_state) -- how emotions produce expressions
p(behavior | motivational_state) -- how motivations produce behavior
```

### Emotional Contagion and Shared Affect

Emotional contagion -- catching others' emotions -- is a consequence of coupled active inference:

1. Other person is sad -> produces sad facial expression, tone of voice
2. Observer perceives these signals and infers sadness in the other
3. Observer's self-model updates to include sadness-consistent interoceptive predictions
4. These predictions drive interoceptive changes through active inference (the body conforms to predictions)
5. Observer actually becomes sad

This is **embodied simulation**: understanding others' emotions by simulating them in one's own body through active inference.

## Social Learning and Imitation

### Imitation as Free Energy Minimization

Imitation can be understood as minimizing the difference between one's own actions and another's observed actions:

```
F_imitation = D_KL[q(a_self) || p(a_self | a_other)]
```

The agent's action distribution is updated to match the observed model's actions. This requires a generative model that maps observed actions to the agent's own motor representations (mirror system).

### Social Conformity

Social conformity -- adjusting behavior to match group norms -- is a consequence of including the social group in one's generative model:

```
p(o_social | s_self = deviant) << p(o_social | s_self = conforming)
```

Deviating from the group generates high social prediction errors (others' surprised or disapproving reactions), which increases free energy. Conforming reduces these social prediction errors.

## Adversarial and Cooperative Dynamics

### Cooperative Active Inference

In cooperative interactions, agents' prior preferences are aligned:

```
p_A(o) ~ p_B(o) (similar preferences)
```

Both agents benefit from reducing each other's uncertainty and achieving shared goals. Communication is cooperative, and trust (high precision on social signals) develops over time.

### Adversarial Active Inference

In adversarial interactions, agents' preferences conflict:

```
p_A(o) conflicts with p_B(o)
```

Each agent may actively try to increase the other's uncertainty (deception, misdirection) or prevent the other from achieving preferred observations (competition).

**Deception** is action designed to install false beliefs in another agent:
```
a_deceptive = argmin_a { F_self + D_KL[q_other(s_true) || q_other(s_false)] }
```

The deceiver acts to make the other agent's beliefs about the world diverge from reality.

## Key References

1. Friston, K., & Frith, C. D. (2015). A duet for one. *Consciousness and Cognition*, 36, 390-405.
2. Vasil, J., Badcock, P. B., Constant, A., Friston, K., & Ramstead, M. J. D. (2020). A world unto itself: Human communication as active inference. *Frontiers in Psychology*, 11, 417.
3. Veissiere, S. P., Constant, A., Ramstead, M. J. D., Friston, K. J., & Kirmayer, L. J. (2020). Thinking through other minds: A variational approach to cognition and culture. *Behavioral and Brain Sciences*, 43, e90.
4. Moutoussis, M., Fearon, P., El-Deredy, W., Dolan, R. J., & Friston, K. J. (2014). Bayesian inferences about the self (and others): A review. *Consciousness and Cognition*, 25, 67-76.
5. Parr, T., Pezzulo, G., & Friston, K. J. (2022). *Active Inference*. MIT Press. Chapter 11.
