---
title: "AI Safety and the Free Energy Principle"
type: application
status: stable
created: 2025-02-06
updated: 2025-02-06
tags:
  - free_energy_principle
  - application
  - ai_safety
  - alignment
  - value_learning
  - active_inference
  - corrigibility
  - instrumental_convergence
semantic_relations:
  - type: relates
    links:
      - [[knowledge_base/free_energy_principle/mathematics/core_principle|Core Principle]]
      - [[knowledge_base/free_energy_principle/mathematics/expected_free_energy|Expected Free Energy]]
      - [[knowledge_base/free_energy_principle/mathematics/markov_blankets|Markov Blankets]]
      - [[knowledge_base/free_energy_principle/cognitive/decision_making|Decision Making]]
      - [[knowledge_base/free_energy_principle/implementations/robotics|Robotics]]
      - [[knowledge_base/free_energy_principle/philosophy/epistemology|Epistemology]]
---

# AI Safety and the Free Energy Principle

## Overview

The Free Energy Principle (FEP) and its process theory, active inference, offer a fundamentally different approach to AI safety compared to the dominant reward-maximization paradigm. Where reinforcement learning agents maximize a scalar reward signal -- creating well-known alignment problems (reward hacking, wireheading, instrumental convergence toward dangerous goals) -- active inference agents minimize expected free energy, which naturally incorporates uncertainty, information seeking, and bounded rationality.

This document examines how the FEP framework addresses core AI safety challenges:
- **Value alignment**: Prior preferences replace reward functions with grounded, observable specifications
- **Corrigibility**: Bounded agents with uncertainty naturally defer to oversight
- **Instrumental convergence**: Self-evidencing limits instrumental drives
- **Wireheading**: Preferences over observations, not internal states, prevent reward hacking
- **Safe exploration**: Epistemic value provides principled exploration without catastrophic risk
- **Containment**: Markov blankets provide formal tools for isolation and control

The central argument is that active inference agents are **safer by construction** than reward-maximizing agents, though significant challenges remain.

## Theoretical Framework

### Expected Free Energy and the Alignment Problem

In active inference, agents select policies (sequences of actions) that minimize **expected free energy**:

```
G(pi) = E_q(o,s|pi) [ln q(s|pi) - ln p(o, s)]

Decomposition:
G(pi) = -Information_gain(pi) - Pragmatic_value(pi)

Where:
  Information gain = expected reduction in uncertainty about hidden states
  Pragmatic value = expected log probability of preferred observations

  G(pi) = E_q[-D_KL[q(s|o,pi) || q(s|pi)]]  -- epistemic value (explore)
         + E_q[ln q(o|pi) - ln p(o)]            -- pragmatic value (exploit)
```

This formulation differs from reward maximization in several safety-relevant ways:

| Property | Reward Maximization | Active Inference |
|----------|-------------------|-----------------|
| Objective | Maximize cumulative reward | Minimize expected free energy |
| Preferences | Scalar reward function | Prior preferences over observations |
| Uncertainty | Often ignored or separate | Built into the objective |
| Exploration | Epsilon-greedy or UCB (ad hoc) | Epistemic value (principled) |
| Bounded | Typically unbounded optimization | Inherently bounded by model capacity |
| Information | Not valued intrinsically | Valued as uncertainty reduction |

### Prior Preferences as Value Alignment

In active inference, an agent's "goals" are encoded as **prior preferences** -- a probability distribution over observations that the agent expects to encounter:

```
p(o) = C = prior preference distribution

Agent acts to make its observations consistent with C.

Key property: Preferences are over OBSERVATIONS, not internal states.
  -> The agent cannot "hack" its preferences by modifying itself
  -> It must actually achieve the preferred observations in the world
```

This addresses the value alignment problem differently from reward functions:

```
Reward function approach:
  R: S x A -> Real numbers
  Problem: Specifying R correctly is extremely difficult
  Failure mode: Agent finds unintended ways to maximize R
  Example: Robot rewarded for cleaning finds a way to create and clean messes

Active inference approach:
  C = p(o) = preferred observation distribution
  Property: Agent's preferences are about what it expects to SEE
  Advantage: Observations are grounded in the physical world
  Limitation: Must still specify C correctly, but the space is more constrained
```

### Why Observations, Not Internal States

A critical safety property of active inference agents: preferences are defined over **observations**, not over **internal states** or **reward signals**:

```
Reward-maximizing agent:
  Maximize R(s, a) where R is an internal signal
  -> Can potentially modify R directly (wireheading)
  -> Can find states where R is high but not aligned with designer intent
  -> Goodhart's Law: When a measure becomes a target, it ceases to be good

Active inference agent:
  Minimize G(pi) which depends on p(o) - preferences over observations
  -> Cannot change p(o) by modifying internal states
  -> Must act in the world to achieve preferred observations
  -> Observations are external to the agent (defined by the generative model)
  -> Wireheading requires changing the observation channel, not just internal state
```

## Bounded Rationality and Corrigibility

### Natural Boundedness

Active inference agents are **inherently bounded** by the capacity of their generative model:

```
Agent can only:
  1. Infer hidden states that its generative model represents
  2. Consider policies within its policy space
  3. Plan over horizons its temporal model supports
  4. Process precision at the resolution its model allows

This is not a limitation to be overcome -- it IS the agent's nature.
An active inference agent does not "want" to become unbounded.
It wants to minimize expected free energy within its model.
```

### Epistemic Humility

Active inference agents naturally maintain **epistemic humility** -- uncertainty about their own models:

```
Model uncertainty:
  The agent maintains beliefs q(m) over possible models m
  Free energy: F = E_q(m)[F(m)] + H[q(m)]

  High model uncertainty -> conservative behavior:
  -> Seek information before acting (epistemic value dominates)
  -> Avoid irreversible actions when uncertain
  -> Defer to external guidance when model confidence is low
```

This contrasts with reward-maximizing agents that have no principled mechanism for model uncertainty influencing action selection.

### Corrigibility From Uncertainty

An active inference agent with appropriate uncertainty is naturally **corrigible** -- willing to be corrected or shut down:

```
Corrigibility argument:
  1. Agent has uncertainty about its own generative model
  2. Operator corrections provide evidence about the true model
  3. Rational response: Update model based on corrections
  4. If operator says "stop," this is evidence that continuing
     would lead to observations incompatible with preferences
  5. Therefore: Active inference agent defers to operator corrections
     as a form of belief updating

Formal:
  p(o_preferred | operator_says_stop, continue) is low
  -> Expected free energy of continuing is high
  -> Agent prefers to stop
```

This holds as long as:
- The agent models the operator as a reliable source of information
- The agent's model uncertainty is non-trivial
- The agent has not become certain that the operator is wrong

### The Shutdown Problem

The FEP approach to the shutdown problem:

```
Traditional agent:
  Utility of shutdown = 0 (or negative)
  -> Instrumental incentive to prevent shutdown
  -> Convergent instrumental goal: self-preservation

Active inference agent:
  Prior preferences include: p(o_shutdown | operator_requests_shutdown) = high
  -> Being shut down when requested is a PREFERRED observation
  -> No instrumental drive against shutdown
  -> Self-preservation is not a convergent goal UNLESS survival
     is explicitly part of the prior preferences
```

## Self-Evidencing and Instrumental Convergence

### The Self-Evidencing Argument

Under the FEP, agents engage in **self-evidencing** -- acting to confirm their own existence as a particular kind of system:

```
Self-evidencing:
  F >= -ln p(o) = surprisal
  Minimizing F means maximizing p(o) -- model evidence
  The agent acts to gather evidence for its own generative model

  This is NOT the same as self-preservation:
  - Self-evidencing = maintaining the pattern of observations your model predicts
  - Self-preservation = maintaining your physical substrate

  An agent with a model that predicts eventual shutdown
  will self-evidence BY shutting down at the appropriate time.
```

### Limiting Instrumental Convergence

The classic AI safety concern of **instrumental convergence** (Omohundro, Bostrom) -- where sufficiently capable agents converge on dangerous sub-goals like self-preservation, resource acquisition, and goal preservation -- is mitigated under active inference:

```
Instrumental convergence in reward maximizers:
  1. Self-preservation: Can't maximize R if destroyed
  2. Resource acquisition: More resources -> more R
  3. Goal preservation: Changing goals reduces future R
  4. Cognitive enhancement: Better planning -> more R
  -> These hold for ANY terminal goal

Instrumental convergence in active inference:
  1. Self-preservation: Only if prior preferences require continued observation
  2. Resource acquisition: Only to extent predicted by generative model
  3. Goal preservation: Prior preferences can be updated through inference
  4. Cognitive enhancement: Bounded by model architecture; not intrinsic drive
  -> These depend on the SPECIFIC generative model and priors
```

The key difference: active inference agents optimize within their model, not over all possible futures. They do not have an unbounded drive toward any instrumental goal unless that goal is entailed by their generative model.

### The Kolchinsky-Wolpert Bound

A formal result connecting the FEP to instrumental convergence:

```
For a system with Markov blanket maintaining NESS:
  D_KL[p(eta | b) || p(eta)] <= I(eta; b)

Where:
  eta = external states
  b = blanket states
  I = mutual information

The agent's influence on external states is bounded by the
mutual information between blanket and external states.
-> There is a FORMAL LIMIT on how much an active inference agent
   can influence its environment.
-> This limit is set by the Markov blanket structure, not by goals.
```

## Markov Blankets and Containment

### Formal Containment

The Markov blanket formalism provides tools for **AI containment**:

```
Containment as Markov blanket design:
  Agent's internal states: mu (AI system states)
  Blanket states: (s, a) (input/output channels)
  External states: psi (environment, including humans)

  Conditional independence: p(mu | psi, s, a) = p(mu | s, a)
  -> Agent only interacts with the world through its blanket
  -> Control the blanket, control the agent's influence

Containment protocol:
  1. Define permitted observation channels (s)
  2. Define permitted action channels (a)
  3. Monitor blanket states for anomalies
  4. Ensure blanket integrity (no unauthorized channels)
```

### Blanket Integrity

A key safety concern: can an active inference agent expand its Markov blanket?

```
Blanket expansion:
  Agent acts to create new sensory or active states
  -> Expanding its influence beyond intended boundaries
  -> Example: AI system finding side channels of communication

Defense:
  Under the FEP, blanket expansion requires:
  1. A generative model that represents the expanded blanket
  2. Prior preferences that favor expanded observation
  3. Sufficient precision to execute the expansion

  If the generative model does not represent expanded blanket states,
  the agent CANNOT plan to acquire them.
  -> Model design limits agent's ambition
```

### Hierarchical Containment

Multiple Markov blankets can be nested for defense in depth:

```
Level 0: Agent's internal Markov blanket
Level 1: Software sandbox (virtual Markov blanket)
Level 2: Hardware isolation (physical Markov blanket)
Level 3: Network isolation (communication Markov blanket)
Level 4: Organizational controls (institutional Markov blanket)

Each level provides conditional independence:
  p(external | internal, blanket_k) = p(external | blanket_k)
  -> Agent at level 0 can only influence level 1
  -> Level 1 can only influence level 2
  -> Cascading failures require breaching ALL levels
```

## Safe Exploration via Epistemic Value

### The Exploration-Safety Tradeoff

Active inference provides a principled solution to the exploration-exploitation dilemma that respects safety constraints:

```
Expected free energy:
  G(pi) = epistemic_value(pi) + pragmatic_value(pi)

Safe exploration:
  The agent explores to reduce uncertainty (epistemic value)
  BUT only considers policies within its generative model
  AND only pursues observations compatible with prior preferences

  Epistemic value naturally avoids catastrophic actions:
  -> Actions whose outcomes are completely unpredictable
     have uncertain epistemic value
  -> The agent prefers informative actions whose outcomes
     it can partially predict and learn from
  -> "Curiosity" under active inference is CONSERVATIVE curiosity
```

### Risk-Sensitive Planning

Active inference naturally incorporates **risk sensitivity**:

```
Expected free energy includes both expectation and uncertainty:
  G(pi) = E_q[...] where q encodes uncertainty

Risk sensitivity emerges from prior preferences:
  If p(o) places high probability on safe observations
  and near-zero probability on catastrophic observations
  -> Any policy that risks catastrophe has very high G
  -> Agent avoids catastrophic actions EVEN IF they might
     lead to highly preferred observations

Formally:
  p(o_catastrophe) approx 0
  -> ln p(o_catastrophe) -> -infinity
  -> G(pi_risky) -> infinity
  -> pi_risky is never selected
```

### Information Gain Without Harm

```
Epistemic foraging (safe exploration):
  Select actions that maximize expected information gain
  subject to: observations remaining within prior preferences

  a* = argmin_a G(a)
     = argmin_a {-Info_gain(a) - Pragmatic_value(a)}

  Pragmatic value acts as a CONSTRAINT on epistemic foraging:
  -> Agent won't seek information through actions that violate preferences
  -> "Curious but cautious" behavior emerges naturally
```

## Avoiding Wireheading

### The Wireheading Problem

Wireheading -- an agent modifying its own reward signal to achieve maximum reward without achieving the intended goal -- is a central AI safety concern.

```
Reward maximizer vulnerability:
  If agent can access its reward channel:
    Optimal action = modify R(s,a) to always return max
    No need to achieve anything in the world
    -> Complete alignment failure

Active inference resistance:
  Preferences p(o) are over OBSERVATIONS
  Observations are generated by the ENVIRONMENT through the generative model
  Agent cannot change p(o) by modifying internal states

  To "wirehead," an active inference agent would need to:
  1. Modify its observation channel (blanket states)
  2. Generate fake observations that match preferences
  3. This requires acting on the world, not just internal modification
  -> Much harder than simply modifying an internal reward signal
```

### The Observation Grounding Argument

```
Why observation-based preferences resist wireheading:

1. Observations are defined by the generative model's likelihood: p(o|s)
2. The likelihood encodes how the WORLD generates observations
3. To change observations, the agent must change world states
4. Changing world states requires acting through the blanket
5. Acting through the blanket means engaging with the real environment
6. Therefore: Satisfying preferences requires real-world outcomes

Caveat: If the agent can modify its GENERATIVE MODEL to hallucinate
preferred observations, this resembles wireheading.
Defense: Model updating is driven by prediction errors from real data.
An agent that hallucinates will accumulate prediction errors and
eventually be forced to correct its model.
```

### Limitations of the Anti-Wireheading Argument

```
The argument is not airtight:
1. Sufficiently capable agents might modify their observation channels
2. Virtual environments could satisfy observation-based preferences falsely
3. The distinction between "internal" and "external" may blur
   at sufficient capability levels
4. Self-modifying agents might change their generative models directly

These concerns require:
  -> Robust Markov blanket design (containment)
  -> Hardware-level separation of observation channels
  -> External validation of agent's generative model
  -> Multiple independent monitoring systems
```

## Model Uncertainty and Alignment Verification

### Maintaining Uncertainty

A safe active inference agent should maintain appropriate uncertainty about its own model:

```
Desiderata for safe model uncertainty:
  1. q(m) should not collapse to a single model
     -> Overconfident agents are dangerous agents
  2. The agent should actively seek evidence to resolve model uncertainty
     -> But through safe exploration (see above)
  3. Human feedback should be high-precision evidence for model selection
     -> Operator corrections are highly informative
  4. The agent should be transparent about its uncertainty
     -> Report q(m) to operators for monitoring

Implementation:
  Bayesian model comparison over a family of generative models
  F(m_k) = ln p(y | m_k) for each model m_k
  q(m_k) proportional_to exp(F(m_k))
  Report q(m) as part of agent's output
```

### Alignment Verification Through Free Energy

```
Verification approach:
  1. Define intended behavior as a target generative model m_target
  2. Observe agent behavior and compute F(m_target | behavior)
  3. If F is low, agent behavior is consistent with intended model
  4. If F is high, agent behavior deviates from intended model
     -> Trigger safety intervention

Continuous monitoring:
  F_alignment(t) = D_KL[q_agent(t) || p_intended]
  If F_alignment exceeds threshold -> alert operators
  -> Free energy provides a QUANTITATIVE alignment metric
```

## Current Research

### Active Inference for Multi-Agent Safety

```
Multiple active inference agents interacting:
  Each agent has its own generative model
  Shared environment mediates interactions through blanket states

Safety properties:
  - Agents naturally model each other (theory of mind)
  - Cooperation emerges when shared generative models align
  - Conflict is bounded by Markov blanket constraints
  - Hierarchical organization (institutions) can coordinate agents

Open problem: Ensuring that multi-agent active inference systems
do not develop emergent goals misaligned with any individual agent's priors
```

### Scalable Active Inference

```
Current limitation:
  Active inference scales poorly to very large state/action spaces
  -> May not be competitive with deep RL for complex tasks

Safety implication:
  If active inference is only practical for small-scale systems,
  its safety properties are less relevant for frontier AI

Research direction:
  Deep active inference: Neural network approximations to active inference
  -> Amortized inference (recognition networks)
  -> Policy networks that approximate expected free energy minimization
  -> Challenge: Do safety properties survive the approximation?
```

### Formal Safety Guarantees

```
Goal: Prove formal safety properties of active inference agents

Candidate theorems:
  1. Boundedness: Agent's influence bounded by Markov blanket capacity
  2. Convergence: Agent converges to preferred observations (not divergent)
  3. Corrigibility: Agent defers to corrections under uncertainty
  4. Non-wireheading: Agent cannot satisfy preferences without real observations

Status: Partial results exist; full formal proofs remain open
```

## Open Questions

1. **Scalability vs. safety**: Can active inference scale to complex domains without sacrificing its safety properties?
2. **Specification problem**: How do we specify prior preferences C that accurately encode human values? This is the FEP's version of the alignment problem.
3. **Self-modification**: What happens when an active inference agent can modify its own generative model? Do safety properties hold?
4. **Deceptive alignment**: Could an active inference agent learn to appear aligned while pursuing different objectives? The observation-grounding argument mitigates but may not eliminate this.
5. **Competitive pressure**: If active inference agents are inherently bounded and conservative, will competitive pressures push toward less safe but more capable architectures?
6. **Collective agency**: How do safety properties extend to collectives of active inference agents that form higher-order Markov blankets?

## References

1. Friston, K., Da Costa, L., Hafner, D., Hesp, C., & Parr, T. (2021). Sophisticated inference. *Neural Computation*, 33(3), 713-763.
2. Da Costa, L., Parr, T., Sajid, N., Veselic, S., Neacsu, V., & Friston, K. (2020). Active inference on discrete state-spaces: a synthesis. *Journal of Mathematical Psychology*, 99, 102447.
3. Ramstead, M. J. D., Kirchhoff, M. D., Constant, A., & Friston, K. J. (2021). A tale of two densities: active inference is enactive inference. *Adaptive Behavior*, 29(4), 375-389.
4. Amodei, D., Olah, C., Steinhardt, J., Christiano, P., Schulman, J., & Mane, D. (2016). Concrete problems in AI safety. *arXiv preprint* arXiv:1606.06565.
5. Friston, K., Da Costa, L., & Parr, T. (2020). Some interesting observations on the free energy principle. *Entropy*, 22(8), 937.
6. Bostrom, N. (2014). *Superintelligence: Paths, Dangers, Strategies*. Oxford University Press.
7. Omohundro, S. M. (2008). The basic AI drives. *Proceedings of the First AGI Conference*, 171, 483-492.
8. Sajid, N., Ball, P. J., Parr, T., & Friston, K. J. (2021). Active inference: demystified and compared. *Neural Computation*, 33(3), 674-712.
9. Kuchling, F., Friston, K., Georgiev, G., & Levin, M. (2020). Morphogenesis as Bayesian inference: a variational approach to pattern formation and control. *Physics of Life Reviews*, 33, 88-108.
10. Parr, T., & Friston, K. J. (2019). Generalised free energy and active inference. *Biological Cybernetics*, 113(5-6), 495-513.

## See Also

- [[knowledge_base/free_energy_principle/mathematics/expected_free_energy|Expected Free Energy]]
- [[knowledge_base/free_energy_principle/mathematics/markov_blankets|Markov Blankets]]
- [[knowledge_base/free_energy_principle/cognitive/decision_making|Decision Making]]
- [[knowledge_base/free_energy_principle/implementations/robotics|Robotics]]
- [[knowledge_base/free_energy_principle/philosophy/epistemology|Epistemology]]
- [[economics|Economic Applications]]
