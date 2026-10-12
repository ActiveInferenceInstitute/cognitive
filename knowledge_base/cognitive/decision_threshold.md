---
title: Decision Threshold
description: The criterion that determines when evidence is sufficient to commit to a choice, covering bounds in sequential sampling models, the signal detection criterion, speed-accuracy trade-offs and neural findings.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - decision_making
  - evidence_accumulation
  - psychophysics
semantic_relations:
  - type: relates
    links:
      - "[[decision_making]]"
      - "[[evidence_accumulation]]"
      - "[[signal_detection]]"
      - "[[reaction_time]]"
      - "[[policy_selection]]"
---

# Decision Threshold

## Overview
A decision threshold, or criterion, is the level of evidence at which an observer stops deliberating and commits to a response. The term has two related meanings. In sequential sampling models it is a bound that accumulated evidence must reach. In signal detection theory it is a criterion placed on a single decision variable that divides responses into "signal" and "noise". In both cases moving the threshold changes the balance between kinds of error and, where time matters, between speed and accuracy.

## Sequential sampling
In the drift-diffusion model (Ratcliff, 1978), evidence $x$ follows $dx = v\,dt + \sigma\,dW$ with drift rate $v$ and noise $\sigma$ until it reaches one of two absorbing bounds at $\pm a$. For an unbiased start point the error probability is $1/(1+e^{2va/\sigma^2})$ and the mean decision time is $(a/v)\tanh(av/\sigma^2)$ (Bogacz et al., 2006). Raising $a$ lowers errors and lengthens decisions, which gives the speed-accuracy trade-off. Moving the starting point instead biases the choice toward one option. Instructions that stress speed or accuracy are commonly modelled as changes in $a$.

## Signal detection criterion
In signal detection theory, an observer reports "signal" when the decision variable exceeds a criterion $c$. A lenient criterion raises hits and false alarms together, and a strict one lowers both, while sensitivity $d'$ is unaffected. Payoffs and prior probabilities of the options shift the criterion.

## Neural findings
Recordings from monkey parietal cortex during motion discrimination show firing rates that ramp up toward a roughly constant level before the response, consistent with accumulation to a bound (Roitman & Shadlen, 2002; Gold & Shadlen, 2007). The existence of a literal bound in neural activity, as opposed to a read-out rule, remains debated.

## Relation to Active Inference
In discrete Active Inference, belief updating accumulates evidence by iterative inference, and action follows from a softmax over expected free energy, whose precision plays a role akin to urgency. A fixed commitment threshold is not built in. It is added by the modeller as a rule, such as acting once the policy posterior exceeds a criterion. The correspondence with drift-diffusion is formal analogy, not identity.

## Related concepts
- [[evidence_accumulation]] — the process that the bound terminates
- [[signal_detection]] — the framework in which a criterion is defined
- [[reaction_time]] — the response latencies that bounds help explain
- [[decision_making]] — the broader field of choice
- [[policy_selection]] — the Active Inference counterpart of choosing among options

## References
- Ratcliff, R. (1978). A theory of memory retrieval. Psychological Review, 85, 59-108.
- Bogacz, R., Brown, E., Moehlis, J., Holmes, P., & Cohen, J. D. (2006). The physics of optimal decision making: A formal analysis of models of performance in two-alternative forced-choice tasks. Psychological Review, 113, 700-765.
- Gold, J. I., & Shadlen, M. N. (2007). The neural basis of decision making. Annual Review of Neuroscience, 30, 535-574.
- Roitman, J. D., & Shadlen, M. N. (2002). Response of neurons in the lateral intraparietal area during a combined visual discrimination reaction time task. Journal of Neuroscience, 22, 9475-9489.
