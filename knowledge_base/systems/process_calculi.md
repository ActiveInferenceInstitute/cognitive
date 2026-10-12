---
title: Process Calculi
description: Formal languages with a small set of operators for describing concurrent, communicating systems, including CSP, CCS and the pi-calculus, whose behaviour is given by transitions and compared by bisimulation.
type: concept
status: stable
created: 2026-10-12
updated: 2026-10-12
tags:
  - concurrency
  - formal_methods
  - computation
semantic_relations:
  - type: relates
    links:
      - "[[message_passing]]"
      - "[[temporal_logic]]"
      - "[[distributed_control]]"
      - "[[coordination]]"
---

# Process Calculi

## Overview
A process calculus, also called a process algebra, is a formal language for describing systems made of independent components that run concurrently and interact by communication. Instead of modelling a program as a function from inputs to outputs, it models it as a process that can perform a sequence of actions, some of which synchronise with other processes. The main families are Hoare's Communicating Sequential Processes (CSP), Milner's Calculus of Communicating Systems (CCS) and the pi-calculus for systems whose communication structure changes over time. They are used to specify protocols and concurrent software and to prove properties of them.

## Core constructs
Using CCS notation (Milner, 1980), a process is built from:
- the inactive process $0$;
- action prefix $a.P$, which performs action $a$ and then behaves as $P$;
- choice $P+Q$;
- parallel composition $P\mid Q$;
- restriction $P\setminus a$, which hides the channel $a$;
- recursion or named definitions.

Actions come in complementary pairs $a$ and $\bar a$ (output and input on a channel), and a silent action $\tau$ stands for an internal step. The central rule is communication: if $P\xrightarrow{a}P'$ and $Q\xrightarrow{\bar a}Q'$ then $P\mid Q\xrightarrow{\tau}P'\mid Q'$. These rules define a labelled transition system for each process.

## Mobility
The pi-calculus (Milner, Parrow and Walker, 1992) lets processes send channel names as messages. An output $\bar x\langle y\rangle.P$ and an input $x(z).Q$ synchronise to give $P\mid Q\{y/z\}$, and the restriction $(\nu x)P$ creates a fresh private name. Since a received name can be used for later communication, the connection pattern between processes can change during execution, which a fixed-topology calculus cannot express directly. Milner (1999) gives a textbook account.

## Behavioural equivalence
Two processes are compared by what an observer can see, not by their syntax. Strong **bisimilarity** is the largest relation $\mathcal{R}$ such that whenever $P\,\mathcal{R}\,Q$ and $P\xrightarrow{\alpha}P'$, there is $Q'$ with $Q\xrightarrow{\alpha}Q'$ and $P'\,\mathcal{R}\,Q'$, and symmetrically for $Q$. Weaker equivalences ignore internal $\tau$ steps. CSP began as a language of sequential processes that communicate by synchronous input and output (Hoare, 1978), and its later form uses named channels and is given semantics in terms of recorded traces and failures (Hoare, 1985).

## Relation to Active Inference
The connection is a loose analogy. In multi-agent formulations of Active Inference, agents exchange messages and their interfaces are the Markov blankets that separate internal from external states, which resembles processes interacting only through channels. Process calculi, however, are specification languages for discrete concurrent behaviour, whereas Active Inference describes inference in probabilistic generative models, and no result in either field depends on the other.

## Related concepts
- [[message_passing]] — communication between components, in the inference sense
- [[temporal_logic]] — logics for stating properties of behaviour over time
- [[distributed_control]] — control of many interacting components
- [[coordination]] — alignment of actions among agents

## References
- Hoare, C. A. R. (1978). Communicating sequential processes. Communications of the ACM, 21(8), 666-677. https://doi.org/10.1145/359576.359585
- Hoare, C. A. R. (1985). Communicating Sequential Processes. Prentice Hall International.
- Milner, R. (1980). A Calculus of Communicating Systems (Lecture Notes in Computer Science, 92). Springer. https://doi.org/10.1007/3-540-10235-3
- Milner, R., Parrow, J., & Walker, D. (1992). A calculus of mobile processes, I. Information and Computation, 100(1), 1-40. https://doi.org/10.1016/0890-5401(92)90008-4
- Milner, R., Parrow, J., & Walker, D. (1992). A calculus of mobile processes, II. Information and Computation, 100(1), 41-77. https://doi.org/10.1016/0890-5401(92)90009-5
- Milner, R. (1999). Communicating and Mobile Systems: The Pi Calculus. Cambridge University Press.
