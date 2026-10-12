---
title: Embedded Systems
description: Computing systems built into larger physical devices to perform dedicated functions under constraints of time, energy and memory, including real-time scheduling and the cyber-physical view of sensing and actuation.
type: concept
status: stable
created: 2026-10-12
updated: 2026-10-12
aliases:
  - Embedded Systems
tags:
  - embedded_computing
  - real_time
  - cyber_physical_systems
semantic_relations:
  - type: relates
    links:
      - "[[internet_of_things]]"
      - "[[edge_computing]]"
      - "[[control_systems]]"
      - "[[fault_tolerance]]"
      - "[[cybernetics]]"
      - "[[digital_twins]]"
---

# Embedded Systems

## Overview
An embedded system is a computer, typically a microcontroller or a small processor with its software, that is built into a larger device and dedicated to a particular function, such as engine control in a car, a pacemaker, a thermostat or a sensor node. Unlike a general-purpose computer, it is judged by how it behaves in its physical surroundings. Lee and Seshia (2017) define the subject through the modelling, design and analysis of cyber-physical systems, which integrate computing, networking and physical processes.

## Defining characteristics
- **Interaction with the physical world.** Sensors convert physical quantities into data and actuators convert computed commands into physical action, so the software is part of a feedback loop with the environment (see [[control_systems]]).
- **Timing as correctness.** Many embedded tasks must finish before a deadline. In a **hard** real-time system a missed deadline is a failure, and in a **soft** real-time system it only degrades quality. Lee (2008) argued that mainstream computing abstractions hide time and are therefore poorly matched to such systems.
- **Resource constraints.** Memory, processing power, energy and cost are limited, and many devices run on batteries for long periods.
- **Reliability.** Devices are expected to run unattended and sometimes in safety-critical roles (see [[fault_tolerance]]).
- **Concurrency.** Software must respond to interrupts and external events while other tasks run.

## Real-time scheduling
When several periodic tasks share one processor, a scheduling policy decides which runs. Liu and Layland (1973) analysed fixed-priority scheduling in which shorter-period tasks receive higher priority (rate-monotonic scheduling). For $n$ independent periodic tasks with deadlines equal to their periods, all deadlines are guaranteed to be met if the total processor utilisation $U$ satisfies
$$U\le n\left(2^{1/n}-1\right),$$
a bound that decreases towards $\ln 2\approx0.693$ as $n$ grows. The condition is sufficient, not necessary, so some task sets exceeding it are still schedulable.

## Design flow
Marwedel (2006) covers embedded system design through specification and modelling, hardware, system software, evaluation and validation, and the mapping of applications onto platforms, with energy and timing among the main design metrics. Embedded devices are the sensing and actuating end of the [[internet_of_things]], and computation placed close to them is the subject of [[edge_computing]].

## Relation to Active Inference
The relation is an analogy between architectures. An embedded controller runs a sense-compute-act loop under tight resource limits, which is structurally like the perception-action cycle that Active Inference ascribes to an agent. Implementations of Active Inference on robots and sensor nodes would be embedded systems and would face the timing and energy constraints described here, but the framework itself makes no claims about them.

## Related concepts
- [[internet_of_things]] — networks of embedded devices
- [[edge_computing]] — processing close to the data source
- [[control_systems]] — feedback loops closed through sensors and actuators
- [[fault_tolerance]] — reliability under component failure
- [[cybernetics]] — the study of control and communication in machines
- [[digital_twins]] — virtual counterparts of physical devices

## References
- Liu, C. L., & Layland, J. W. (1973). Scheduling algorithms for multiprogramming in a hard-real-time environment. Journal of the ACM, 20(1), 46-61. https://doi.org/10.1145/321738.321743
- Lee, E. A. (2008). Cyber physical systems: Design challenges. In 2008 11th IEEE International Symposium on Object and Component-Oriented Real-Time Distributed Computing (ISORC), 363-369. https://doi.org/10.1109/ISORC.2008.25
- Marwedel, P. (2006). Embedded System Design. Springer.
- Lee, E. A., & Seshia, S. A. (2017). Introduction to Embedded Systems: A Cyber-Physical Systems Approach (2nd ed.). MIT Press. https://ptolemy.berkeley.edu/books/leeseshia/
