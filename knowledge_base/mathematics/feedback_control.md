---
title: Feedback Control
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - feedback_control
  - pid
  - transfer_function
  - good_regulator
aliases:
  - closed_loop_systems
semantic_relations:
  - type: foundation
    links:
      - "[[control_theory]]"
      - "[[state_space]]"
  - type: relates
    links:
      - "[[adaptive_control]]"
      - "[[homeostasis]]"
      - "[[motor_control]]"
---

# Feedback Control

## Overview
Feedback control uses a measurement of a system's output to compute a corrective input, reducing the effect of disturbances and model error. It is the central design pattern of control engineering and of biological regulation.

## Definition / Formalism
With reference $r$, output $y$ and error $e=r-y$, a PID controller computes

$$u(t)=K_pe(t)+K_i\!\int_0^te(\tau)\,d\tau+K_d\dot e(t).$$

In the Laplace domain with plant $G(s)$ and controller $C(s)$, the loop gain is $L=CG$ and the closed-loop map from reference to output is $T=L/(1+L)$. The closed loop is stable iff all roots of $1+L(s)=0$ have negative real part; the Nyquist criterion tests this from the frequency response of $L$.

Worked example: plant $G=1/(s+1)$ with proportional gain $K$ gives $T=K/(s+1+K)$, a single pole at $-(1+K)$, so higher gain speeds the response. The steady-state error to a unit step is $1/(1+K)$, nonzero for finite $K$; adding integral action drives it to zero. The sensitivity function $S=1/(1+L)$ and complementary sensitivity $T$ satisfy $S+T=1$, which forces a trade-off between disturbance rejection and noise sensitivity. In state-space form, state feedback $u=-Kx$ places the closed-loop eigenvalues of $A-BK$ when the pair $(A,B)$ is controllable, and an observer provides $\hat x$ when $(A,C)$ is observable. Time delays and unmodelled dynamics limit achievable gain. The internal model principle and the Conant-Ashby good regulator theorem state that a regulator must embody a model of what it regulates.

## Role in Active Inference
Active inference casts action as feedback: in the continuous formulation action descends the free-energy gradient, $\dot a=-\partial_aF$, which acts on sensory prediction errors in the way a reflex arc acts on proprioceptive error ("predictions, not commands"; Adams, Shipp and Friston, 2013). The set point is a prior belief about sensory input, not a separate reference signal. The good regulator theorem gives conceptual support for generative models inside regulators. Equivalence with classical designs holds in restricted settings (linear-Gaussian models give Kalman-filter-based control, and with suitable priors reduce to LQG-like structure), but general equivalence claims should be checked case by case.

## Related concepts
- [[control_theory]] — broader context including optimal and robust control
- [[adaptive_control]] — controllers that tune their own parameters
- [[feedback_loops]] — positive and negative loops in general systems
- [[homeostasis]] — biological negative feedback
- [[motor_control]] — feedback control of movement
- [[optimal_control]] — feedback laws derived from cost minimisation

## References
- Conant & Ashby (1970), Every good regulator of a system must be a model of that system, International Journal of Systems Science 1.
- Astrom & Murray, Feedback Systems: An Introduction for Scientists and Engineers.
- Adams, Shipp & Friston (2013), Predictions not commands: active inference in the motor system, Brain Structure and Function 218.
