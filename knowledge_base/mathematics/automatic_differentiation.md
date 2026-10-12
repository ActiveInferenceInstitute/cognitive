---
title: Automatic Differentiation
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - automatic_differentiation
  - gradients
  - backpropagation
semantic_relations:
  - type: foundation
    links:
      - "[[calculus]]"
  - type: relates
    links:
      - "[[optimization_algorithms]]"
      - "[[gradient_descent]]"
      - "[[numerical_methods]]"
wikidata: Q787371
---

# Automatic Differentiation

## Overview
Automatic differentiation (AD) computes exact derivatives of a function specified as a program by applying the chain rule to each elementary operation. It differs from symbolic differentiation (no expression swell) and from finite differences (no truncation error and no step-size choice), though it does incur floating-point rounding like any computation.

## Definition / Formalism
Any program is a composition of primitives, giving a computational graph. **Forward mode** propagates a tangent alongside each value, equivalently evaluating on dual numbers $x+\epsilon\,\dot x$ with $\epsilon^2=0$, so that $f(x+\epsilon\dot x)=f(x)+\epsilon\,f'(x)\dot x$. One forward pass yields one Jacobian-vector product $J v$. **Reverse mode** first runs the function forward, storing intermediate values, then propagates adjoints backward, yielding a vector-Jacobian product $v^{\top}J$ in one backward pass. For $f:\mathbb R^n\to\mathbb R$ the full gradient costs $n$ forward passes but a single reverse pass, at a small constant multiple of the cost of evaluating $f$ (the Baur-Strassen result), at the price of memory for the stored intermediates. Backpropagation in neural networks is reverse-mode AD.

Worked check: for $f(x)=\sin(x^2)+x$, dual-number evaluation at $x=2$ gives $f'(2)=4\cos 4+1\approx-1.6146$, matching the hand derivative $2x\cos(x^2)+1$. Higher-order derivatives follow by nesting modes, for example forward-over-reverse for Hessian-vector products.

## Role in Active Inference
Variational free energy is a differentiable function of beliefs and parameters, so perception, learning, and, in continuous time, action are gradients of $F$. Frameworks built on AD libraries (JAX, PyTorch) obtain these gradients automatically, which is also what makes deep and amortised active inference implementable at scale. This repository takes a different route: its continuous agent uses analytic gradients, and a regression test (`test_step_follows_free_energy_gradient`) checks the applied update against a finite-difference gradient of the reported free energy. AD would replace the hand-derived gradient and remove one class of derivation error, but would not change the mathematics.

## Related concepts
- [[optimization_algorithms]] — consumers of the derivatives AD produces
- [[gradient_descent]] — free-energy gradient flow in this vault
- [[numerical_methods]] — finite differences and their error behaviour
- [[numerical_stability]] — rounding and conditioning in derivative computation
- [[neural_network_theory]] — backpropagation as reverse-mode AD
- [[taylor_series]] — higher-order expansions behind forward-mode extensions

## References
- Griewank & Walther (2008), Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation.
- Baydin, Pearlmutter, Radul & Siskind (2018), Automatic differentiation in machine learning: a survey, JMLR 18.
