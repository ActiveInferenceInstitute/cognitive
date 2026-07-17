# Introduction {#sec:introduction}

Active inference is a useful computational vocabulary for systems that must
infer hidden causes while selecting actions under uncertainty. Its practical
value depends on an unbroken correspondence between the probability model,
the numerical operations, the configuration surface, and the evidence used to
describe the result. A mathematically attractive interface that accepts
invalid distributions or silently ignores configuration cannot provide that
correspondence.

This manuscript audits and reconstructs the repository around four hard
requirements: valid probability calculus, executable configuration, explicit
randomness, and reproducible evidence. The implementation boundary is the
public `cognitive` package and the two maintained agent families in
`code/Things/`. The research question is therefore operational:

> Can one configuration produce valid inference, deterministic evidence, and
> a rendered manuscript whose numerical claims come from the same run?

## Formal object

The discrete generative model is the tuple

$$
\mathcal{M} = (A, B, C, D, E),
$$ {#eq:generative_model}

where $A_{os}=P(o\mid s)$ is the observation likelihood,
$B_{s'sa}=P(s'\mid s,a)$ is the controlled transition tensor, $C_o$ is a
log-preference over observations, $D_s$ is the initial state prior, and
$E_a$ is the action prior. The implementation in
`code/tools/src/models/active_inference/generative_model.py` enforces these
conventions at construction time. In particular, each column of $A$ and each
state-column of every action slice of $B$ must sum to one.

Given an observation and prior, the exact categorical posterior is

$$
q(s\mid o) = \frac{P(o\mid s)q(s)}{\sum_j P(o\mid j)q(j)}.
$$ {#eq:posterior}

When an observation has zero likelihood under the model, the implementation
retains the normalized prior rather than manufacturing a non-probability.
This behavior makes the failure mode explicit and preserves a valid state for
the next update.

## Decision problem

For a policy sequence $\pi=(a_0,\ldots,a_{H-1})$, the dispatcher evaluates
each action sequence over the configured horizon and returns a distribution
over its first action:

$$
q(a_0) = \operatorname{softmax}\left(-\frac{G(a_0)}{\tau}\right),
$$ {#eq:policy_distribution}

where $G$ is expected free energy and $\tau>0$ is the configured temperature.
This marginalization is important: a policy distribution is not a raw vector
of energies, and a temperature is not a license to bypass normalization.

## Scope

The paper covers discrete categorical inference, a continuous generalized-
coordinate model, homeostatic control, a compact POMDP agent, matrix
utilities, deterministic visualization, node creation, link validation, and
benchmarks. It does not claim additional dispatcher modes beyond the exported
API. Related theories and biological interpretations are discussed in
[@friston2010free; @friston2017active; @parr2019active], while this work
focuses on the executable numerical contract.
