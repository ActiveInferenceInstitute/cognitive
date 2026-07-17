# Scope, Related Work, and Positioning {#sec:scope}

## Scholarship

The free-energy principle provides the broad theoretical setting for treating
perception and action as inference under a generative model
[@friston2010free]. Active inference develops that setting into policies that
trade preference satisfaction against uncertainty reduction
[@friston2017active; @parr2019active]. Bogacz presents a compact derivation of
the relationship between variational inference and neural computation
[@bogacz2017]. This repository uses those ideas as a computational design
constraint, not as evidence that a software agent reproduces a biological
mechanism.

The discrete model also sits within the established POMDP tradition, where a
belief state is a sufficient statistic for action selection under partial
observability [@kaelbling1998planning; @ross2008online]. The epistemic term
uses the information-gain perspective associated with Bayesian experimental
design [@lindley1956measure] and information theory [@mackay2003information].
The implementation chooses explicit finite categorical arrays so that these
relationships can be tested directly.

The continuous path follows generalized-coordinate treatments of temporal
state estimation and predictive processing [@friston2008hierarchical;
@friston2010free]. Its role here is a validated numerical model with explicit
observation and state precision, not a claim that one discretization is
canonical.

## Positioning

The contribution is at the boundary between computational cognitive science
and research software engineering. It makes four choices visible:

1. probability validity is a precondition for inference;
2. configuration is a typed input with an explicit rejection policy;
3. randomness is an input to reproducibility, not ambient process state;
4. documentation is built from runtime evidence and linked to source paths.

These choices complement mathematical treatments of variational inference
[@bishop2006pattern] and practical reproducibility guidance
[@peng2011reproducible]. They do not replace domain-specific model comparison,
empirical validation, or a full treatment of hierarchical and multi-agent
systems.

## Boundaries

The package currently exposes discrete categorical inference through
`ActiveInferenceDispatcher` and continuous inference through
`ContinuousActiveInference`. The dispatcher accepts only its exported
discrete policy type. Extensions should add a separately validated model and
tests rather than widening an enum without a corresponding numerical path.
The manuscript's claims are limited to the included configurations and
regression tests.
