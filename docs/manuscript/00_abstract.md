# Abstract {#sec:abstract}

Active inference joins probabilistic state estimation, preference-sensitive
decision making, and uncertainty reduction in a single generative-model
workflow. This paper presents a validated Python implementation whose public
discrete model uses $A$, $B$, $C$, $D$, and $E$ arrays, whose dispatcher
implements variational, mean-field, and particle-sampling paths, and whose
continuous model performs precision-weighted generalized-coordinate updates.

The implementation is deliberately constrained to behaviors that can be
validated from executable code. The configured experiment has
{{MODEL_STATES}} latent states, {{MODEL_OBSERVATIONS}} observations,
{{MODEL_ACTIONS}} actions, a planning horizon of {{HORIZON}}, and seed
{{SEED}}. The manuscript builder generates five figures and records their
hash-linked build manifest. The result is a reproducible bridge between
active-inference formalisms and installable research software.

The contribution is an engineering and validation specification rather than a
claim of biological equivalence. Probability vectors are normalized at every
inference boundary; matrices are checked for shape, finiteness,
non-negativity, and stochasticity; configuration keys are consumed or
rejected; state persistence is versioned; and documented examples execute
against the package exported as `cognitive`.

**Keywords:** active inference, Bayesian inference, generative models,
partially observable decision processes, generalized coordinates,
reproducible research.
