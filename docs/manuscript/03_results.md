# Results {#sec:results}

## Runtime configuration

The canonical build uses the model dimensions shown in [@tbl:model_dimensions]
and evaluates {{POLICY_COUNT}} policies at horizon {{HORIZON}}. The observed
sequence is {{OBS_SEQUENCE}}. These values are read from
`docs/manuscript/config.yaml` and are emitted again in
`manuscript_variables.json`; the prose is not maintained as a separate data
source.

| Quantity | Value |
| --- | ---: |
| Latent states | {{MODEL_STATES}} |
| Observations | {{MODEL_OBSERVATIONS}} |
| Actions | {{MODEL_ACTIONS}} |
| A shape | `{{MODEL_A_SHAPE}}` |
| B shape | `{{MODEL_B_SHAPE}}` |
| Planning horizon | {{HORIZON}} |
| Policy sequences | {{POLICY_COUNT}} |
| Random seed | {{SEED}} |

: Dimensions and deterministic experiment settings {#tbl:model_dimensions}

## Belief updates and policies

[@fig:belief_updates] compares the three dispatcher paths as they process the
same sequence. The particle path uses a seeded Dirichlet sample and therefore
has a controlled stochastic approximation; the variational and mean-field
paths are deterministic for this configuration. Every plotted trajectory is
the direct output of `dispatch_belief_update()`.

![Posterior trajectories for variational, mean-field, and sampling inference.](figures/belief_updates.png){#fig:belief_updates width=85%}

The policy distributions in [@fig:policy_distributions] are normalized over
the first action after sequence evaluation. They are distributions, not
unnormalized scores, and their support is exactly the configured action set.

![Policy distributions after horizon-based policy evaluation.](figures/policy_distributions.png){#fig:policy_distributions width=85%}

## Information terms

[@fig:efe_decomposition] decomposes the one-step expected free energy for each
action. Risk measures deviation from the softmax of $C$; ambiguity measures
expected observation entropy; epistemic value is subtracted because expected
information gain reduces expected free energy.

![Risk, ambiguity, and epistemic information gain for each action.](figures/efe_decomposition.png){#fig:efe_decomposition width=85%}

## Continuous dynamics and matrix structure

The continuous trajectory in [@fig:continuous_trajectory] uses the same seed
and observation schedule as the discrete experiment, with one-hot observation
vectors mapped through the validated observation matrix. [@fig:model_matrices]
shows the matrices used by the discrete generative model and makes the
stochastic orientation visually inspectable.

![Continuous generalized-coordinate belief means over the configured run.](figures/continuous_trajectory.png){#fig:continuous_trajectory width=85%}

![Observation and transition matrices used by the configured model.](figures/model_matrices.png){#fig:model_matrices width=85%}

## Evidence table

| Path | Observable contract | Regression evidence |
| --- | --- | --- |
| Variational | normalized posterior and policy | `test_all_dispatcher_methods_return_valid_distributions` |
| Mean-field | finite log-space update | `test_all_dispatcher_methods_return_valid_distributions` |
| Sampling | seeded particle mean | `test_sampling_seed_is_reproducible` |
| Simple POMDP | one-state matrices and round trip | `test_simple_pomdp_one_state_and_state_round_trip` |
| Continuous | multi-frame history and shape checks | `test_continuous_animation_has_multiple_frames` |
| Utilities | entropy, KL, softmax, and constraint checks | `test_invalid_distributions_and_one_state_initializer_are_rejected_or_valid` |

: Runtime evidence mapped to public behaviors {#tbl:evidence}
