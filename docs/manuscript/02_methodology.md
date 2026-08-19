# Methodology {#sec:methodology}

## Discrete inference

`DiscreteGenerativeModel` is immutable after validation. Its constructor
rejects rank mismatches, non-finite values, negative likelihoods, invalid
stochastic axes, empty priors, and preference vectors with non-finite values.
The shared implementation is consumed by
`ActiveInferenceDispatcher`, `SimplePOMDP`, and `HomeostaticInference`, which
prevents each agent family from silently adopting a different matrix
convention.

The variational path blends the current belief with the exact posterior using
the configured learning rate. The mean-field path performs a log-space update
and stable softmax. The sampling path draws Dirichlet particles around the
posterior and normalizes their empirical mean. All three paths return the
same contract: a finite, non-negative vector of length {{MODEL_STATES}} whose
sum is one.

## Expected free energy

For a predicted state distribution $q(s)$ and predicted observation
distribution $q(o)$, expected free energy is the sum of a risk term and an
ambiguity term:

$$
G = \underbrace{D_{KL}(q(o)\,\|\,p^*(o))}_{\text{risk}}
  + \underbrace{\mathbb{E}_{q(s)}[H(P(o\mid s))]}_{\text{ambiguity}}.
$$ {#eq:efe_decomposition}

This is algebraically identical to the expected-surprise form, in which the
expected information gain is subtracted from the expected cost:

$$
G = \underbrace{\mathbb{E}_{q(o)}[-\log p^*(o)]}_{\text{expected surprise}}
  - \underbrace{\left(H[q(s)]-\mathbb{E}_{q(o)}H[q(s\mid o)]\right)}_{\text{epistemic gain}},
$$ {#eq:efe_surprise}

because the ambiguity term already contains the epistemic gain:
$\mathbb{E}_{q(s)}[H(P(o\mid s))] = H[q(o)] - I(s; o)$, where $I(s;o)$ is the
mutual information that [@eq:efe_surprise] names the epistemic gain. Versions of
this package before 1.1.0 subtracted the epistemic term from
[@eq:efe_decomposition] as well, which weighted information seeking twice and
broke the equality between the two forms. The regression test
`test_efe_decompositions_agree` now pins it for randomly generated models.

The method `expected_free_energy()` in
`code/tools/src/models/active_inference/generative_model.py` returns the total
alongside all three components, so the epistemic gain remains available as a
diagnostic without entering the objective a second time. A configurable
`exploration_weight` subtracts a further multiple of the epistemic gain for
callers that want an explicit information-seeking bias; it defaults to zero,
and a value of one reproduces the pre-1.1.0 objective exactly.

Policy evaluation sums discounted one-step values over the configured horizon,
then the dispatcher selects the minimum value for each possible first action
before applying softmax.

## Continuous generalized coordinates

`ContinuousActiveInference` stores means and precisions for orders
$0,\ldots,n-1$. With observation mapping $g(x)=Cx$ and dynamics $f(x)=Fx$,
the sensory prediction error is

$$
\varepsilon_y = y - Cx,
$$ {#eq:continuous_prediction_error}

and the precision-weighted update is

$$
\dot{x} = Fx + \alpha C^T\Pi_y\varepsilon_y,
\qquad
x_{t+1}=x_t+\Delta t\,\dot{x}.
$$ {#eq:continuous_update}

Higher generalized coordinates are advanced by the same dynamics matrix.
Observation and state precisions are positive vectors, and the update can
learn observation precision at a configured rate. The visualizer renders the
full history through `matplotlib.animation.FuncAnimation` and Pillow, so the
number of animation frames equals the number of recorded states.

## Agent lifecycle and control

`ActiveInferenceModel` supplies a concrete lifecycle around validated state,
belief updates, policy inference, precision adaptation, free-energy
calculation, and versioned YAML persistence. Abstract hooks raise explicitly
until a subclass supplies behavior. `StateSpace`, `ObservationModel`, and
`TransitionModel` validate dimensions and lookup labels before
`HomeostaticInference` can act. `HomeostaticControl` and `AdaptiveControl`
produce policy priors that are combined with the dispatcher distribution.

## Numerical safeguards

The matrix utilities reject non-finite arrays, negative probability mass,
zero-mass distributions, incompatible KL inputs, non-positive temperatures,
and unknown constraints. Axis arguments are preserved for N-dimensional
normalization. Random initializers accept an explicit NumPy generator; all
manuscript runs use the seed in `config.yaml`.
