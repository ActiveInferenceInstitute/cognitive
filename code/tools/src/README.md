# Package source

This directory is packaged as `cognitive`. The source tree is split into:

- `models/active_inference`: validated generative models, inference dispatch,
  lifecycle state, and homeostatic control;
- `models/matrices`: stochastic matrix operations, loading, initialization,
  and plotting adapters;
- `utils`: matrix helpers, safe node creation, and deterministic network
  visualization;
- `visualization`: matrix plotting implementations;
- `benchmarks.py`: the `cognitive-benchmark` console command.

Import public classes from `cognitive` or the documented submodules after an
editable install. The package does not advertise continuous or hierarchical
dispatcher modes; continuous inference is implemented by
`Things.Continuous_Generic.ContinuousActiveInference`.
