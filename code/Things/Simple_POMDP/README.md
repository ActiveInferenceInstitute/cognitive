# Simple POMDP

`SimplePOMDP` is a compact seeded discrete POMDP implementation. Its YAML
configuration defines state, observation, action, matrix, inference, and
visualization sections. Matrix constraints are explicit: `A` and each `B`
slice are column-stochastic, preferences are finite, and `D` and `E` are
normalized distributions.

The implementation supports one-state matrices, deterministic `run` and
`reset` operations, versioned YAML state persistence, history tracking,
expected-free-energy components, and plot types listed in
`_SimplePOMDPPlotter`.

Tests provide configurations in temporary directories, so running the suite
does not create repository output trees:

```bash
python -m pytest code/tests/test_simple_pomdp.py -q
```
