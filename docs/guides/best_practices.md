---
title: Best Practices Guide
type: guide
status: stable
created: 2025-01-01
updated: 2026-02-07
tags:
  - best_practices
  - development
  - active_inference
  - quality
semantic_relations:
  - type: relates_to
    links:
      - [[docs/guides/api_implementation]]
      - [[docs/guides/agent_development]]
      - [[docs/implementation/README]]
---

# Best Practices

## API Best Practices

1. **Use typed configurations** — define explicit parameter types with validation
2. **Handle errors gracefully** — catch specific exceptions and provide meaningful fallbacks
3. **Specify precision parameters** — always set precision explicitly; do not rely on defaults

## Generative Model Best Practices

### Matrix Construction

- Ensure all likelihood matrices (**A**) have columns that sum to 1
- Ensure all transition matrices (**B**) have columns that sum to 1 for each action
- Use small epsilon values (1e-16) to avoid log(0) in free energy calculations
- Validate matrix dimensions match before running inference

### Belief Updating

- Initialize beliefs with informative priors when possible (not uniform)
- Monitor for numerical instabilities (NaN, Inf) during message passing
- Use log-space computations for numerical stability with small probabilities
- Set convergence thresholds explicitly rather than using fixed iteration counts

### Policy Selection

- Use softmax (not argmax) for action selection to maintain exploration
- Tune the precision parameter (β) to balance exploration vs exploitation
- Evaluate expected free energy components (epistemic + pragmatic) separately for debugging
- Limit planning horizon to manageable depths (1–3 for discrete models)

## Code Quality Best Practices

### Logging


- Log at appropriate levels: DEBUG for internals, INFO for flow, WARNING for anomalies
- Include free energy values in logs to track convergence
- Use structured logging for machine-parseable output

### Testing

- Write unit tests for each generative model component (A, B, C, D matrices)
- Test belief updating with known posterior distributions
- Verify policy selection produces correct actions for simple test cases
- Run regression tests when modifying inference algorithms

### Documentation

- Document all matrix dimensions and their semantic meaning
- Include mathematical notation alongside code implementations
- Provide worked examples with expected output for key functions
- Cross-reference theoretical sources (Friston, Parr, Da Costa) in docstrings

## Performance Best Practices

- Profile inference loops before optimizing
- Use NumPy vectorized operations instead of Python loops
- Cache repeated computations (e.g., log-likelihood ratios)
- Consider sparse representations for large state spaces
- Batch observations when processing time-series data

## Related Resources

- [[docs/guides/agent_development|Agent Development Guide]] — agent architecture patterns
- [[docs/guides/api_implementation|API Implementation Guide]] — API design patterns
- [[docs/implementation/README|Implementation Guides]] — detailed implementation reference
- [[docs/repo_docs/documentation_standards|Documentation Standards]] — documentation conventions
