# Appendix: Validation matrix {#sec:appendix_validation}

The acceptance suite in `code/tests/` combines unit-level numerical checks,
integration checks, and documentation gates. The truth-audit regression file
contains negative controls as well as successful cases: invalid distributions,
unsupported constraints, malformed configuration, out-of-range actions,
one-state matrices, unsafe node names, path-like links, observation shape
errors, and missing animation frames are all exercised.

## Runtime gate

For every public inference method $m$ and every accepted input $x$, the
observable distribution contract is

$$
\operatorname{valid}(m,x) \iff
\left(\forall i:q_i\in\mathbb{R}_{\geq 0}\right)
\land \left|\sum_iq_i-1\right|\leq\epsilon.
$$ {#eq:coverage_contract}

This contract is checked for variational, mean-field, and sampling inference.
The same test family checks that two seeded sampling runs are identical,
while unseeded callers remain free to choose their own generator.

## Artifact gate

The manuscript build is valid only if all of the following are true:

| Artifact | Check |
| --- | --- |
| `combined.md` | no unresolved double-brace variables remain |
| `manuscript_variables.json` | computed variables come from the declared configuration and executable results and hydrate the prose |
| `references.bib` | all cited keys exist |
| `figure_registry.json` | unique label/path pairs agree with manuscript image declarations; references resolve; every registered asset is a regular nonempty file, not a symlink |
| `manuscript.html` | Pandoc exits successfully |
| `manuscript.pdf` | XeLaTeX exits successfully when requested |
| `build_manifest.json` | source hash and package version are present |

The registry is checked before rendering, not merely serialized afterward.
Missing, empty, non-regular, or symlinked assets; duplicate labels or registry
paths; swapped label-to-image associations; source figures absent from the
registry; unused registry entries; and unresolved figure cross-references fail
the build. `--no-render` exercises
these checks without requiring Pandoc or XeLaTeX. The `manuscript` gate runs
that path followed by HTML; `manuscript --pdf` also requires the full PDF.

## Limitations of evidence

The figures demonstrate implementation behavior for one small model. They do
not establish asymptotic performance, biological validity, or superiority over
other inference algorithms. Benchmark timings are host-dependent. The
scholarly claims are interpretive context; the executable claims are the
ones tied to source paths, configuration, and tests.
