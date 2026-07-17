# Experimental Setup {#sec:experimental_setup}

## Configuration

The full experiment is declared in `docs/manuscript/config.yaml`. Its model is

$$
A=\begin{bmatrix}0.9&0.1\\0.1&0.9\end{bmatrix},\quad
B_{0}=\begin{bmatrix}1.0&0.1\\0.0&0.9\end{bmatrix},\quad
B_{1}=\begin{bmatrix}0.0&0.9\\1.0&0.1\end{bmatrix},
$$ {#eq:configured_matrices}

with $C=(0,1)$ and uniform $D$ and $E$. The model contains {{MODEL_STATES}}
states, {{MODEL_OBSERVATIONS}} observations, and {{MODEL_ACTIONS}} actions.
The transition columns sum to one, so the tensor is compatible with
$B_{s'sa}=P(s'\mid s,a)$.

The sequence {{OBS_SEQUENCE}} is processed at horizon {{HORIZON}} with seed
{{SEED}}. The seed is passed into NumPy's `default_rng` for the sampling
dispatcher, matrix initializers, and continuous agent. No global random state
is used by the manuscript build.

## Measurements

The build records:

1. posterior trajectories for all {{METHOD_COUNT}} dispatcher methods;
2. first-action policy distributions after policy enumeration;
3. risk, ambiguity, and epistemic information gain by action;
4. continuous state means over {{FIGURE_COUNT}} generated figure assets;
5. matrix shapes, package version, figure registry, and combined-source hash.

The timing benchmark is a separate command because host timing is diagnostic
rather than a scientific result. `cognitive-benchmark --repetitions 10`
reports matrix operations, dispatcher inference, `SimplePOMDP`, and continuous
inference in JSON.

## Software environment

The package declares Python >=3.10 and its runtime dependencies in
`pyproject.toml`. The publication build requires Pandoc, `pandoc-crossref`,
and XeLaTeX for PDF output; `--no-pdf` retains the deterministic data and HTML
path on systems without those external tools.
