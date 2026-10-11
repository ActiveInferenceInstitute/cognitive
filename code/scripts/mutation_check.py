"""Run a declared set of source mutations and require every one to be killed.

Coverage says a line executed. It does not say a test would notice if that line
were wrong. Each entry below is a specific, plausible way one of the
package's mathematical or validation lines could be written incorrectly -- a
dropped Jacobian transpose, a flipped sign, a wrong axis, a boundary moved by
one. The mutation is written to disk, a focused subset of the suite runs, and
the mutation is reverted. A mutation the suite still passes is a *survivor*: the
line is executed by the tests but not checked by them.

Every mutation here corresponds to a defect this package actually shipped, or to
the boundary that a defect turned on. Add an entry whenever a bug is fixed, so
the test that caught it stays load-bearing.

Usage::

    python -m scripts.mutation_check            # all mutations
    python -m scripts.mutation_check --json     # machine-readable report
    python -m scripts.mutation_check --filter gradient
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Mutation:
    """One source edit that the suite must reject."""

    name: str
    path: str
    original: str
    mutated: str
    rationale: str
    tests: tuple[str, ...]

    def target(self, root: Path) -> Path:
        return root / self.path


MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        name="efe_subtracts_epistemic",
        path="code/tools/src/models/active_inference/generative_model.py",
        original="        total = risk + ambiguity\n",
        mutated="        total = risk + ambiguity - epistemic\n",
        rationale=(
            "The defect corrected in 1.1.0: expected free energy double-counted "
            "the information gain that ambiguity already contains."
        ),
        tests=("code/tests/test_math_oracles.py",),
    ),
    Mutation(
        name="efe_ambiguity_uses_predicted_observation_entropy",
        path="code/tools/src/utils/matrix_utils.py",
        original=(
            "    ambiguity = float(np.sum(predicted_states * "
            "compute_entropy(likelihood, axis=0)))\n"
        ),
        mutated=(
            "    ambiguity = float(np.sum(expected_obs * np.log(1.0 / "
            "np.maximum(expected_obs, EPS))))\n"
        ),
        rationale=(
            "The helper's original ambiguity term: the entropy of the predicted "
            "observation rather than the expected entropy of the likelihood."
        ),
        tests=("code/tests/test_math_oracles.py",),
    ),
    Mutation(
        name="efe_ambiguity_drops_state_weighting",
        path="code/tools/src/models/active_inference/generative_model.py",
        original="                predicted_states[state] * _entropy(self.A[:, state])\n",
        mutated="                _entropy(self.A[:, state])\n",
        rationale="Ambiguity is an expectation under q(s), not a bare sum over states.",
        tests=("code/tests/test_math_oracles.py",),
    ),
    Mutation(
        name="efe_risk_reverses_divergence_arguments",
        path="code/tools/src/models/active_inference/generative_model.py",
        original="        risk = _kl_divergence(predicted_observations, self.preferences)\n",
        mutated="        risk = _kl_divergence(self.preferences, predicted_observations)\n",
        rationale="KL divergence is asymmetric; risk is D_KL(q(o) || p*(o)), not its reverse.",
        tests=("code/tests/test_math_oracles.py",),
    ),
    Mutation(
        name="efe_risk_sign_is_flipped",
        path="code/tools/src/models/active_inference/generative_model.py",
        original="        risk = _kl_divergence(predicted_observations, self.preferences)\n",
        mutated="        risk = -_kl_divergence(predicted_observations, self.preferences)\n",
        rationale=(
            "Risk is a divergence and therefore non-negative; a negated risk "
            "would reward predicting the least preferred observation."
        ),
        tests=("code/tests/test_math_oracles.py",),
    ),
    Mutation(
        name="continuous_drops_dynamics_jacobian",
        path="code/Things/Continuous_Generic/continuous_generic.py",
        original="            updates[:, order] += self.dynamics_matrix.T @ weighted\n",
        mutated="            updates[:, order] += weighted\n",
        rationale=(
            "The defect corrected in 1.1.0: the belief update omitted the "
            "dynamics Jacobian transpose, which the identity default hides."
        ),
        tests=("code/tests/test_math_oracles.py",),
    ),
    Mutation(
        name="continuous_flips_dynamic_error_sign",
        path="code/Things/Continuous_Generic/continuous_generic.py",
        original="            updates[:, order + 1] -= weighted\n",
        mutated="            updates[:, order + 1] += weighted\n",
        rationale="Gradient descent requires the higher coordinate to move against the error.",
        tests=("code/tests/test_math_oracles.py",),
    ),
    Mutation(
        name="entropy_normalizes_along_the_wrong_axis",
        path="code/tools/src/utils/matrix_utils.py",
        original='    probabilities = _probability(distribution, "distribution", axis=axis)\n',
        mutated='    probabilities = _probability(distribution, "distribution", axis=0)\n',
        rationale=(
            "The defect corrected in 1.1.0: normalization and summation used "
            "different axes, which coincide only for a vector."
        ),
        tests=("code/tests/test_math_oracles.py",),
    ),
    Mutation(
        name="policy_evaluation_drops_the_discount",
        path="code/tools/src/models/active_inference/generative_model.py",
        original=(
            "            total += float(discount_factor**step) * "
            "(value - epistemic_weight * epistemic)\n"
        ),
        mutated="            total += (value - epistemic_weight * epistemic)\n",
        rationale="A horizon longer than one step must discount later steps.",
        tests=("code/tests/test_math_oracles.py",),
    ),
    Mutation(
        name="homeostatic_preferences_ignore_the_likelihood",
        path="code/tools/src/models/active_inference/homeostatic.py",
        original="            likelihood @ target, self.observation_space.dimensions[0]\n",
        mutated="            target, self.observation_space.dimensions[0]\n",
        rationale=(
            "The defect corrected in 1.1.0: the state target was read directly "
            "as an observation preference."
        ),
        tests=("code/tests/test_model_lifecycle.py",),
    ),
    Mutation(
        name="policy_limit_boundary_is_off_by_one",
        path="code/tools/src/models/active_inference/generative_model.py",
        original="        if limit is not None and limit < 1:\n",
        mutated="        if limit is not None and limit < 0:\n",
        rationale="A limit of zero must be rejected, not silently return no policies.",
        tests=("code/tests/test_truth_audit.py", "code/tests/test_math_oracles.py"),
    ),
    Mutation(
        name="model_validation_accepts_row_stochastic_likelihood",
        path="code/tools/src/models/active_inference/generative_model.py",
        original="        if not np.allclose(self.A.sum(axis=0), 1.0):\n",
        mutated="        if not np.allclose(self.A.sum(axis=1), 1.0):\n",
        rationale="A is column stochastic: P(o | s) sums to one over observations.",
        tests=("code/tests/test_truth_audit.py", "code/tests/test_math_oracles.py"),
    ),
    Mutation(
        name="belief_blend_swaps_the_learning_rate",
        path="code/tools/src/models/active_inference/dispatcher.py",
        original=(
            "            (1.0 - self.config.learning_rate) * prior_vector\n"
            "            + self.config.learning_rate * posterior_vector,\n"
        ),
        mutated=(
            "            self.config.learning_rate * prior_vector\n"
            "            + (1.0 - self.config.learning_rate) * posterior_vector,\n"
        ),
        rationale=(
            "learning_rate weights the posterior. A rate of 0.5 is symmetric, so "
            "a suite that only ever uses 0.5 cannot tell the two apart."
        ),
        tests=("code/tests/test_dispatcher_semantics.py",),
    ),
    Mutation(
        name="policy_inference_drops_the_temperature",
        path="code/tools/src/models/active_inference/dispatcher.py",
        original="        return self._softmax(-energies / self.config.temperature)\n",
        mutated="        return self._softmax(-energies)\n",
        rationale="Temperature controls how deterministic action selection is.",
        tests=("code/tests/test_dispatcher_semantics.py",),
    ),
    Mutation(
        name="policy_aggregation_inverts_to_a_maximum",
        path="code/tools/src/models/active_inference/dispatcher.py",
        original="                first_action_values[action] = min(candidates)\n",
        mutated="                first_action_values[action] = max(candidates)\n",
        rationale=(
            "Scoring each first action by its worst continuation rather than its "
            "best inverts the planner's preference ordering."
        ),
        tests=("code/tests/test_dispatcher_semantics.py",),
    ),
    Mutation(
        name="precision_clip_upper_bound_moves",
        path="code/tools/src/models/active_inference/dispatcher.py",
        original=(
            "            np.clip(0.9 * self.config.precision_init + 0.1 * target, 0.1, 1e6)\n"
        ),
        mutated=(
            "            np.clip(0.9 * self.config.precision_init + 0.1 * target, 0.1, 1e3)\n"
        ),
        rationale="Both ends of a documented clip must be reached by some test.",
        tests=("code/tests/test_dispatcher_semantics.py",),
    ),
    Mutation(
        name="posterior_treats_small_evidence_as_impossible",
        path="code/tools/src/models/active_inference/generative_model.py",
        original="        if evidence <= 0.0:\n",
        mutated="        if evidence <= EPS:\n",
        rationale=(
            "Bayes' rule is scale-invariant. Comparing the evidence with a "
            "tolerance discards a legitimate posterior whose likelihood is small."
        ),
        tests=("code/tests/test_boundary_validation.py",),
    ),
    Mutation(
        name="entropy_floors_every_probability",
        path="code/tools/src/utils/matrix_utils.py",
        original=(
            "        np.where(probabilities > 0, probabilities * "
            "np.log(np.maximum(probabilities, EPS)), 0.0),\n"
        ),
        mutated=(
            "        np.where(probabilities > 0, probabilities * "
            "np.log(probabilities + EPS), 0.0),\n"
        ),
        rationale=(
            "Adding a floor to every entry makes the entropy of a deterministic "
            "distribution slightly negative."
        ),
        tests=("code/tests/test_boundary_validation.py",),
    ),
    Mutation(
        name="index_conversion_truncates_fractions",
        path="code/tools/src/utils/matrix_utils.py",
        original="    if index != value:\n",
        mutated="    if False:\n",
        rationale="int(1.9) is 1: an index must be integral, not silently rounded down.",
        tests=("code/tests/test_boundary_validation.py",),
    ),
    Mutation(
        name="matrix_constraints_accept_non_finite_input",
        path="code/tools/src/utils/matrix_utils.py",
        original="    if values.size == 0 or not np.all(np.isfinite(values)):\n",
        mutated="    if values.size == 0:\n",
        rationale="A NaN column fails sum > EPS and is replaced by a uniform distribution.",
        tests=("code/tests/test_boundary_validation.py",),
    ),
    Mutation(
        name="landscape_grid_is_transposed",
        path="code/tools/src/visualization/matrix_plots.py",
        original='        X, Y = np.meshgrid(x, y, indexing="ij")\n',
        mutated="        X, Y = np.meshgrid(x, y)\n",
        rationale="The default xy indexing transposes the grid, so Z mismatches non-square input.",
        tests=("code/tests/test_boundary_validation.py",),
    ),
    Mutation(
        name="term_policy_exemption_is_ignored",
        path="code/scripts/validate_docs.py",
        original="        if relative in exempt_paths:\n            continue\n",
        mutated="        if False:\n            continue\n",
        rationale=(
            "Exempting the policy file and its validator is what lets the terms "
            "be spelled literally; without it the gate must fail on itself."
        ),
        tests=("code/tests/test_build_pipeline.py",),
    ),
    Mutation(
        name="normalization_floors_the_total_at_eps",
        path="code/tools/src/utils/matrix_utils.py",
        original="    positive = totals > 0.0\n",
        mutated="    positive = totals > EPS\n",
        rationale=(
            "The defect fixed in the shared normalizer: a total below 1e-12 was "
            "treated as empty, so a distribution of small entries became uniform."
        ),
        tests=("code/tests/test_scale_invariant_normalization.py",),
    ),
    Mutation(
        name="goal_prior_is_not_scaled_by_temperature",
        path="code/tools/src/models/active_inference/dispatcher.py",
        original=(
            "            first_action_values -= self.config.temperature * np.log(prior + 1e-12)\n"
        ),
        mutated="            first_action_values -= np.log(prior + 1e-12)\n",
        rationale=(
            "The variational and sampling policies applied ln(prior) / T while "
            "the mean-field policy applied ln(prior); they agree only at T = 1."
        ),
        tests=("code/tests/test_behaviour_changes.py",),
    ),
    Mutation(
        name="adaptive_prior_ignores_prediction_error",
        path="code/tools/src/models/active_inference/homeostatic.py",
        original=(
            "        sharpness = (1.0 - self.exploration_weight) / "
            "(1.0 + self.exploration_weight * error)\n"
        ),
        mutated="        sharpness = (1.0 - self.exploration_weight) / 1.0\n",
        rationale=(
            "The exploration term was constant across entries, so the max-shift "
            "removed it and prediction_error had no effect on the prior."
        ),
        tests=("code/tests/test_behaviour_changes.py",),
    ),
    Mutation(
        name="learned_precision_target_is_unbounded",
        path="code/Things/Continuous_Generic/continuous_generic.py",
        original=(
            "                np.clip(observed_precision.mean(), "
            "self.precision_min, self.precision_max),\n"
        ),
        mutated="                observed_precision.mean(),\n",
        rationale=(
            "An unclipped 1 / residual**2 target sent the precision to 1e12 after "
            "one zero-residual step and threw the beliefs far from the data."
        ),
        tests=("code/tests/test_behaviour_changes.py",),
    ),
    Mutation(
        name="pomdp_inference_seed_is_ignored",
        path="code/Things/Simple_POMDP/simple_pomdp.py",
        original='            for label, value in (("seed", root), ("inference.seed", nested))\n',
        mutated='            for label, value in (("seed", root), ("inference.seed", None))\n',
        rationale="inference.seed was accepted by the schema and read by nothing.",
        tests=("code/tests/test_behaviour_changes.py",),
    ),
    Mutation(
        name="pomdp_temperature_is_ignored",
        path="code/Things/Simple_POMDP/simple_pomdp.py",
        original="        if temperature is None:\n",
        mutated="        if True:\n",
        rationale="inference.temperature was accepted by the schema and read by nothing.",
        tests=("code/tests/test_behaviour_changes.py",),
    ),
    Mutation(
        name="fence_closes_on_any_three_backticks",
        path="code/scripts/markdown_fences.py",
        original='len(closing["fence"]) >= length:\n',
        mutated='len(closing["fence"]) >= 3:\n',
        rationale=(
            "A closing fence must be at least as long as its opener; otherwise "
            "a three-backtick example ends a four-backtick block early."
        ),
        tests=("code/tests/test_markdown_fences.py",),
    ),
    Mutation(
        name="unterminated_fence_ends_where_it_opens",
        path="code/scripts/markdown_fences.py",
        original="        last = len(lines) - 1\n",
        mutated="        last = index\n",
        rationale=(
            "An unterminated fence runs to the end of the file; ending it at the "
            "opening line reads its contents as headings."
        ),
        tests=("code/tests/test_markdown_fences.py",),
    ),
    Mutation(
        name="markdown_link_to_a_missing_file_is_not_reported",
        path="code/scripts/check_markdown_links.py",
        original="            if resolved is None:\n",
        mutated="            if resolved is None and False:\n",
        rationale=("A [text](path) link whose target does not exist must be reported as broken."),
        tests=("code/tests/test_check_markdown_links.py",),
    ),
    Mutation(
        name="explicit_missing_wiki_file_is_a_concept_link",
        path="code/scripts/verify_links.py",
        original=(
            "            if not strict_wiki_links and not _is_explicit_file_reference(target):\n"
        ),
        mutated="            if not strict_wiki_links:\n",
        rationale=(
            "A wiki link that names a file (missing.md) is broken, "
            "not an unwritten concept to skip."
        ),
        tests=("code/tests/test_verify_links.py",),
    ),
    Mutation(
        name="external_markdown_links_are_resolved_as_files",
        path="code/scripts/check_markdown_links.py",
        original="            if _is_external(target):\n",
        mutated="            if False:\n",
        rationale=(
            "An http(s) link is not a file path; resolving it "
            "reports every external link as missing."
        ),
        tests=("code/tests/test_check_markdown_links.py",),
    ),
    Mutation(
        name="wiki_link_anchor_is_not_checked",
        path="code/scripts/verify_links.py",
        original="                    if _slugify(fragment) not in slugs:\n",
        mutated="                    if False:\n",
        rationale=(
            "A [[page#heading]] link whose heading does not exist must produce an anchor warning."
        ),
        tests=("code/tests/test_markdown_fences.py",),
    ),
    Mutation(
        name="same_file_anchor_is_not_checked",
        path="code/scripts/check_markdown_links.py",
        original="                    if slug not in own_slugs:\n",
        mutated="                    if False:\n",
        rationale=("A (#heading) link to a heading the page lacks must produce an anchor warning."),
        tests=("code/tests/test_check_markdown_links.py",),
    ),
    Mutation(
        name="falsy_relation_links_are_read_as_empty",
        path="code/scripts/normalize_frontmatter.py",
        original='        links = [] if entry.get("links") is None else entry["links"]\n',
        mutated='        links = entry.get("links") or []\n',
        rationale="`or []` turns 0, an empty string, and false into a valid empty list.",
        tests=("code/tests/test_markdown_fences.py",),
    ),
)


def _run(tests: tuple[str, ...], root: Path, cache_prefix: Path) -> bool:
    """Return True when the given tests pass against the mutated source.

    ``PYTHONPYCACHEPREFIX`` points at a directory unique to this run. Python
    invalidates a cached ``.pyc`` by comparing the source's modification time
    and size, and several of the mutations below replace a character with
    another of the same width -- ``limit < 1`` becomes ``limit < 0``. Written
    inside the same clock tick, such an edit leaves both fields unchanged, the
    interpreter loads the stale bytecode, and the mutation never reaches the
    tests, which then pass and are reported as a survivor. Isolating the cache
    removes that failure mode entirely.

    ``COV_CORE_*`` variables are stripped as well: when the harness itself runs
    under ``pytest --cov``, pytest-cov injects its subprocess support into the
    environment, and the child pytest then measures coverage of the probe run
    alone and fails the configured floor. That non-zero exit looks exactly like
    a kill and silently converts an unkillable probe mutation into a killed
    one, which would defeat the survivor-detection tests.
    """
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("COV_CORE") and key != "PYTEST_DISABLE_PLUGIN_AUTOLOAD"
    }
    environment["PYTHONPYCACHEPREFIX"] = str(cache_prefix)
    # ``--no-cov`` requires the pytest-cov plugin to be loaded to be a
    # recognized option. When the parent suite was started with
    # PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 (as CI's coverage run does), a child
    # that inherits it fails with an argument error -- exit code 4 with
    # "unrecognized arguments" -- which contains no FAILED line but does
    # contain "ERROR", and would be miscounted as a kill. Explicitly loading
    # the plugin makes the flag valid and the run meaningful.
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-x",
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
            "-p",
            "pytest_cov",
            "--no-cov",
            *tests,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        env=environment,
    )
    if result.returncode == 0:
        return True
    # A non-zero exit is only a kill when a test actually failed. A mutation
    # that breaks the import or the parse makes pytest exit non-zero during
    # collection, which proves the file no longer runs, not that a wrong
    # answer would be noticed. Requiring a reported failure keeps the gate
    # measuring what it claims to measure.
    output = result.stdout + result.stderr
    if "FAILED" in output or "ERROR" in output:
        return False
    raise ValueError(
        "A mutation made pytest exit without reporting a test failure, which "
        "usually means it broke collection rather than the answer. Rewrite the "
        "mutation so the module still imports.\n"
        f"exit code {result.returncode}\n{output[-2000:]}"
    )


def check(root: Path | None = None, name_filter: str | None = None) -> dict[str, object]:
    """Apply each mutation, run its tests, and restore the source."""
    repository = Path(root or REPO_ROOT).resolve()
    selected = [
        mutation for mutation in MUTATIONS if name_filter is None or name_filter in mutation.name
    ]
    if not selected:
        raise ValueError(f"No mutation matches filter {name_filter!r}")

    results: list[dict[str, object]] = []
    for mutation in selected:
        target = mutation.target(repository)
        source = target.read_text(encoding="utf-8")
        if source.count(mutation.original) != 1:
            raise ValueError(
                f"Mutation {mutation.name!r} does not apply: expected exactly one "
                f"occurrence of its anchor in {mutation.path}, found "
                f"{source.count(mutation.original)}. The source moved; update the "
                "mutation or delete it."
            )
        with tempfile.TemporaryDirectory(prefix="cognitive-mutation-") as cache_dir:
            try:
                target.write_text(
                    source.replace(mutation.original, mutation.mutated), encoding="utf-8"
                )
                survived = _run(mutation.tests, repository, Path(cache_dir))
            finally:
                target.write_text(source, encoding="utf-8")
        results.append(
            {
                "name": mutation.name,
                "path": mutation.path,
                "rationale": mutation.rationale,
                "tests": list(mutation.tests),
                "killed": not survived,
            }
        )

    survivors = [entry for entry in results if not entry["killed"]]
    return {
        "schema_version": 1,
        "mutations": len(results),
        "killed": len(results) - len(survivors),
        "survivors": survivors,
        "results": results,
        "ok": not survivors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify that the suite kills declared mutations")
    parser.add_argument("root", nargs="?", default=str(REPO_ROOT))
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable report")
    parser.add_argument(
        "--filter", dest="name_filter", help="Run only mutations matching this text"
    )
    args = parser.parse_args(argv)
    report = check(Path(args.root), args.name_filter)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for entry in report["results"]:
            status = "killed " if entry["killed"] else "SURVIVED"
            print(f"{status}  {entry['name']}  ({entry['path']})")
        print(f"\n{report['killed']}/{report['mutations']} mutations killed.")
        for entry in report["survivors"]:
            print(f"\nSurvivor: {entry['name']}\n  {entry['rationale']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
