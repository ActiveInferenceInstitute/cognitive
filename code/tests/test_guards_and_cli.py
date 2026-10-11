"""Constructor guards, reset paths, precision learning, and the link-check CLI.

Parameter validation in the continuous agent, its optional precision learning,
and the command-line surfaces of the Markdown link checker.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scripts import check_markdown_links
from Things.Continuous_Generic import ContinuousActiveInference


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_states": 0}, "n_states must be a positive integer"),
        ({"n_obs": -1}, "n_obs must be a positive integer"),
        ({"n_orders": 1.5}, "n_orders must be a positive integer"),
        ({"dt": 0.0}, "dt must be finite and positive"),
        ({"dt": float("inf")}, "dt must be finite and positive"),
        ({"alpha": -0.1}, "alpha must be finite and positive"),
        ({"precision_learning_rate": 1.5}, "precision_learning_rate must be in"),
        ({"dynamics_weight": float("nan")}, "dynamics_weight must be finite"),
        ({"seed": -1}, "seed must be a non-negative integer"),
        ({"seed": 1.5}, "seed must be a non-negative integer"),
    ],
)
def test_continuous_constructor_rejects_invalid_parameters(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ContinuousActiveInference(**kwargs)


def test_continuous_matrices_and_precisions_are_shape_checked() -> None:
    with pytest.raises(ValueError, match="observation_matrix must have shape"):
        ContinuousActiveInference(n_states=2, n_obs=2, observation_matrix=np.eye(3))
    with pytest.raises(ValueError, match="dynamics_matrix must have shape"):
        ContinuousActiveInference(n_states=2, dynamics_matrix=np.eye(3))
    with pytest.raises(ValueError, match="observation_precision must be a positive"):
        ContinuousActiveInference(n_obs=2, observation_precision=np.array([1.0, -1.0]))
    with pytest.raises(ValueError, match="state_precision must be a positive"):
        ContinuousActiveInference(n_states=2, state_precision=0.0)
    with pytest.raises(ValueError, match="must contain at least one finite value"):
        ContinuousActiveInference(n_states=2, dynamics_matrix=np.full((2, 2), np.nan))

    # A rectangular observation matrix is supported, and a scalar precision
    # broadcasts to every channel.
    agent = ContinuousActiveInference(n_states=3, n_obs=2, observation_precision=2.0)
    assert agent.observation_matrix.shape == (2, 3)
    assert agent.observation_precision == pytest.approx([2.0, 2.0])


def test_reset_restores_time_precisions_and_optional_means() -> None:
    agent = ContinuousActiveInference(n_states=2, n_obs=2, n_orders=3, seed=1)
    agent.step(np.array([0.5, -0.5]))
    assert agent.state.time > 0.0

    agent.reset()
    assert agent.state.time == 0.0
    assert agent.state.belief_means == pytest.approx(np.zeros((2, 3)))

    seeded = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    agent.reset(seeded)
    assert agent.state.belief_means == pytest.approx(seeded)
    # reset must copy, not alias, or a later step would mutate the caller's array.
    agent.step(np.zeros(2))
    assert seeded == pytest.approx(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))

    with pytest.raises(ValueError, match="belief_means must have shape"):
        agent.reset(np.zeros((2, 2)))


def test_step_validates_its_observation() -> None:
    agent = ContinuousActiveInference(n_states=2, n_obs=2)
    with pytest.raises(ValueError, match="observation must have shape"):
        agent.step(np.zeros(3))
    with pytest.raises(ValueError, match="must contain at least one finite value"):
        agent.step(np.array([np.nan, 0.0]))
    with pytest.raises(ValueError, match="observation must have shape"):
        agent._compute_free_energy(np.zeros(3), agent.state.belief_means)
    with pytest.raises(ValueError, match="belief_means must have shape"):
        agent._sensory_mapping(np.zeros((3, 3)))


def test_precision_learning_moves_precision_toward_the_observed_residual() -> None:
    """With learning off precision is fixed; with it on precision adapts."""
    fixed = ContinuousActiveInference(n_states=2, n_obs=2, precision_learning_rate=0.0)
    before = fixed.observation_precision.copy()
    for _ in range(5):
        fixed.step(np.array([1.0, -1.0]))
    assert fixed.observation_precision == pytest.approx(before)

    learning = ContinuousActiveInference(
        n_states=2, n_obs=2, precision_learning_rate=0.5, observation_precision=1.0
    )
    learning.step(np.array([1.0, -1.0]))
    # A residual of one gives an observed precision of one, so the blend holds
    # at one; a smaller residual must raise it above the starting value.
    assert learning.observation_precision == pytest.approx([1.0, 1.0], abs=1e-6)
    learning.step(np.array([0.0, 0.0]))
    assert np.all(learning.observation_precision > 1.0)
    assert np.all(np.isfinite(learning.observation_precision))


def test_step_returns_an_action_and_advances_time_by_dt() -> None:
    agent = ContinuousActiveInference(n_states=2, n_obs=2, dt=0.02)
    action, energy = agent.step(np.array([0.3, 0.4]))
    assert action.shape == (2,)
    assert np.isfinite(energy)
    assert agent.state.time == pytest.approx(0.02)
    agent.step(np.array([0.3, 0.4]))
    assert agent.state.time == pytest.approx(0.04)


def test_continuous_state_copy_is_independent() -> None:
    agent = ContinuousActiveInference(n_states=2, n_obs=2)
    snapshot = agent.state.copy()
    agent.step(np.array([1.0, 1.0]))
    assert snapshot.belief_means == pytest.approx(np.zeros((2, 3)))
    assert snapshot.time == 0.0


def _link_tree(root: Path) -> None:
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "target.md").write_text("# Target\n\n## Deep Section\n", encoding="utf-8")


def test_markdown_link_cli_reports_success_and_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both CLI shapes are exercised, and the exit code follows the finding."""
    import subprocess

    _link_tree(tmp_path)
    (tmp_path / "docs" / "page.md").write_text(
        "[ok](target.md)\n\n[same file](#deep-heading-that-is-absent)\n", encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)

    assert check_markdown_links.main([str(tmp_path)]) == 0
    text_output = capsys.readouterr().out
    assert "verified successfully" in text_output
    assert "anchor warning" in text_output

    assert check_markdown_links.main([str(tmp_path), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["checked_links"] >= 1
    assert report["broken_links"] == []
    assert report["anchor_warnings"]

    (tmp_path / "docs" / "broken.md").write_text("[gone](missing.md)\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    assert check_markdown_links.main([str(tmp_path)]) == 1
    assert "broken links" in capsys.readouterr().out
    assert check_markdown_links.main([str(tmp_path), "--json"]) == 1


def test_markdown_link_checker_ignores_external_latex_and_template_targets(
    tmp_path: Path,
) -> None:
    """Only repository-relative links are resolved, so the gate stays quiet."""
    import subprocess

    _link_tree(tmp_path)
    (tmp_path / "docs" / "page.md").write_text(
        "[site](https://example.org/page)\n\n"
        "[mail](mailto:someone@example.org)\n\n"
        "[token]({{OUTPUT_PATH}})\n\n"
        "[escaped](a\\\\b)\n\n"
        "```\n[fenced](missing.md)\n```\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)

    report = check_markdown_links.verify_markdown_link_report(tmp_path)
    assert report.broken_links == []
    assert report.indexed_markdown_files >= 2
