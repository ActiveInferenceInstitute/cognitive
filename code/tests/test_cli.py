"""The ``cognitive`` command line, run in-process and as a module."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from cognitive.cli import build_parser, main

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL = REPO_ROOT / "examples" / "configs" / "two_state.yaml"


def run(capsys: pytest.CaptureFixture[str], *arguments: str) -> tuple[int, str, str]:
    status = main(list(arguments))
    captured = capsys.readouterr()
    return status, captured.out, captured.err


def test_settings_prints_the_resolved_values_as_json_or_text(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    config = tmp_path / "c.yaml"
    config.write_text("simulation:\n  steps: 9\n", encoding="utf-8")
    status, out, _ = run(capsys, "settings", "--config", str(config), "--json")
    assert status == 0 and json.loads(out)["simulation"]["steps"] == 9
    status, out, _ = run(capsys, "settings", "--config", str(config), "--set", "output.dpi=70")
    assert status == 0 and "steps: 9" in out and "dpi: 70" in out


def test_simulate_writes_the_table_json_and_figures(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    status, out, _ = run(
        capsys,
        "simulate",
        "--model",
        str(MODEL),
        "--steps",
        "8",
        "--seed",
        "3",
        "--output",
        str(tmp_path),
        "--json",
    )
    result = json.loads(out)
    assert status == 0 and result["summary"]["steps"] == 8 and result["summary"]["seed"] == 3
    names = sorted(Path(path).name for path in result["files"])
    assert names == [
        "efe_step0.png",
        "episode.png",
        "model.png",
        "trajectory.csv",
        "trajectory.json",
    ]
    assert all(Path(path).is_file() for path in result["files"])
    assert json.loads((tmp_path / "trajectory.json").read_text())["labels"]["actions"] == [
        "stay",
        "flip",
    ]


def test_simulate_without_figures_or_json_prints_the_summary(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    status, out, _ = run(
        capsys,
        "simulate",
        "--steps",
        "5",
        "--no-figures",
        "--set",
        "output.figure_format=svg",
        "--output",
        str(tmp_path),
    )
    assert status == 0 and "belief_accuracy:" in out
    assert sorted(path.name for path in tmp_path.iterdir()) == ["trajectory.csv", "trajectory.json"]


def test_the_output_directory_defaults_to_the_setting(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    status, _, _ = run(
        capsys, "simulate", "--steps", "3", "--no-figures", "--set", "output.directory=results"
    )
    assert status == 0 and (tmp_path / "results" / "trajectory.json").is_file()


def test_sweep_writes_rows_a_summary_and_a_figure(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    status, out, _ = run(
        capsys,
        "sweep",
        "--parameter",
        "temperature",
        "--values",
        "0.2, 2",
        "--seeds",
        "2",
        "--steps",
        "6",
        "--output",
        str(tmp_path),
        "--set",
        "output.figure_format=svg",
    )
    assert status == 0 and "temperature=0.2:" in out and "temperature=2.0:" in out
    assert (tmp_path / "sweep.svg").is_file()
    rows = (tmp_path / "sweep_episodes.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 + 2 * 2  # header, two values, two seeds each
    status, out, _ = run(
        capsys,
        "sweep",
        "--parameter",
        "policy_aggregation",
        "--values",
        "marginal,min",
        "--seeds",
        "1",
        "--steps",
        "4",
        "--no-figures",
        "--json",
        "--output",
        str(tmp_path / "second"),
    )
    assert status == 0 and [row["value"] for row in json.loads(out)["summary"]] == [
        "marginal",
        "min",
    ]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["settings", "--set", "simulation.steps"], "expects SECTION.KEY=VALUE"),
        (["settings", "--set", "=3"], "expects SECTION.KEY=VALUE"),
        (["settings", "--set", "simulation.steps=0"], "at least one"),
        (["settings", "--config", "absent.yaml"], "Cannot read configuration"),
        (["simulate", "--model", "absent.yaml", "--no-figures"], "Cannot read model file"),
        (["sweep", "--parameter", "temperature", "--values", " , "], "at least one value"),
        (["sweep", "--parameter", "temperature", "--values", "1", "--seeds", "0"], "--seeds"),
        (
            ["sweep", "--parameter", "learning_rate", "--values", "3", "--no-figures"],
            "learning_rate",
        ),
    ],
)
def test_rejected_input_exits_2_with_the_reason_on_stderr(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, arguments: list[str], message: str
) -> None:
    if arguments[0] != "settings":
        arguments = [*arguments, "--output", str(tmp_path)]
    status, out, err = run(capsys, *arguments)
    assert status == 2 and out == "" and message in err and err.startswith("error:")


def test_benchmark_is_passed_through(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    target = tmp_path / "bench.json"
    status, out, _ = run(capsys, "benchmark", "--repetitions", "1", "--output", str(target))
    assert status == 0 and json.loads(out)["repetitions"] == 1
    assert json.loads(target.read_text())["benchmarks"]


def test_an_unknown_sweep_parameter_is_refused_by_the_parser(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["sweep", "--parameter", "bogus", "--values", "1"])
    assert raised.value.code == 2
    assert "invalid choice: 'bogus'" in capsys.readouterr().err
    assert build_parser().prog == "cognitive"


def test_python_dash_m_cognitive_runs_the_same_command() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "cognitive", "settings", "--json", "--set", "simulation.seed=4"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    assert json.loads(result.stdout)["simulation"]["seed"] == 4
