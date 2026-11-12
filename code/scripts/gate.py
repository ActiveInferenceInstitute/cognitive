"""Run the same independently selectable quality arms locally and in CI."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PYTHON = sys.executable
COMMANDS = {
    "lint": [
        [PYTHON, "-m", "ruff", "check", "."],
        [PYTHON, "-m", "ruff", "format", "--check", "code"],
        [PYTHON, "-m", "compileall", "-q", "code"],
    ],
    "types": [[PYTHON, "-m", "mypy", "code/tools/src", "code/Things", "code/scripts"]],
    "test": [
        [
            PYTHON,
            "-m",
            "pytest",
            "-q",
            "-m",
            "not visualization",
            "code/tests/test_active_inference_api.py",
            "code/tests/test_dispatcher_semantics.py",
            "code/tests/test_simple_pomdp.py",
            "code/tests/test_math_oracles.py",
        ]
    ],
    "test-full": [[PYTHON, "-m", "pytest", "-q", "-p", "pytest_cov", "--cov"]],
    "docs": [
        [PYTHON, "code/scripts/docs_index.py", "--check"],
        [PYTHON, "code/scripts/gate_evidence.py", "--check"],
        [PYTHON, "code/scripts/validate_docs.py", "--json"],
        [PYTHON, "-m", "scripts.export_rdf", "--check-submission"],
        [PYTHON, "-m", "scripts.export_rdf", "--quality-report"],
        [PYTHON, "-m", "scripts.semantic_review", "--root", ".", "--output", "docs/okn", "--check"],
    ],
}


def provision(runtime: Path) -> None:
    """Install only the recorded upstream commit after checking its tag."""
    # Resolving the executable follows a venv symlink into the shared Python
    # installation. Keep its lexical path; only canonicalize the environment.
    runtime = runtime.absolute()
    environment = runtime.parent.parent
    protected = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}
    if environment.resolve() in protected:
        raise ValueError("Refusing to provision into the running Python environment")
    if environment.exists() and not (environment / "pyvenv.cfg").is_file():
        raise ValueError("Refusing to replace an existing non-venv directory")
    pin = json.loads(Path("specs/gnn-pin.json").read_text())
    tag = pin["tag"]
    output = subprocess.check_output(
        [
            "git",
            "ls-remote",
            "--tags",
            pin["repository"],
            f"refs/tags/{tag}",
            f"refs/tags/{tag}^{{}}",
        ],
        stdin=subprocess.DEVNULL,
        text=True,
    )
    refs = {ref: commit for commit, ref in (line.split() for line in output.splitlines())}
    resolved = refs.get(f"refs/tags/{tag}^{{}}") or refs.get(f"refs/tags/{tag}")
    if resolved != pin["commit"]:
        raise ValueError(f"tag {tag} resolves to {resolved}, pin says {pin['commit']}")
    subprocess.run(
        ["uv", "venv", "--allow-existing", "--python", "3.12", str(environment)],
        check=True,
    )
    requirement = f"generalized-notation-notation @ git+{pin['repository']}@{pin['commit']}"
    subprocess.run(["uv", "pip", "install", "--python", str(runtime), requirement], check=True)


def verify_gnn_version(runtime: Path) -> None:
    """Keep the installed version report bound to the recorded tag."""
    pin = json.loads(Path("specs/gnn-pin.json").read_text())
    version = subprocess.check_output(
        [str(runtime), "-c", "import gnn; print(gnn.__version__)"],
        stdin=subprocess.DEVNULL,
        text=True,
    ).strip()
    if version != pin["tag"].removeprefix("v"):
        raise ValueError(f"Installed GNN version {version} differs from {pin['tag']}")
    print(f"gnn-version={version}")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a") as handle:
            handle.write(f"gnn-version={version}\n")


def commands(arm: str, runtime: Path, output: Path, pdf: bool) -> list[list[str]]:
    """Return each arm's authoritative commands."""
    if arm in COMMANDS:
        return COMMANDS[arm]
    if arm == "gnn":
        return [
            [PYTHON, "-m", "scripts.validate_gnn_spec", "--gnn-runtime", str(runtime), "--json"],
            [PYTHON, "-m", "pytest", "-q", "code/tests/test_gnn_conformance.py"],
        ]
    render = [PYTHON, "-m", "scripts.build_manuscript", "--output", str(output)]
    return [
        [
            PYTHON,
            "-m",
            "scripts.build_manuscript",
            "--no-render",
            "--output",
            str(output) + "-source",
        ],
        render if pdf else render + ["--no-pdf"],
    ]


def verify_render(output: Path, pdf: bool) -> None:
    """Reject a skipped render or unresolved tokens and cross-references."""
    manifest = json.loads((output / "build_manifest.json").read_text())
    if manifest["rendered_documents"] is not True or manifest["rendered_pdf"] != pdf:
        raise ValueError("Manuscript render flags do not match the requested gate")
    html = (output / "manuscript.html").read_text()
    for marker in ("{{", "[@fig:", "[@sec:", "[@eq:", "[@tbl:"):
        if marker in html:
            raise ValueError(f"Unresolved manuscript marker: {marker}")
    if pdf and not (output / "manuscript.pdf").read_bytes().startswith(b"%PDF-"):
        raise ValueError("The PDF renderer did not produce a PDF")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", nargs="?", default="test", choices=[*COMMANDS, "gnn", "manuscript"])
    parser.add_argument(
        "--runtime",
        type=Path,
        default=Path(os.environ.get("GNN_RUNTIME_PYTHON", "build/gnn-runtime/bin/python")),
    )
    parser.add_argument(
        "--provision", action="store_true", help="Acquire the pinned GNN runtime; needs network"
    )
    parser.add_argument("--output", type=Path, default=Path("build/manuscript"))
    parser.add_argument("--pdf", action="store_true", help="Also render PDF; requires XeLaTeX")
    parser.add_argument("--record", type=Path, help="Record wall time and exit code as JSON")
    args = parser.parse_args(argv)
    if args.provision and args.arm != "gnn":
        parser.error("--provision applies only to gnn")
    args.runtime = args.runtime.absolute()
    start = time.perf_counter()
    result = 0
    steps = commands(args.arm, args.runtime, args.output.resolve(), args.pdf)
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    if args.arm == "gnn":
        env["GNN_RUNTIME_PYTHON"] = str(args.runtime)
    try:
        if args.provision:
            provision(args.runtime)
        for command in steps:
            print("+", " ".join(command), flush=True)
            subprocess.run(command, check=True, stdin=subprocess.DEVNULL, env=env)
        if args.arm == "manuscript":
            verify_render(args.output, args.pdf)
        if args.arm == "gnn":
            verify_gnn_version(args.runtime)
    except subprocess.CalledProcessError as error:
        result = error.returncode
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        result = 1
    seconds = round(time.perf_counter() - start, 3)
    print(f"{args.arm}: {seconds:.3f}s, exit {result}", flush=True)
    if args.record:
        report = json.loads(args.record.read_text()) if args.record.exists() else {}
        report[args.arm] = {"seconds": seconds, "exit_code": result, "commands": steps}
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(report, indent=2) + "\n")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
