from __future__ import annotations

import argparse
import json
import importlib.metadata
import tomllib
import os
import platform
import subprocess
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


PROJECT_QUALITY_PATHS = ("src", "tests", "scripts")


def _path_for_evidence(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _run(name: str, command: list[str], log_dir: Path, evidence: list[dict]) -> None:
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=os.environ.copy(),
    )
    duration = round(time.perf_counter() - started, 3)
    stdout_path = log_dir / f"{name}.stdout.log"
    stderr_path = log_dir / f"{name}.stderr.log"
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    record = {
        "name": name,
        "command": command,
        "exit_code": result.returncode,
        "duration_seconds": duration,
        "stdout": _path_for_evidence(stdout_path),
        "stderr": _path_for_evidence(stderr_path),
    }
    evidence.append(record)
    print(f"[{name}] exit={result.returncode} duration={duration:.3f}s")
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="run canonical local/CI qualification gates"
    )
    parser.add_argument("--profile", choices=["core", "standard"], default="standard")
    parser.add_argument("--log-dir", default="qualification_logs")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--full-suite-repeats", type=int)
    parser.add_argument("--smoke-repeats", type=int)
    args = parser.parse_args()

    full_repeats = args.full_suite_repeats or (1 if args.profile == "core" else 2)
    smoke_repeats = args.smoke_repeats or (2 if args.profile == "core" else 3)
    if full_repeats < 1 or smoke_repeats < 1:
        parser.error("repeat counts must be positive")

    log_dir = (ROOT / args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    evidence: list[dict] = []
    python = sys.executable
    expected_version = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["version"]
    installed_version = importlib.metadata.version("payment-ops-hardening-overlay")
    if installed_version != expected_version:
        raise SystemExit(
            f"installed package version {installed_version} does not match source {expected_version}"
        )

    _run("pip_check", [python, "-m", "pip", "check"], log_dir, evidence)
    script_name = (
        "payment-ops-hardening-validate.exe"
        if os.name == "nt"
        else "payment-ops-hardening-validate"
    )
    adjacent_cli = Path(sys.executable).parent / script_name
    cli = (
        str(adjacent_cli)
        if adjacent_cli.is_file()
        else shutil.which("payment-ops-hardening-validate")
    )
    if cli is None:
        raise SystemExit(
            "installed console script payment-ops-hardening-validate was not found"
        )
    _run("cli_version", [cli, "--version"], log_dir, evidence)
    _run("cli_help", [cli, "--help"], log_dir, evidence)
    _run(
        "ruff_check",
        [python, "-m", "ruff", "check", "src", "tests", "scripts"],
        log_dir,
        evidence,
    )
    _run(
        "ruff_format",
        [python, "-m", "ruff", "format", "--check", "src", "tests", "scripts"],
        log_dir,
        evidence,
    )
    _run("mypy", [python, "-m", "mypy"], log_dir, evidence)
    for index in range(full_repeats):
        command = [python, "-m", "pytest"]
        if index == 0:
            command += [
                "--cov=payment_ops_hardening",
                "--cov-report=term-missing",
                "--cov-report=json:" + str(log_dir / "coverage.json"),
                "--cov-fail-under=80",
            ]
        _run(f"pytest_{index + 1}", command, log_dir, evidence)
    for index in range(smoke_repeats):
        _run(
            f"synthetic_validation_{index + 1}",
            [python, "scripts/validate_overlay.py"],
            log_dir,
            evidence,
        )
    if not args.skip_build:
        dist_dir = ROOT / "dist"
        if dist_dir.exists():
            shutil.rmtree(dist_dir)
        _run(
            "build",
            [python, "-m", "build", "--no-isolation"],
            log_dir,
            evidence,
        )
        wheels = sorted((ROOT / "dist").glob("*.whl"))
        sdists = sorted((ROOT / "dist").glob("*.tar.gz"))
        if len(wheels) != 1 or len(sdists) != 1:
            raise SystemExit("expected exactly one wheel and one sdist in dist/")
        _run(
            "verify_distribution",
            [python, "scripts/verify_distribution.py", str(wheels[0]), str(sdists[0])],
            log_dir,
            evidence,
        )

    summary = {
        "schema_version": "1.0",
        "profile": args.profile,
        "python": sys.version,
        "platform": platform.platform(),
        "commands": evidence,
        "status": "PASS",
    }
    (log_dir / "qualification_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "commands": len(evidence)}, indent=2))


if __name__ == "__main__":
    main()
