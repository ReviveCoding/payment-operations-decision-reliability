from __future__ import annotations

import sys
from pathlib import Path

from scripts.qualify_local import _path_for_evidence, _run


def test_evidence_path_allows_external_log_directory(tmp_path: Path) -> None:
    external = tmp_path / "outside logs"
    external.mkdir()
    evidence: list[dict] = []

    _run("echo_ok", [sys.executable, "-c", "print('ok')"], external, evidence)

    assert evidence[0]["exit_code"] == 0
    assert evidence[0]["stdout"].endswith("echo_ok.stdout.log")
    assert Path(evidence[0]["stdout"]).is_absolute()
    assert (external / "echo_ok.stdout.log").read_text(encoding="utf-8") == "ok\n"


def test_evidence_path_is_relative_for_repository_log_directory(tmp_path: Path) -> None:
    # A repo-internal log path should remain compact and reproducible in evidence.
    from scripts.qualify_local import ROOT

    inside = ROOT / "qualification_logs_test"
    inside.mkdir(exist_ok=True)
    try:
        path = inside / "sample.log"
        path.write_text("x", encoding="utf-8")
        assert _path_for_evidence(path) == "qualification_logs_test/sample.log"
    finally:
        for child in inside.glob("*"):
            child.unlink()
        inside.rmdir()
