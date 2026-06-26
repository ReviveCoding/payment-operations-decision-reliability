from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from payment_ops_hardening.release_security import (
    finalize_release_security,
    verify_release_security,
)

KEY = "qualification-benchmark-key-material-32-bytes"


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _build_release(root: Path) -> list[str]:
    paths = ["models/model.bin", "data/features.csv", "reports/decision.json"]
    (root / "models").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "reports").mkdir()
    (root / paths[0]).write_bytes(b"model-bytes")
    (root / paths[1]).write_text("case_id,score\na,0.1\nb,0.2\n", encoding="utf-8")
    (root / paths[2]).write_text(
        json.dumps({"run_id": "benchmark", "release_state": "PROMOTE"}),
        encoding="utf-8",
    )
    (root / "release_manifest.json").write_text(
        json.dumps({"run_id": "benchmark", "release_state": "PROMOTE"}),
        encoding="utf-8",
    )
    finalize_release_security(
        root,
        required_paths=paths,
        content_contracts={
            "data/features.csv": {
                "type": "csv",
                "required_columns": ["case_id", "score"],
                "min_rows": 2,
            }
        },
        key=KEY,
        key_id="benchmark-key",
        require_signature=True,
        release_sequence=1,
        allowed_release_states=["PROMOTE"],
        reject_untracked_files=True,
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="benchmark release verification stability"
    )
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.iterations < 1 or args.workers < 1:
        parser.error("iterations and workers must be positive")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        paths = _build_release(root)
        kwargs = {
            "key": KEY,
            "require_signature": True,
            "expected_key_id": "benchmark-key",
            "minimum_release_sequence": 1,
            "expected_paths": paths,
            "allowed_release_states": ["PROMOTE"],
            "reject_untracked_files": True,
        }
        concurrent_kwargs = {**kwargs, "lock_timeout": 5.0}
        verify_release_security(root, **kwargs)
        fd_before = (
            len(os.listdir("/proc/self/fd")) if Path("/proc/self/fd").exists() else None
        )
        tracemalloc.start()
        sequential: list[float] = []
        errors: list[str] = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            try:
                verify_release_security(root, **kwargs)
            except (
                Exception
            ) as exc:  # benchmark records unexpected public-entrypoint errors
                errors.append(type(exc).__name__)
            sequential.append((time.perf_counter() - started) * 1000)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        fd_after = (
            len(os.listdir("/proc/self/fd")) if Path("/proc/self/fd").exists() else None
        )

        concurrent: list[float] = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            for _ in range(args.iterations):

                def run_once() -> float:
                    started = time.perf_counter()
                    verify_release_security(root, **concurrent_kwargs)
                    return (time.perf_counter() - started) * 1000

                futures.append(executor.submit(run_once))
            for future in as_completed(futures):
                try:
                    concurrent.append(future.result())
                except Exception as exc:
                    errors.append(type(exc).__name__)

        result = {
            "schema_version": "1.0",
            "iterations": args.iterations,
            "workers": args.workers,
            "errors": errors,
            "sequential_ms": {
                "mean": round(statistics.fmean(sequential), 4),
                "p50": round(_percentile(sequential, 0.50), 4),
                "p95": round(_percentile(sequential, 0.95), 4),
                "p99": round(_percentile(sequential, 0.99), 4),
            },
            "concurrent_ms": {
                "mean": round(statistics.fmean(concurrent), 4),
                "p50": round(_percentile(concurrent, 0.50), 4),
                "p95": round(_percentile(concurrent, 0.95), 4),
                "p99": round(_percentile(concurrent, 0.99), 4),
            },
            "tracemalloc_current_bytes": current,
            "tracemalloc_peak_bytes": peak,
            "file_descriptors_before": fd_before,
            "file_descriptors_after": fd_after,
        }
        print(json.dumps(result, indent=2))
        if errors or (fd_before is not None and fd_after != fd_before):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
