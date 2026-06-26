#!/usr/bin/env python
"""Evaluate frozen HI-trained v0.9.3 models on a chronologically held-out LI test stream."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--source-run", required=True, help="HI v0.9.3 run directory")
    parser.add_argument(
        "--target-input", required=True, help="LI-Small transaction CSV"
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo / "src"))
    import joblib
    import pandas as pd
    from catboost import CatBoostClassifier
    from payment_ops_hardening.modeling import (
        ExperimentConfig,
        _apply_sigmoid_calibrator,
        _bootstrap_capture_delta,
        _feature_columns,
        _metrics,
        _predict_probability,
        _promotion_decision,
        prepare_leakage_safe_frame,
        split_chronologically,
    )

    source_run = Path(args.source_run).resolve()
    target_input = Path(args.target_input).resolve()
    destination = Path(args.output_dir).resolve()
    source_metrics_path = source_run / "reports" / "experiment_metrics.json"
    if not source_metrics_path.exists():
        raise FileNotFoundError(source_metrics_path)
    source_metrics = json.loads(source_metrics_path.read_text(encoding="utf-8"))
    config = ExperimentConfig(**source_metrics["config"])

    source_models = source_run / "models"
    baseline_path = source_models / "baseline_logistic.joblib"
    baseline_calibrator_path = source_models / "baseline_sigmoid_calibrator.joblib"
    candidate_calibrator_path = source_models / "candidate_sigmoid_calibrator.joblib"
    candidate_path = source_models / "candidate_catboost.cbm"
    required_paths = [
        baseline_path,
        baseline_calibrator_path,
        candidate_calibrator_path,
        candidate_path,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Frozen transfer needs a v0.9.3 source run with CatBoost .cbm artifacts. Missing: "
            + "; ".join(missing)
        )

    baseline = joblib.load(baseline_path)
    baseline_calibrator = joblib.load(baseline_calibrator_path)
    candidate_calibrator = joblib.load(candidate_calibrator_path)
    candidate = CatBoostClassifier()
    candidate.load_model(str(candidate_path))

    raw = pd.read_csv(target_input)
    target_frame = prepare_leakage_safe_frame(raw, config)
    _, _, target_test = split_chronologically(target_frame, config)
    categorical, numeric = _feature_columns(config)
    columns = categorical + numeric
    missing_features = sorted(set(columns) - set(target_test.columns))
    if missing_features:
        raise RuntimeError(f"Target feature schema mismatch: {missing_features}")

    baseline_raw = _predict_probability(baseline, target_test, columns)
    candidate_raw = _predict_probability(candidate, target_test, columns)
    baseline_probability = _apply_sigmoid_calibrator(baseline_calibrator, baseline_raw)
    candidate_probability = _apply_sigmoid_calibrator(
        candidate_calibrator, candidate_raw
    )
    baseline_metrics = _metrics(
        target_test["target"], baseline_probability, config.review_fraction
    )
    candidate_metrics = _metrics(
        target_test["target"], candidate_probability, config.review_fraction
    )
    ci_low, ci_high = _bootstrap_capture_delta(
        target_test["target"],
        baseline_probability,
        candidate_probability,
        config.review_fraction,
        config.bootstrap_repeats,
        config.random_seed,
        config.bootstrap_sample_size,
        event_times=target_test["event_time"].to_numpy(dtype="datetime64[ns]"),
        method=config.bootstrap_method,
        block_seconds=config.bootstrap_block_seconds,
    )
    decision = _promotion_decision(
        baseline_metrics, candidate_metrics, ci_low, ci_high, config
    )

    destination.mkdir(parents=True, exist_ok=True)
    report_dir = destination / "reports"
    report_dir.mkdir(exist_ok=True)
    source_provenance_path = source_run / "reports" / "candidate_model_metadata.json"
    source_provenance = (
        json.loads(source_provenance_path.read_text(encoding="utf-8"))
        if source_provenance_path.exists()
        else {}
    )

    result = {
        "schema_version": "1.0",
        "status": "PASS",
        "protocol": "frozen_hi_to_li_transfer",
        "source_run": str(source_run),
        "target_input": str(target_input),
        "candidate_provenance": source_provenance,
        "source_model_sha256": {path.name: sha256(path) for path in required_paths},
        "feature_schema": {"categorical": categorical, "numeric": numeric},
        "target_test_rows": len(target_test),
        "target_test_time_range": [
            str(target_test["event_time"].min()),
            str(target_test["event_time"].max()),
        ],
        "review_fraction": config.review_fraction,
        "bootstrap": {
            "method": (
                "paired_temporal_block_bootstrap"
                if config.bootstrap_method == "time_block"
                else "paired_iid_row_bootstrap"
            ),
            "unit": "time_block" if config.bootstrap_method == "time_block" else "row",
            "block_seconds": config.bootstrap_block_seconds
            if config.bootstrap_method == "time_block"
            else None,
            "repeats": config.bootstrap_repeats,
            "sample_size_target": config.bootstrap_sample_size,
        },
        "metrics": {
            "baseline": asdict(baseline_metrics),
            "candidate": asdict(candidate_metrics),
        },
        "promotion_decision": asdict(decision),
        "claim_boundary": [
            "Models and sigmoid calibrators were frozen from the source HI run.",
            "LI transaction history was used only as an online, label-free feature state.",
            "No LI labels were used for fitting, tuning, or calibration.",
            "This is IBM AML synthetic benchmark evidence, not production banking evidence.",
        ],
    }
    write_json(report_dir / "frozen_transfer_metrics.json", result)

    predictions = target_test[[config.timestamp_column, "event_time", "target"]].copy()
    predictions["baseline_probability"] = baseline_probability
    predictions["candidate_probability_raw"] = candidate_raw
    predictions["candidate_probability"] = candidate_probability
    review_count = baseline_metrics.review_count
    predictions["baseline_review"] = False
    predictions.loc[
        predictions["baseline_probability"].rank(method="first", ascending=False)
        <= review_count,
        "baseline_review",
    ] = True
    predictions["candidate_review"] = False
    predictions.loc[
        predictions["candidate_probability"].rank(method="first", ascending=False)
        <= review_count,
        "candidate_review",
    ] = True
    predictions.to_csv(report_dir / "frozen_transfer_test_predictions.csv", index=False)

    summary = "\n".join(
        [
            "# PaymentOps AML v0.9.3 Frozen HI-to-LI Transfer",
            "",
            f"- Source run: `{source_run}`",
            f"- Target test rows: {len(target_test)}",
            f"- Review budget: {config.review_fraction:.2%}",
            f"- Champion: **{decision.champion}**",
            "",
            "| Metric | Baseline | Frozen CatBoost candidate | Delta |",
            "|---|---:|---:|---:|",
            f"| Capture at review budget | {baseline_metrics.capture_at_budget:.4%} | {candidate_metrics.capture_at_budget:.4%} | {decision.capture_delta:+.4%} |",
            f"| Average precision | {baseline_metrics.average_precision:.6f} | {candidate_metrics.average_precision:.6f} | {candidate_metrics.average_precision - baseline_metrics.average_precision:+.6f} |",
            f"| Brier score | {baseline_metrics.brier:.6f} | {candidate_metrics.brier:.6f} | {decision.brier_delta:+.6f} |",
            "",
            f"- Paired bootstrap CI: [{ci_low:+.4%}, {ci_high:+.4%}]",
            "- No LI labels were used for fitting, tuning, or calibration.",
        ]
    )
    (report_dir / "frozen_transfer_summary.md").write_text(
        summary + "\n", encoding="utf-8"
    )
    print(json.dumps(result["promotion_decision"], indent=2))
    print(f"Evidence: {report_dir / 'frozen_transfer_metrics.json'}")


if __name__ == "__main__":
    main()
