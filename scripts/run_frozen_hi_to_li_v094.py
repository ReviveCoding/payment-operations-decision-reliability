#!/usr/bin/env python
"""Evaluate frozen v0.9.4 HI-Medium models on a held-out LI-Medium stream."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _load_v094_helpers(repo: Path) -> Any:
    path = repo / "scripts" / "run_champion_challenger_v094.py"
    spec = importlib.util.spec_from_file_location(
        "paymentops_v094_runner_helpers",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load v0.9.4 helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summary(path: Path, result: dict[str, Any]) -> None:
    baseline = result["metrics"]["baseline"]
    candidate = result["metrics"]["candidate"]
    decision = result["promotion_decision"]
    stability = result["bootstrap"]["stability_diagnostic"]
    lines = [
        "# PaymentOps AML v0.9.4 Frozen HI-Medium-to-LI-Medium Transfer",
        "",
        f"- Source run: `{result['source_run']}`",
        f"- Target test rows: {result['target_test_rows']}",
        f"- Review budget: {result['review_fraction']:.2%}",
        f"- Champion: **{decision['champion']}**",
        "",
        "| Metric | Baseline | Frozen CatBoost candidate | Delta |",
        "|---|---:|---:|---:|",
        (
            f"| Capture at review budget | {baseline['capture_at_budget']:.4%} | "
            f"{candidate['capture_at_budget']:.4%} | "
            f"{decision['capture_delta']:+.4%} |"
        ),
        (
            f"| Average precision | {baseline['average_precision']:.6f} | "
            f"{candidate['average_precision']:.6f} | "
            f"{candidate['average_precision'] - baseline['average_precision']:+.6f} |"
        ),
        (
            f"| Brier score | {baseline['brier']:.6f} | "
            f"{candidate['brier']:.6f} | {decision['brier_delta']:+.6f} |"
        ),
        "",
        (
            "- Primary transaction-mass CI: "
            f"[{decision['capture_delta_ci_low']:+.4%}, "
            f"{decision['capture_delta_ci_high']:+.4%}]"
        ),
        (
            "- Secondary equal-calendar-time CI: "
            f"[{stability['capture_delta_ci_low']:+.4%}, "
            f"{stability['capture_delta_ci_high']:+.4%}]"
        ),
        "- No LI-Medium labels were used for fitting, tuning, or calibration.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--target-input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo / "src"))
    helpers = _load_v094_helpers(repo)

    import joblib
    import numpy as np
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
    if not source_metrics_path.is_file():
        raise FileNotFoundError(source_metrics_path)
    if not target_input.is_file():
        raise FileNotFoundError(target_input)

    source_metrics = json.loads(source_metrics_path.read_text(encoding="utf-8"))
    # The source experiment report embeds the complete v0.9.4 protocol config.
    protocol_config = source_metrics["config"]
    accepted = {
        field.name for field in __import__("dataclasses").fields(ExperimentConfig)
    }
    experiment_kwargs = {
        key: value for key, value in protocol_config.items() if key in accepted
    }
    experiment_kwargs["bootstrap_method"] = "time_block"
    experiment_kwargs["bootstrap_block_seconds"] = int(
        protocol_config["stability_diagnostic"]["block_seconds"]
    )
    config = ExperimentConfig(**experiment_kwargs)
    primary = protocol_config["primary_bootstrap"]
    stability = protocol_config["stability_diagnostic"]

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
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing frozen source artifacts: " + "; ".join(missing)
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
    baseline_probability = _apply_sigmoid_calibrator(
        baseline_calibrator,
        baseline_raw,
    )
    candidate_probability = _apply_sigmoid_calibrator(
        candidate_calibrator,
        candidate_raw,
    )
    baseline_metrics = _metrics(
        target_test["target"],
        baseline_probability,
        config.review_fraction,
    )
    candidate_metrics = _metrics(
        target_test["target"],
        candidate_probability,
        config.review_fraction,
    )

    primary_low, primary_high, primary_audit = helpers._transaction_mass_bootstrap(
        target_test["target"],
        baseline_probability,
        candidate_probability,
        target_test["event_time"].to_numpy(dtype="datetime64[ns]"),
        config.review_fraction,
        int(primary["repeats"]),
        int(primary["random_seed"]),
        int(primary["sample_size_target"]),
        int(primary["target_block_rows"]),
        _metrics,
        np,
    )
    stability_low, stability_high = _bootstrap_capture_delta(
        target_test["target"],
        baseline_probability,
        candidate_probability,
        config.review_fraction,
        int(stability["repeats"]),
        int(stability["random_seed"]),
        int(stability["sample_size_target"]),
        event_times=target_test["event_time"].to_numpy(dtype="datetime64[ns]"),
        method="time_block",
        block_seconds=int(stability["block_seconds"]),
    )
    decision = _promotion_decision(
        baseline_metrics,
        candidate_metrics,
        primary_low,
        primary_high,
        config,
    )

    report_dir = destination / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    source_provenance_path = source_run / "reports" / "candidate_model_metadata.json"
    source_provenance = (
        json.loads(source_provenance_path.read_text(encoding="utf-8"))
        if source_provenance_path.is_file()
        else {}
    )
    stability_payload = {
        "method": "paired_temporal_block_bootstrap",
        "unit": "equal_calendar_time_block",
        "block_seconds": int(stability["block_seconds"]),
        "repeats": int(stability["repeats"]),
        "sample_size_target": int(stability["sample_size_target"]),
        "capture_delta_ci_low": stability_low,
        "capture_delta_ci_high": stability_high,
        "role": "secondary_stability_diagnostic_only",
    }
    result = {
        "schema_version": "1.0",
        "status": "PASS",
        "protocol": "frozen_hi_medium_to_li_medium_transfer",
        "protocol_id": protocol_config["protocol_id"],
        "source_run": str(source_run),
        "target_input": str(target_input),
        "candidate_provenance": source_provenance,
        "source_model_sha256": {path.name: _sha256(path) for path in required_paths},
        "feature_schema": {"categorical": categorical, "numeric": numeric},
        "target_test_rows": len(target_test),
        "target_test_time_range": [
            str(target_test["event_time"].min()),
            str(target_test["event_time"].max()),
        ],
        "review_fraction": config.review_fraction,
        "bootstrap": {
            "primary": {
                **primary_audit,
                "capture_delta_ci_low": primary_low,
                "capture_delta_ci_high": primary_high,
                "role": "pre_registered_primary_promotion_inference",
            },
            "stability_diagnostic": stability_payload,
        },
        "metrics": {
            "baseline": asdict(baseline_metrics),
            "candidate": asdict(candidate_metrics),
        },
        "promotion_decision": asdict(decision),
        "claim_boundary": [
            "Models and sigmoid calibrators were frozen from the source HI-Medium run.",
            "LI-Medium transaction history was used only as an online, label-free feature state.",
            "No LI-Medium labels were used for fitting, tuning, or calibration.",
            "Promotion uses the pre-registered transaction-mass bootstrap.",
            "Equal-calendar-time bootstrap is a secondary stability diagnostic.",
            "This is IBM AML synthetic benchmark evidence, not production banking evidence.",
        ],
    }
    _write_json(report_dir / "frozen_transfer_metrics.json", result)
    _write_json(
        report_dir / "transaction_mass_bootstrap_audit.json",
        primary_audit,
    )
    _write_json(
        report_dir / "calendar_stability_bootstrap_diagnostic.json",
        stability_payload,
    )

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
    predictions.to_csv(
        report_dir / "frozen_transfer_test_predictions.csv",
        index=False,
    )
    _summary(report_dir / "frozen_transfer_summary.md", result)

    print(json.dumps(asdict(decision), indent=2))
    print(f"Evidence: {report_dir / 'frozen_transfer_metrics.json'}")


if __name__ == "__main__":
    main()
