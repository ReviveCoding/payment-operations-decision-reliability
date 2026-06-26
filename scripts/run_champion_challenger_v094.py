#!/usr/bin/env python
"""Run the v0.9.4 pre-registered AML confirmatory experiment.

The v0.9.4 protocol is additive: it does not modify the v0.9.3 implementation
or its historical promotion decisions. Its primary promotion inference uses
paired contiguous transaction-mass blocks; equal-calendar 6-hour blocks are
reported as a separate stability diagnostic.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import hashlib
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


def _read_v094_config(
    path: Path, experiment_config_type: Any
) -> tuple[dict[str, Any], Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != "paymentops_aml_v094_medium_confirmatory":
        raise ValueError(
            "v0.9.4 config must declare protocol_id "
            "'paymentops_aml_v094_medium_confirmatory'"
        )
    if payload.get("primary_bootstrap", {}).get("method") != "transaction_mass_block":
        raise ValueError("v0.9.4 primary bootstrap must be transaction_mass_block")
    if payload.get("stability_diagnostic", {}).get("method") != "time_block":
        raise ValueError("v0.9.4 stability diagnostic must be time_block")

    accepted = {field.name for field in fields(experiment_config_type)}
    experiment_kwargs = {
        key: value for key, value in payload.items() if key in accepted
    }
    # Existing v0.9.3 ExperimentConfig does not know transaction_mass_block.
    # It is used only by v0.9.4's own primary bootstrap implementation.
    experiment_kwargs["bootstrap_method"] = "time_block"
    experiment_kwargs["bootstrap_block_seconds"] = int(
        payload["stability_diagnostic"]["block_seconds"]
    )
    return payload, experiment_config_type(**experiment_kwargs)


def _build_transaction_mass_blocks(
    event_times: Any,
    target_block_rows: int,
    np: Any,
) -> list[Any]:
    """Build contiguous timestamp-preserving blocks near a target row count."""
    if target_block_rows < 1_000:
        raise ValueError("target_block_rows must be at least 1000")

    times = np.asarray(event_times, dtype="datetime64[ns]")
    if len(times) == 0:
        raise ValueError("cannot block an empty test stream")
    nanoseconds = times.astype("int64")
    if (nanoseconds == np.iinfo("int64").min).any():
        raise ValueError("event_times contains invalid datetime values")
    if (nanoseconds[1:] < nanoseconds[:-1]).any():
        raise ValueError("event_times must be chronologically ordered")

    group_starts = np.r_[0, np.flatnonzero(nanoseconds[1:] != nanoseconds[:-1]) + 1]
    group_ends = np.r_[group_starts[1:], len(times)]
    minimum_tail_rows = max(1_000, target_block_rows // 2)

    blocks: list[Any] = []
    block_start = int(group_starts[0])
    block_rows = 0
    for group_start, group_end in zip(group_starts, group_ends, strict=True):
        group_rows = int(group_end - group_start)
        if (
            block_rows > 0
            and block_rows + group_rows > target_block_rows
            and block_rows >= minimum_tail_rows
        ):
            blocks.append(np.arange(block_start, int(group_start), dtype=int))
            block_start = int(group_start)
            block_rows = 0
        block_rows += group_rows

    if block_start < len(times):
        blocks.append(np.arange(block_start, len(times), dtype=int))

    if len(blocks) >= 2 and len(blocks[-1]) < minimum_tail_rows:
        blocks[-2] = np.concatenate([blocks[-2], blocks[-1]])
        blocks.pop()
    if len(blocks) < 2:
        raise ValueError("transaction-mass blocking produced fewer than two blocks")
    return blocks


def _quantiles(values: Any, np: Any) -> dict[str, float]:
    values = np.asarray(values)
    return {
        "min": float(np.min(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p025": float(np.quantile(values, 0.025)),
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p975": float(np.quantile(values, 0.975)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def _transaction_mass_bootstrap(
    labels: Any,
    baseline_probabilities: Any,
    candidate_probabilities: Any,
    event_times: Any,
    review_fraction: float,
    repeats: int,
    seed: int,
    sample_size_target: int,
    target_block_rows: int,
    metrics_fn: Any,
    np: Any,
) -> tuple[float, float, dict[str, Any]]:
    """Paired contiguous transaction-mass bootstrap with complete blocks."""
    labels = np.asarray(labels, dtype=int)
    baseline = np.asarray(baseline_probabilities, dtype=float)
    candidate = np.asarray(candidate_probabilities, dtype=float)
    if not (len(labels) == len(baseline) == len(candidate)):
        raise ValueError("bootstrap arrays must have identical length")
    if labels.sum() == 0:
        raise ValueError("test stream contains no positives")
    if repeats < 100:
        raise ValueError("repeats must be at least 100")
    if sample_size_target < target_block_rows:
        raise ValueError("sample_size_target must be at least target_block_rows")

    blocks = _build_transaction_mass_blocks(event_times, target_block_rows, np)
    generator = np.random.default_rng(seed)
    row_counts: list[int] = []
    positive_counts: list[int] = []
    blocks_drawn: list[int] = []
    deltas: list[float] = []

    for _ in range(repeats):
        selected: list[Any] = []
        selected_rows = 0
        while selected_rows < sample_size_target:
            block = blocks[int(generator.integers(0, len(blocks)))]
            selected.append(block)
            selected_rows += len(block)
        indices = np.concatenate(selected)
        sampled_labels = labels[indices]
        if sampled_labels.sum() == 0:
            continue

        baseline_metrics = metrics_fn(
            sampled_labels,
            baseline[indices],
            review_fraction,
        )
        candidate_metrics = metrics_fn(
            sampled_labels,
            candidate[indices],
            review_fraction,
        )
        deltas.append(
            float(
                candidate_metrics.capture_at_budget - baseline_metrics.capture_at_budget
            )
        )
        row_counts.append(int(len(indices)))
        positive_counts.append(int(sampled_labels.sum()))
        blocks_drawn.append(int(len(selected)))

    if len(deltas) < max(50, repeats // 2):
        raise ValueError("bootstrap produced insufficient positive resamples")

    delta_array = np.asarray(deltas, dtype=float)
    row_array = np.asarray(row_counts, dtype=int)
    positive_array = np.asarray(positive_counts, dtype=int)
    draws_array = np.asarray(blocks_drawn, dtype=int)
    block_size_array = np.asarray([len(block) for block in blocks], dtype=int)

    audit = {
        "method": "paired_transaction_mass_block_bootstrap",
        "unit": "contiguous_timestamp_preserving_transaction_mass_block",
        "target_block_rows": target_block_rows,
        "sample_size_target": sample_size_target,
        "repeats_requested": repeats,
        "repeats_used": int(len(deltas)),
        "random_seed": seed,
        "timestamp_groups_preserved": True,
        "block_count": int(len(blocks)),
        "block_rows": _quantiles(block_size_array, np),
        "blocks_drawn_per_replicate": _quantiles(draws_array, np),
        "replicate_rows": _quantiles(row_array, np),
        "replicate_positive_counts": _quantiles(positive_array, np),
        "capture_uplift": _quantiles(delta_array, np),
        "nonpositive_uplift_replicates": int((delta_array <= 0.0).sum()),
        "negative_uplift_replicates": int((delta_array < 0.0).sum()),
        "near_zero_uplift_replicates": int(
            np.isclose(delta_array, 0.0, atol=1e-15).sum()
        ),
    }
    return (
        float(np.quantile(delta_array, 0.025)),
        float(np.quantile(delta_array, 0.975)),
        audit,
    )


def _candidate_metadata(candidate: Any, backend: str, config: Any) -> dict[str, Any]:
    family, _, selected_weight_mode = backend.partition(":")
    observed: dict[str, Any] = {}
    if family == "catboost" and hasattr(candidate, "get_all_params"):
        observed = candidate.get_all_params()
    requested_task_type = "GPU" if config.use_gpu else "CPU"
    return {
        "artifact": (
            "models/candidate_catboost.cbm"
            if family == "catboost"
            else "models/candidate_extra_trees.joblib"
        ),
        "backend_family": family,
        "selected_weight_mode": selected_weight_mode,
        "runtime_class": f"{type(candidate).__module__}.{type(candidate).__name__}",
        "task_type_requested": requested_task_type,
        "task_type_observed": observed.get("task_type", requested_task_type),
        "devices_requested": config.gpu_devices,
        "devices_observed": observed.get("devices", config.gpu_devices),
        "iterations_observed": observed.get("iterations", config.max_iterations),
        "random_seed_observed": observed.get("random_seed", config.random_seed),
        "has_time_observed": observed.get("has_time", None),
    }


def _write_summary(path: Path, result: dict[str, Any]) -> None:
    baseline = result["metrics"]["baseline"]
    candidate = result["metrics"]["candidate"]
    decision = result["promotion_decision"]
    primary = result["bootstrap"]["primary"]
    stability = result["bootstrap"]["stability_diagnostic"]
    lines = [
        "# PaymentOps AML v0.9.4 Confirmatory Result",
        "",
        f"- Protocol: `{result['protocol_id']}`",
        f"- Input rows: {result['split_sizes']['total']}",
        f"- Review budget: {result['review_fraction']:.2%}",
        f"- Candidate backend: {result['candidate_backend']}",
        f"- Champion: **{decision['champion']}**",
        f"- Decision: {decision['reason']}",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "|---|---:|---:|---:|",
        (
            f"| Average precision | {baseline['average_precision']:.6f} | "
            f"{candidate['average_precision']:.6f} | "
            f"{candidate['average_precision'] - baseline['average_precision']:+.6f} |"
        ),
        (
            f"| ROC-AUC | {baseline['roc_auc'] if baseline['roc_auc'] is not None else 'n/a'} | "
            f"{candidate['roc_auc'] if candidate['roc_auc'] is not None else 'n/a'} | — |"
        ),
        (
            f"| Brier score (lower better) | {baseline['brier']:.6f} | "
            f"{candidate['brier']:.6f} | {decision['brier_delta']:+.6f} |"
        ),
        (
            f"| Capture at review budget | {baseline['capture_at_budget']:.4%} | "
            f"{candidate['capture_at_budget']:.4%} | "
            f"{decision['capture_delta']:+.4%} |"
        ),
        "",
        "## Pre-registered primary promotion inference",
        "",
        (
            "- Method: paired contiguous transaction-mass block bootstrap "
            f"({primary['target_block_rows']:,}-row target blocks)."
        ),
        (
            "- Capture uplift 95% CI: "
            f"[{decision['capture_delta_ci_low']:+.4%}, "
            f"{decision['capture_delta_ci_high']:+.4%}]"
        ),
        "",
        "## Secondary equal-calendar-time stability diagnostic",
        "",
        (f"- Method: paired {stability['block_seconds']}-second calendar blocks."),
        (
            "- Capture uplift 95% CI: "
            f"[{stability['capture_delta_ci_low']:+.4%}, "
            f"{stability['capture_delta_ci_high']:+.4%}]"
        ),
        (
            "- This diagnostic does not replace the primary pre-registered "
            "promotion decision."
        ),
        "",
        "## Claim boundary",
        "",
        "- Result applies only to the supplied IBM AML synthetic dataset and declared chronological split.",
        "- The no-bank configuration excludes direct raw bank/corridor categorical identifiers.",
        "- Bank-pair historical aggregates remain label-free temporal features.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo / "src"))

    import joblib
    import numpy as np
    import pandas as pd
    from payment_ops_hardening.modeling import (
        ExperimentConfig,
        _apply_sigmoid_calibrator,
        _bootstrap_capture_delta,
        _fit_baseline,
        _fit_candidate,
        _fit_sigmoid_calibrator,
        _feature_columns,
        _metrics,
        _predict_probability,
        _promotion_decision,
        prepare_leakage_safe_frame,
        split_chronologically,
        split_validation_for_tuning_and_calibration,
    )

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    config_path = Path(args.config).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    protocol_config, config = _read_v094_config(config_path, ExperimentConfig)
    primary = protocol_config["primary_bootstrap"]
    stability = protocol_config["stability_diagnostic"]
    if not protocol_config.get("exclude_bank_identity_categoricals", False):
        raise ValueError(
            "v0.9.4 confirmatory protocol must lock no-bank-identity features"
        )

    report_dir = output_dir / "reports"
    model_dir = output_dir / "models"
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    preregistration = {
        "protocol_id": protocol_config["protocol_id"],
        "protocol_version": protocol_config["protocol_version"],
        "pre_registered_at_utc": protocol_config["pre_registered_at_utc"],
        "protocol_config_sha256": _sha256(config_path),
        "input_sha256": _sha256(input_path),
        "input_bytes": input_path.stat().st_size,
        "implementation_sha256": _sha256(Path(__file__).resolve()),
        "modeling_sha256": _sha256(
            repo / "src" / "payment_ops_hardening" / "modeling.py"
        ),
        "promotion_analysis": "primary_transaction_mass_block_bootstrap",
        "stability_analysis": "secondary_equal_calendar_time_block_bootstrap",
        "config": protocol_config,
    }
    _write_json(report_dir / "pre_registered_protocol.json", preregistration)

    raw = pd.read_csv(input_path)
    frame = prepare_leakage_safe_frame(raw, config)
    train, validation, test = split_chronologically(frame, config)
    tuning, calibration = split_validation_for_tuning_and_calibration(
        validation, config
    )

    categorical, numeric = _feature_columns(config)
    columns = categorical + numeric
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"Feature engineering failed to create: {missing}")

    baseline = _fit_baseline(train, categorical, numeric)
    candidate, backend = _fit_candidate(
        train,
        tuning,
        categorical,
        numeric,
        config,
    )
    baseline_calibration_raw = _predict_probability(baseline, calibration, columns)
    candidate_calibration_raw = _predict_probability(candidate, calibration, columns)
    baseline_calibrator = _fit_sigmoid_calibrator(
        baseline_calibration_raw,
        calibration["target"],
    )
    candidate_calibrator = _fit_sigmoid_calibrator(
        candidate_calibration_raw,
        calibration["target"],
    )

    baseline_test_raw = _predict_probability(baseline, test, columns)
    candidate_test_raw = _predict_probability(candidate, test, columns)
    baseline_probability = _apply_sigmoid_calibrator(
        baseline_calibrator,
        baseline_test_raw,
    )
    candidate_probability = _apply_sigmoid_calibrator(
        candidate_calibrator,
        candidate_test_raw,
    )
    baseline_metrics = _metrics(
        test["target"],
        baseline_probability,
        config.review_fraction,
    )
    candidate_metrics = _metrics(
        test["target"],
        candidate_probability,
        config.review_fraction,
    )

    primary_low, primary_high, primary_audit = _transaction_mass_bootstrap(
        test["target"],
        baseline_probability,
        candidate_probability,
        test["event_time"].to_numpy(dtype="datetime64[ns]"),
        config.review_fraction,
        int(primary["repeats"]),
        int(primary["random_seed"]),
        int(primary["sample_size_target"]),
        int(primary["target_block_rows"]),
        _metrics,
        np,
    )
    stability_low, stability_high = _bootstrap_capture_delta(
        test["target"],
        baseline_probability,
        candidate_probability,
        config.review_fraction,
        int(stability["repeats"]),
        int(stability["random_seed"]),
        int(stability["sample_size_target"]),
        event_times=test["event_time"].to_numpy(dtype="datetime64[ns]"),
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

    joblib.dump(baseline, model_dir / "baseline_logistic.joblib")
    joblib.dump(
        baseline_calibrator,
        model_dir / "baseline_sigmoid_calibrator.joblib",
    )
    joblib.dump(
        candidate_calibrator,
        model_dir / "candidate_sigmoid_calibrator.joblib",
    )
    backend_family, _, _weight_mode = backend.partition(":")
    if backend_family == "catboost":
        candidate_artifact = model_dir / "candidate_catboost.cbm"
        candidate.save_model(str(candidate_artifact))
    else:
        candidate_artifact = model_dir / "candidate_extra_trees.joblib"
        joblib.dump(candidate, candidate_artifact)

    provenance = _candidate_metadata(candidate, backend, config)
    _write_json(report_dir / "candidate_model_metadata.json", provenance)
    _write_json(
        report_dir / "feature_schema.json",
        {"categorical": categorical, "numeric": numeric},
    )
    _write_json(
        report_dir / "transaction_mass_bootstrap_audit.json",
        primary_audit,
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
    _write_json(
        report_dir / "calendar_stability_bootstrap_diagnostic.json",
        stability_payload,
    )

    result = {
        "schema_version": "1.0",
        "status": "PASS",
        "protocol_id": protocol_config["protocol_id"],
        "protocol_version": protocol_config["protocol_version"],
        "candidate_backend": backend,
        "candidate_backend_family": backend_family,
        "candidate_weight_mode": _weight_mode,
        "candidate_provenance": provenance,
        "input_path": str(input_path),
        "config": protocol_config,
        "feature_schema": {"categorical": categorical, "numeric": numeric},
        "excluded_outcome_columns": [],
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
        "split_sizes": {
            "train": len(train),
            "validation": len(validation),
            "tuning": len(tuning),
            "calibration": len(calibration),
            "test": len(test),
            "total": len(frame),
        },
        "time_ranges": {
            "train": [
                str(train["event_time"].min()),
                str(train["event_time"].max()),
            ],
            "validation": [
                str(validation["event_time"].min()),
                str(validation["event_time"].max()),
            ],
            "test": [
                str(test["event_time"].min()),
                str(test["event_time"].max()),
            ],
        },
        "pre_registration_artifact": "reports/pre_registered_protocol.json",
        "candidate_artifact": str(candidate_artifact.relative_to(output_dir)),
        "claim_boundary": [
            "This is a pre-registered v0.9.4 confirmatory protocol.",
            "Promotion uses only the primary transaction-mass bootstrap.",
            "The equal-calendar-time bootstrap is a separately reported stability diagnostic.",
            "Results apply only to IBM AML synthetic data and declared chronological splits.",
        ],
    }
    _write_json(report_dir / "experiment_metrics.json", result)
    _write_json(report_dir / "promotion_decision.json", asdict(decision))

    predictions = test[[config.timestamp_column, "event_time", "target"]].copy()
    predictions["baseline_probability"] = baseline_probability
    predictions["candidate_probability_raw"] = candidate_test_raw
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
    predictions.to_csv(report_dir / "test_predictions.csv", index=False)
    _write_summary(report_dir / "uplift_summary.md", result)

    print(json.dumps(asdict(decision), indent=2))
    print(f"Evidence: {report_dir / 'experiment_metrics.json'}")


if __name__ == "__main__":
    main()
