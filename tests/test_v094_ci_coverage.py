from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from payment_ops_hardening.modeling import (  # noqa: E402
    ExperimentConfig,
    run_champion_challenger,
)


def _ibm_transaction_frame(rows: int = 600):
    timestamps = pd.date_range(
        "2025-02-01 00:00:00",
        periods=rows,
        freq="min",
        tz="UTC",
    )
    return pd.DataFrame(
        {
            "Timestamp": timestamps.astype(str),
            "From Bank": [f"B{index % 4:02d}" for index in range(rows)],
            "Account": [f"A{index % 17:03d}" for index in range(rows)],
            "To Bank": [f"B{(index * 3 + 1) % 5:02d}" for index in range(rows)],
            "Account.1": [f"R{(index * 5 + 2) % 19:03d}" for index in range(rows)],
            "Amount Received": [float(80 + (index * 7) % 43) for index in range(rows)],
            "Receiving Currency": [
                "USD" if index % 3 else "EUR" for index in range(rows)
            ],
            "Amount Paid": [float(85 + (index * 11) % 47) for index in range(rows)],
            "Payment Currency": [
                "USD" if index % 2 else "EUR" for index in range(rows)
            ],
            "Payment Format": [
                ("Wire", "ACH", "Cash", "Cheque")[index % 4] for index in range(rows)
            ],
            "Is Laundering": ["1" if index % 11 == 0 else "0" for index in range(rows)],
        }
    )


def _extra_trees_config() -> ExperimentConfig:
    available = {field.name for field in fields(ExperimentConfig)}
    requested = {
        "timestamp_column": "Timestamp",
        "target_column": "Is Laundering",
        "positive_values": ("1",),
        "amount_column": "Amount Paid",
        "categorical_columns": (
            "From Bank",
            "To Bank",
            "Payment Currency",
            "Receiving Currency",
            "Payment Format",
        ),
        "payer_column": "Account",
        "payee_column": "Account.1",
        "corridor_from_column": "From Bank",
        "corridor_to_column": "To Bank",
        "train_fraction": 0.60,
        "validation_fraction": 0.20,
        "calibration_fraction": 0.50,
        "review_fraction": 0.10,
        "bootstrap_repeats": 100,
        "bootstrap_sample_size": None,
        "bootstrap_method": "iid_row",
        "candidate_backend": "extra_trees",
        "candidate_weight_modes": ("none",),
        "exclude_bank_identity_categoricals": True,
        "use_gpu": False,
        "gpu_devices": None,
        "max_iterations": 50,
        "random_seed": 20260625,
    }
    return ExperimentConfig(
        **{name: value for name, value in requested.items() if name in available}
    )


def test_extra_trees_champion_challenger_writes_auditable_evidence(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "ibm_transactions.csv"
    output_dir = tmp_path / "run"
    _ibm_transaction_frame().to_csv(input_path, index=False)

    result = run_champion_challenger(
        input_path=input_path,
        output_dir=output_dir,
        config=_extra_trees_config(),
    )

    assert result["status"] == "PASS"
    assert result["candidate_backend"] == "extra_trees"
    assert result["candidate_backend_family"] == "extra_trees"
    assert result["split_sizes"]["total"] == 600
    assert result["metrics"]["baseline"]["positive_count"] > 0
    assert result["metrics"]["candidate"]["positive_count"] > 0

    categorical = set(result["feature_schema"]["categorical"])
    assert "From Bank" not in categorical
    assert "To Bank" not in categorical
    assert "bank_pair" not in categorical
    assert {"Payment Currency", "Receiving Currency", "Payment Format"} <= categorical

    numeric = set(result["feature_schema"]["numeric"])
    assert {"received_amount_log1p", "paid_to_received_ratio"} <= numeric

    report_dir = output_dir / "reports"
    model_dir = output_dir / "models"

    for artifact in (
        model_dir / "baseline_logistic.joblib",
        model_dir / "candidate_extra_trees.joblib",
        model_dir / "candidate_sigmoid_calibrator.joblib",
        report_dir / "experiment_metrics.json",
        report_dir / "promotion_decision.json",
        report_dir / "test_predictions.csv",
        report_dir / "uplift_summary.md",
    ):
        assert artifact.is_file(), artifact

    recorded = json.loads(
        (report_dir / "experiment_metrics.json").read_text(encoding="utf-8")
    )
    assert recorded["status"] == "PASS"
    assert recorded["candidate_backend_family"] == "extra_trees"
    assert recorded["config"]["exclude_bank_identity_categoricals"] is True
