from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from payment_ops_hardening.modeling import (
    ExperimentConfig,
    _bootstrap_capture_delta,
    _feature_columns,
    prepare_leakage_safe_frame,
    split_chronologically,
)


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        timestamp_column="Timestamp",
        target_column="Is Laundering",
        positive_values=("1",),
        amount_column="Amount Paid",
        categorical_columns=(
            "From Bank",
            "To Bank",
            "Payment Currency",
            "Receiving Currency",
            "Payment Format",
        ),
        payer_column="Account",
        payee_column="Account.1",
        corridor_from_column="From Bank",
        corridor_to_column="To Bank",
        max_rows=None,
        train_fraction=0.5,
        validation_fraction=0.25,
        calibration_fraction=0.5,
        review_fraction=0.1,
        bootstrap_repeats=100,
        bootstrap_sample_size=100,
        bootstrap_method="time_block",
        bootstrap_block_seconds=3600,
        random_seed=7,
        candidate_backend="catboost",
        use_gpu=False,
        gpu_devices=None,
        max_iterations=10,
        candidate_weight_modes=("none",),
        min_capture_delta=0.0,
        max_brier_regression=0.01,
        target_maturity_seconds=0,
        exclude_bank_identity_categoricals=False,
    )


def _raw_frame(rows: int = 240) -> pd.DataFrame:
    event_time = pd.date_range("2022-01-01", periods=rows // 4, freq="h", tz="UTC")
    timestamps = np.repeat(event_time.to_numpy(), 4)[:rows]
    return pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(timestamps).strftime("%Y-%m-%d %H:%M:%S"),
            "From Bank": [f"B{i % 3}" for i in range(rows)],
            "Account": [f"A{i % 11}" for i in range(rows)],
            "To Bank": [f"B{(i + 1) % 3}" for i in range(rows)],
            "Account.1": [f"Z{i % 13}" for i in range(rows)],
            "Amount Paid": [float(i % 29 + 1) for i in range(rows)],
            "Amount Received": [float(i % 29 + 1) for i in range(rows)],
            "Payment Currency": ["USD"] * rows,
            "Receiving Currency": ["USD"] * rows,
            "Payment Format": ["Wire" if i % 2 else "ACH" for i in range(rows)],
            "Is Laundering": ["1" if i % 17 == 0 else "0" for i in range(rows)],
        }
    )


def test_raw_account_columns_are_not_model_features() -> None:
    config = _config()
    categorical, numeric = _feature_columns(config)
    assert "Account" not in categorical + numeric
    assert "Account.1" not in categorical + numeric


def test_future_label_changes_do_not_change_prior_features() -> None:
    config = _config()
    raw = _raw_frame()
    baseline = prepare_leakage_safe_frame(raw, config)
    altered = raw.copy()
    altered.loc[160:, "Is Laundering"] = "1"
    perturbed = prepare_leakage_safe_frame(altered, config)
    categorical, numeric = _feature_columns(config)
    columns = categorical + numeric
    pd.testing.assert_frame_equal(
        baseline.loc[:159, columns].reset_index(drop=True),
        perturbed.loc[:159, columns].reset_index(drop=True),
        check_dtype=True,
    )


def test_future_transaction_changes_do_not_change_prior_features() -> None:
    config = _config()
    raw = _raw_frame()
    baseline = prepare_leakage_safe_frame(raw, config)
    altered = raw.copy()
    altered.loc[160:, "Amount Paid"] = 1_000_000.0
    perturbed = prepare_leakage_safe_frame(altered, config)
    categorical, numeric = _feature_columns(config)
    columns = categorical + numeric
    pd.testing.assert_frame_equal(
        baseline.loc[:159, columns].reset_index(drop=True),
        perturbed.loc[:159, columns].reset_index(drop=True),
        check_dtype=True,
    )


def test_timestamp_groups_do_not_cross_partition_boundaries() -> None:
    frame = prepare_leakage_safe_frame(_raw_frame(), _config())
    train, validation, test = split_chronologically(frame, _config())
    assert train["event_time"].max() < validation["event_time"].min()
    assert validation["event_time"].max() < test["event_time"].min()


def test_bank_identity_ablation_drops_raw_bank_categoricals() -> None:
    config = replace(_config(), exclude_bank_identity_categoricals=True)
    categorical, _ = _feature_columns(config)
    assert "From Bank" not in categorical
    assert "To Bank" not in categorical
    assert "bank_pair" not in categorical
    assert "Payment Format" in categorical


def test_time_block_bootstrap_returns_finite_interval() -> None:
    labels = np.array([0, 1] * 100)
    baseline = np.linspace(0.0, 1.0, len(labels))
    candidate = np.roll(baseline, 5)
    timestamps = pd.date_range(
        "2022-01-01", periods=len(labels), freq="5min", tz="UTC"
    ).to_numpy(dtype="datetime64[ns]")
    low, high = _bootstrap_capture_delta(
        labels,
        baseline,
        candidate,
        review_fraction=0.1,
        repeats=100,
        seed=7,
        sample_size=100,
        event_times=timestamps,
        method="time_block",
        block_seconds=3600,
    )
    assert np.isfinite(low)
    assert np.isfinite(high)
