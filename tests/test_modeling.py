from __future__ import annotations

import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from payment_ops_hardening.modeling import (  # noqa: E402
    ExperimentConfig,
    ModelingError,
    prepare_leakage_safe_frame,
    split_chronologically,
)


def _frame(rows: int = 400):
    timestamp = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "TxDateTime": timestamp.astype(str),
            "TxSts": ["RJCT" if index % 13 == 0 else "ACSP" for index in range(rows)],
            "InstdAmt": [100.0 + index for index in range(rows)],
            "Currency": ["USD" if index % 2 else "EUR" for index in range(rows)],
            "DbtrAgtBIC": ["DBTR1" if index % 3 else "DBTR2" for index in range(rows)],
            "CdtrAgtBIC": ["CDTR1" if index % 4 else "CDTR2" for index in range(rows)],
            "DbtrCountry": ["GB"] * rows,
            "CdtrCountry": ["US" if index % 2 else "GB" for index in range(rows)],
            "PurposeCode": ["TREA" if index % 5 else "SALA" for index in range(rows)],
            "RejectionReason": [
                "AC01" if index % 13 == 0 else "" for index in range(rows)
            ],
            "StatusTimestamp": timestamp.astype(str),
            "ProcessingTimeSecs": [index % 8 + 1 for index in range(rows)],
        }
    )


def test_feature_builder_excludes_known_post_outcome_columns() -> None:
    config = ExperimentConfig(bootstrap_repeats=100)
    result = prepare_leakage_safe_frame(_frame(), config)
    assert "RejectionReason" in result.attrs["protected_columns"]
    assert "payer_prior_count_24h" in result
    assert "pair_amount_ratio" in result
    train, validation, test = split_chronologically(result, config)
    assert train["event_time"].max() < validation["event_time"].min()
    assert validation["event_time"].max() < test["event_time"].min()


def test_config_rejects_unknown_fields(tmp_path: Path) -> None:
    from payment_ops_hardening.modeling import load_config

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"unknown": 1}), encoding="utf-8")
    with pytest.raises(ModelingError, match="unknown experiment"):
        load_config(path)


def test_ibm_aml_schema_uses_account_velocity_and_bank_corridors() -> None:
    rows = 400
    timestamp = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "Timestamp": timestamp.astype(str),
            "From Bank": ["B1" if index % 3 else "B2" for index in range(rows)],
            "Account": [f"A{index % 11}" for index in range(rows)],
            "To Bank": ["B3" if index % 4 else "B1" for index in range(rows)],
            "Account.1": [f"Z{index % 17}" for index in range(rows)],
            "Amount Paid": [100.0 + index for index in range(rows)],
            "Payment Currency": [
                "USD" if index % 2 else "EUR" for index in range(rows)
            ],
            "Receiving Currency": [
                "USD" if index % 2 else "EUR" for index in range(rows)
            ],
            "Payment Format": [
                "Wire" if index % 5 else "Cheque" for index in range(rows)
            ],
            "Is Laundering": [1 if index % 13 == 0 else 0 for index in range(rows)],
        }
    )
    config = ExperimentConfig(
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
        bootstrap_repeats=100,
    )
    result = prepare_leakage_safe_frame(frame, config)
    assert result["bank_pair"].iloc[0].startswith("B")
    assert "payer_prior_count_24h" in result
    assert "payee_prior_count_24h" in result
    assert "account_pair_prior_count_24h" in result
    assert "source_entity" in result
    assert result["is_cross_border"].isin([0, 1]).all()
