from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REQUIRED = [
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account.1",
    "Amount Paid",
    "Payment Currency",
    "Receiving Currency",
    "Payment Format",
    "Is Laundering",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="streaming profile for IBM AML transaction CSV"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()
    source = Path(args.input)
    output = Path(args.output)
    header = list(pd.read_csv(source, nrows=0).columns)
    missing = [column for column in REQUIRED if column not in header]
    if missing:
        raise SystemExit(
            f"INPUT_BLOCKED: IBM AML CSV missing required columns: {missing}"
        )
    rows = positives = invalid_timestamps = invalid_amounts = 0
    min_time = max_time = None
    payment_formats: set[str] = set()
    for chunk in pd.read_csv(source, chunksize=args.chunksize):
        rows += len(chunk)
        target = chunk["Is Laundering"].astype(str).eq("1")
        positives += int(target.sum())
        timestamps = pd.to_datetime(chunk["Timestamp"], utc=True, errors="coerce")
        amounts = pd.to_numeric(chunk["Amount Paid"], errors="coerce")
        invalid_timestamps += int(timestamps.isna().sum())
        invalid_amounts += int(amounts.isna().sum())
        valid_time = timestamps.dropna()
        if not valid_time.empty:
            current_min, current_max = valid_time.min(), valid_time.max()
            min_time = (
                current_min if min_time is None or current_min < min_time else min_time
            )
            max_time = (
                current_max if max_time is None or current_max > max_time else max_time
            )
        payment_formats.update(
            chunk["Payment Format"].dropna().astype(str).unique().tolist()
        )
    profile = {
        "schema": "ibm_aml_profile.v1",
        "input": str(source.resolve()),
        "rows": rows,
        "positive_rows": positives,
        "positive_rate": (positives / rows) if rows else None,
        "timestamp_min": str(min_time) if min_time is not None else None,
        "timestamp_max": str(max_time) if max_time is not None else None,
        "invalid_timestamps": invalid_timestamps,
        "invalid_amounts": invalid_amounts,
        "payment_formats": sorted(payment_formats),
        "columns": header,
        "status": "PASS"
        if rows
        and positives
        and positives < rows
        and not invalid_timestamps
        and not invalid_amounts
        else "BLOCKED",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(profile, indent=2, sort_keys=True))
    if profile["status"] != "PASS":
        raise SystemExit(
            "INPUT_BLOCKED: profile did not meet AML experiment data contract"
        )


if __name__ == "__main__":
    main()
