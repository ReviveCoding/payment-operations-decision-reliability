"""Generate a deterministic, non-production payment-risk dataset for smoke tests.

The synthetic target intentionally includes nonlinear interactions so the candidate
model can be checked end-to-end.  It is only a harness test and must never be used
for portfolio performance claims.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/synthetic_payment_risk.csv")
    parser.add_argument("--rows", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=20260620)
    args = parser.parse_args()
    if args.rows < 1000:
        raise SystemExit("--rows must be at least 1000")
    generator = np.random.default_rng(args.seed)
    timestamp = pd.date_range("2025-01-01", periods=args.rows, freq="2min", tz="UTC")
    debtor = generator.choice(
        ["BANKGB01", "BANKGB02", "BANKDE01", "BANKFR01"], args.rows
    )
    creditor = generator.choice(
        ["BANKUS01", "BANKDE02", "BANKNL01", "BANKGB03"], args.rows
    )
    purpose = generator.choice(["SALA", "GDSV", "SUPP", "TREA", "TAXS"], args.rows)
    currency = generator.choice(["USD", "EUR", "GBP"], args.rows)
    amount = np.exp(generator.normal(7.1, 0.9, args.rows)).clip(20, 25000)
    debtor_country = np.select(
        [
            debtor == "BANKGB01",
            debtor == "BANKGB02",
            debtor == "BANKDE01",
            debtor == "BANKFR01",
        ],
        ["GB", "GB", "DE", "FR"],
        default="GB",
    )
    creditor_country = np.select(
        [
            creditor == "BANKUS01",
            creditor == "BANKDE02",
            creditor == "BANKNL01",
            creditor == "BANKGB03",
        ],
        ["US", "DE", "NL", "GB"],
        default="GB",
    )
    hour = timestamp.hour.to_numpy()
    night = ((hour < 7) | (hour > 20)).astype(int)
    # Interactions are deliberately nonlinear: neither individual bank identity
    # has a stable marginal effect, but the pair/purpose combinations do.
    xor_pair = ((debtor == "BANKGB01") ^ (creditor == "BANKUS01")).astype(int)
    xor_risk = (xor_pair & (purpose == "TREA")).astype(int)
    amount_purpose_risk = ((amount > 2500) & (purpose == "SALA")).astype(int)
    night_treasury_risk = (night & (purpose == "TAXS")).astype(int)
    logits = (
        -4.8 + 4.1 * xor_risk + 3.0 * amount_purpose_risk + 2.0 * night_treasury_risk
    )
    probability = 1.0 / (1.0 + np.exp(-logits))
    rejected = generator.binomial(1, probability)
    frame = pd.DataFrame(
        {
            "MessageId": [f"SYN-{index:08d}" for index in range(args.rows)],
            "TxDateTime": timestamp.astype(str),
            "InstdAmt": amount.round(2),
            "Currency": currency,
            "DbtrAgtBIC": debtor,
            "CdtrAgtBIC": creditor,
            "DbtrCountry": debtor_country,
            "CdtrCountry": creditor_country,
            "PurposeCode": purpose,
            "TxSts": np.where(rejected == 1, "RJCT", "ACSP"),
        }
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(
        f"wrote {len(frame)} rows to {destination}; positive_rate={frame['TxSts'].eq('RJCT').mean():.4f}"
    )


if __name__ == "__main__":
    main()
