from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from payment_ops_hardening.modeling import (  # noqa: E402
    ModelingError,
    load_config,
    run_champion_challenger,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="run leakage-aware PaymentOps baseline-vs-candidate experiment"
    )
    parser.add_argument("--input", required=True, help="input payment-operation CSV")
    parser.add_argument(
        "--output-dir",
        default="artifacts/champion_challenger",
        help="directory for models, predictions, and metrics",
    )
    parser.add_argument(
        "--config",
        default="contracts/payment_risk_experiment_config.json",
        help="JSON experiment configuration",
    )
    args = parser.parse_args()
    try:
        config = load_config(ROOT / args.config)
        result = run_champion_challenger(
            ROOT / args.input,
            ROOT / args.output_dir,
            config,
        )
    except ModelingError as exc:
        raise SystemExit(f"EXPERIMENT_BLOCKED: {exc}") from exc
    print(json.dumps(result["promotion_decision"], indent=2, sort_keys=True))
    print(f"Evidence: {ROOT / args.output_dir / 'reports' / 'experiment_metrics.json'}")


if __name__ == "__main__":
    main()
