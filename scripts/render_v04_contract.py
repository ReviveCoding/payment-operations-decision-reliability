from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from payment_ops_hardening.contract_template import render_release_contract_template  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="render the v0.4 PaymentOps release contract template"
    )
    parser.add_argument("--selected-model", required=True)
    parser.add_argument("--key-id", required=True)
    parser.add_argument("--minimum-release-sequence", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--template",
        type=Path,
        default=ROOT / "contracts/payment_ops_v04_release_contract.template.json",
    )
    args = parser.parse_args()
    rendered = render_release_contract_template(
        args.template,
        args.output,
        {
            "__SELECTED_MODEL__": args.selected_model,
            "__EXPECTED_KEY_ID__": args.key_id,
            "__MINIMUM_RELEASE_SEQUENCE__": args.minimum_release_sequence,
        },
    )
    print(json.dumps(rendered, indent=2))


if __name__ == "__main__":
    main()
