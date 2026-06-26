from __future__ import annotations

import argparse
import json

from payment_ops_hardening import __version__

from payment_ops_hardening.release_contract import (
    ReleaseContractError,
    load_release_contract,
)
from payment_ops_hardening.release_security import (
    ReleaseSecurityError,
    verify_release_security,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="verify a hardened PaymentOps release")
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument("artifact_dir")
    parser.add_argument("--require-signature", action="store_true")
    parser.add_argument("--contract-file")
    parser.add_argument("--expected-key-id")
    parser.add_argument("--require-promote", action="store_true")
    parser.add_argument("--lock-timeout", type=float, default=0.0)
    args = parser.parse_args()

    contract = {}
    if args.contract_file:
        try:
            contract = load_release_contract(args.contract_file)
        except ReleaseContractError as exc:
            parser.error(str(exc))
    contract_key_id = contract.get("expected_key_id")
    if (
        args.expected_key_id is not None
        and contract_key_id is not None
        and args.expected_key_id != contract_key_id
    ):
        parser.error(
            "--expected-key-id conflicts with the trusted contract; "
            f"contract={contract_key_id!r}, command={args.expected_key_id!r}"
        )
    expected_key_id = contract_key_id or args.expected_key_id
    allowed_states = contract.get("allowed_release_states")
    if args.require_promote:
        allowed_states = ["PROMOTE"]
    contract_requires_signature = bool(contract.get("require_signature", False))
    try:
        result = verify_release_security(
            args.artifact_dir,
            require_signature=bool(
                args.require_signature or contract_requires_signature
            ),
            expected_key_id=expected_key_id,
            minimum_release_sequence=contract.get("minimum_release_sequence"),
            expected_paths=contract.get("required_paths"),
            expected_contracts=contract.get("content_contracts"),
            allowed_release_states=allowed_states,
            reject_untracked_files=bool(contract.get("reject_untracked_files", False)),
            allowed_untracked_paths=contract.get("allowed_untracked_paths", []),
            json_value_equalities=contract.get("json_value_equalities", []),
            lock_timeout=args.lock_timeout,
        )
    except ReleaseSecurityError as exc:
        parser.exit(2, f"release verification failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
