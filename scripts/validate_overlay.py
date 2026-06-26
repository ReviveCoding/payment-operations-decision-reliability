from __future__ import annotations

import json
import tempfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from payment_ops_hardening.release_contract import load_release_contract  # noqa: E402
from payment_ops_hardening.release_security import (  # noqa: E402
    ReleaseSecurityError,
    finalize_release_security,
    verify_release_security,
)

KEY = "validation-key-material-32-bytes!"
KEY_ID = "paymentops-2026-q2"
CONTRACT_PATH = ROOT / "contracts/synthetic_validation_contract.json"


def _write_release(root: Path, paths: list[str]) -> None:
    for relative in paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative.endswith(".csv"):
            path.write_text("case_id,value\nc1,1\nc2,2\n", encoding="utf-8")
        elif relative.endswith(".json"):
            path.write_text(
                json.dumps({"status": "PASS", "path": relative}),
                encoding="utf-8",
            )
        else:
            path.write_bytes(f"binary:{relative}".encode())
    (root / "release_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.1",
                "run_id": "small-release-001",
                "release_state": "PROMOTE",
            }
        ),
        encoding="utf-8",
    )


def _attempt(label: str, function) -> tuple[str, str]:
    try:
        function()
    except ReleaseSecurityError as exc:
        return label, f"BLOCKED: {exc}"
    return label, "UNEXPECTED_PASS"


def main() -> None:
    contract = load_release_contract(CONTRACT_PATH)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        paths = contract["required_paths"]
        _write_release(root, paths)
        finalized = finalize_release_security(
            root,
            required_paths=paths,
            content_contracts=contract["content_contracts"],
            key=KEY,
            key_id=KEY_ID,
            require_signature=True,
            release_sequence=100,
            allowed_release_states=contract["allowed_release_states"],
            reject_untracked_files=contract["reject_untracked_files"],
            allowed_untracked_paths=contract["allowed_untracked_paths"],
            json_value_equalities=contract.get("json_value_equalities", []),
        )
        verified = verify_release_security(
            root,
            key=KEY,
            require_signature=True,
            expected_key_id=contract["expected_key_id"],
            minimum_release_sequence=contract["minimum_release_sequence"],
            expected_paths=paths,
            expected_contracts=contract["content_contracts"],
            allowed_release_states=contract["allowed_release_states"],
            reject_untracked_files=contract["reject_untracked_files"],
            allowed_untracked_paths=contract["allowed_untracked_paths"],
            json_value_equalities=contract.get("json_value_equalities", []),
        )

        attacks: dict[str, str] = {}
        model_path = root / paths[0]
        original_model = model_path.read_bytes()
        label, result = _attempt(
            "artifact_tamper",
            lambda: (
                model_path.write_bytes(original_model + b"tamper"),
                verify_release_security(root, key=KEY),
            ),
        )
        attacks[label] = result
        model_path.write_bytes(original_model)

        extra = root / "unexpected.bin"
        label, result = _attempt(
            "untracked_file",
            lambda: (
                extra.write_bytes(b"unexpected"),
                verify_release_security(
                    root,
                    key=KEY,
                    expected_paths=paths,
                    reject_untracked_files=True,
                ),
            ),
        )
        attacks[label] = result
        extra.unlink(missing_ok=True)

        manifest_path = root / "release_manifest.json"
        original_manifest = manifest_path.read_bytes()
        manifest = json.loads(original_manifest)
        manifest["release_state"] = "HOLD"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        label, result = _attempt(
            "manifest_tamper",
            lambda: verify_release_security(root, key=KEY, require_signature=True),
        )
        attacks[label] = result
        manifest_path.write_bytes(original_manifest)

        label, result = _attempt(
            "wrong_key_id",
            lambda: verify_release_security(
                root,
                key=KEY,
                expected_key_id="paymentops-2026-q3",
            ),
        )
        attacks[label] = result

        finalized_report = dict(finalized)
        finalized_report["inventory_file"] = Path(
            finalized_report["inventory_file"]
        ).name
        output = {
            "overlay_version": "0.8.2",
            "contract": str(CONTRACT_PATH.relative_to(ROOT)),
            "finalized": finalized_report,
            "verified": verified,
            "attack_results": attacks,
        }
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
