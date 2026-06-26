from __future__ import annotations

import argparse
import json
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path

EXPECTED_VERSION = "0.9.4"
REQUIRED_DISTRIBUTION_DEPENDENCIES = (
    "pandas",
    "scikit-learn",
    "joblib",
    "catboost",
)
REQUIRED_SDIST_FILES = {
    "README.md",
    "pyproject.toml",
    "MANIFEST.in",
    "requirements-modeling.txt",
    "contracts/payment_risk_experiment_ibm_aml_medium_confirmatory_no_bank_identity_v094.json",
    "scripts/run_champion_challenger_v094.py",
    "scripts/run_frozen_hi_to_li_v094.py",
    "scripts/run_ibm_aml_v094.ps1",
    "scripts/verify_distribution.py",
    "src/payment_ops_hardening/modeling.py",
    "tests/test_v094_confirmatory.py",
}
EXPECTED_DOCUMENTS = {
    "docs/LEGACY_OVERLAY_CONTEXT.md",
    "docs/MODEL_CARD_V094.md",
    "docs/V094_EVIDENCE.md",
    "docs/known_limitations.md",
    "docs/operations.md",
    "docs/release_qualification.md",
    "docs/v094_confirmatory_protocol.md",
}
FORBIDDEN_ROOT_FILES = {
    "VALIDATION_REPORT.md",
    "README_UPLIFT_EXTENSION.md",
    "README_V092.md",
    "README_v093_hardening.md",
    "qualification_manifest.json",
    "release_bundle_manifest.json",
    "release_candidate_handoff.json",
}
FORBIDDEN_PREFIXES = (
    ".local-run/",
    ".venv",
    "build/",
    "dist/",
    "reports/",
)


def _distribution_members(sdist_path: Path) -> tuple[str, set[str]]:
    with tarfile.open(sdist_path, "r:gz") as archive:
        names = [member.name for member in archive.getmembers() if member.isfile()]

    roots = {name.split("/", 1)[0] for name in names if "/" in name}
    expected_root = f"payment_ops_hardening_overlay-{EXPECTED_VERSION}"

    if roots != {expected_root}:
        raise ValueError(
            f"sdist root mismatch: expected={expected_root!r}, observed={sorted(roots)!r}"
        )

    inner_names = {
        name.split("/", 1)[1] for name in names if name.startswith(f"{expected_root}/")
    }
    return expected_root, inner_names


def verify_wheel(wheel_path: Path) -> dict[str, object]:
    with zipfile.ZipFile(wheel_path) as archive:
        metadata_paths = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_paths) != 1:
            raise ValueError(
                "wheel must contain exactly one .dist-info/METADATA file; "
                f"observed={metadata_paths!r}"
            )

        metadata = Parser().parsestr(archive.read(metadata_paths[0]).decode("utf-8"))
        files = set(archive.namelist())

    observed_version = metadata.get("Version")
    if observed_version != EXPECTED_VERSION:
        raise ValueError(
            f"wheel version mismatch: expected={EXPECTED_VERSION}, observed={observed_version}"
        )

    requires_dist = metadata.get_all("Requires-Dist") or []
    missing_dependencies = [
        dependency
        for dependency in REQUIRED_DISTRIBUTION_DEPENDENCIES
        if not any(
            requirement.lower().startswith(dependency) for requirement in requires_dist
        )
    ]
    if missing_dependencies:
        raise ValueError(
            "wheel metadata missing required runtime dependencies: "
            f"{missing_dependencies}"
        )

    required_module = "payment_ops_hardening/modeling.py"
    if required_module not in files:
        raise ValueError(f"wheel missing required module: {required_module}")

    return {
        "artifact": wheel_path.name,
        "version": observed_version,
        "metadata_path": metadata_paths[0],
        "required_dependencies": list(REQUIRED_DISTRIBUTION_DEPENDENCIES),
    }


def verify_sdist(sdist_path: Path) -> dict[str, object]:
    root, files = _distribution_members(sdist_path)

    missing_files = sorted(REQUIRED_SDIST_FILES - files)
    if missing_files:
        raise ValueError(f"sdist missing required files: {missing_files}")

    actual_docs = {
        name for name in files if name.startswith("docs/") and name.endswith(".md")
    }
    missing_docs = sorted(EXPECTED_DOCUMENTS - actual_docs)
    unexpected_docs = sorted(actual_docs - EXPECTED_DOCUMENTS)

    if missing_docs:
        raise ValueError(f"sdist missing required v0.9.4 documentation: {missing_docs}")
    if unexpected_docs:
        raise ValueError(
            f"sdist contains unexpected legacy documentation: {unexpected_docs}"
        )

    leaked_root_files = sorted(FORBIDDEN_ROOT_FILES & files)
    if leaked_root_files:
        raise ValueError(
            f"sdist contains forbidden legacy root files: {leaked_root_files}"
        )

    leaked_paths = sorted(
        name
        for name in files
        if any(name.startswith(prefix) for prefix in FORBIDDEN_PREFIXES)
    )
    if leaked_paths:
        raise ValueError(
            f"sdist contains forbidden generated/local paths: {leaked_paths}"
        )

    return {
        "artifact": sdist_path.name,
        "root": root,
        "required_files": sorted(REQUIRED_SDIST_FILES),
        "documentation": sorted(actual_docs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate v0.9.4 PaymentOps wheel and source-distribution boundaries."
    )
    parser.add_argument("wheel", type=Path)
    parser.add_argument("sdist", type=Path)
    args = parser.parse_args()

    if not args.wheel.is_file():
        raise FileNotFoundError(f"wheel was not found: {args.wheel}")
    if not args.sdist.is_file():
        raise FileNotFoundError(f"source distribution was not found: {args.sdist}")

    result = {
        "status": "PASS",
        "expected_version": EXPECTED_VERSION,
        "wheel": verify_wheel(args.wheel),
        "sdist": verify_sdist(args.sdist),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
