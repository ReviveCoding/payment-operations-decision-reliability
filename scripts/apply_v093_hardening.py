#!/usr/bin/env python
"""Apply source-level PaymentOps AML v0.9.3 hardening changes.

The patch is intentionally fail-closed: it modifies only recognized v0.9.2
source shapes and raises an error if the repository differs materially.

v0.9.3.1 accepts either historical ordering of bootstrap_repeats and
bootstrap_sample_size fields.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

MARKER = "# PAYMENTOPS_V093_HARDENING"


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return source.replace(old, new, 1)


def replace_regex_once(source: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, source, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one regex match, found {count}")
    return updated


def patch_modeling(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    if MARKER in source:
        print(f"Already patched: {path}")
        return

    if "import json" not in source:
        if "from pathlib import Path\n" not in source:
            raise RuntimeError("Cannot find pathlib import insertion point")
        source = source.replace(
            "from pathlib import Path\n", "from pathlib import Path\nimport json\n", 1
        )

    # v0.9.2 field ordering varies across prior local overlays. Insert the new
    # fields immediately after the declared bootstrap_sample_size field rather
    # than assuming it appears before bootstrap_repeats.
    if 'bootstrap_method: Literal["iid_row", "time_block"]' not in source:
        source, count = re.subn(
            r"^(    bootstrap_sample_size: [^\n]*\n)",
            r"\1"
            '    bootstrap_method: Literal["iid_row", "time_block"] = "iid_row"\n'
            "    bootstrap_block_seconds: int = 21600\n",
            source,
            count=1,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise RuntimeError(
                "ExperimentConfig bootstrap fields: could not locate bootstrap_sample_size declaration"
            )
    source = replace_once(
        source,
        '    candidate_backend: Literal["catboost", "extra_trees"] = "catboost"\n',
        '    candidate_backend: Literal["catboost", "extra_trees"] = "catboost"\n'
        '    gpu_devices: str | None = "0"\n'
        "    exclude_bank_identity_categoricals: bool = False\n",
        "ExperimentConfig candidate fields",
    )
    source = replace_once(
        source,
        "        if self.bootstrap_repeats < 100:\n",
        '        if self.bootstrap_method not in {"iid_row", "time_block"}:\n'
        "            raise ModelingError(\"bootstrap_method must be 'iid_row' or 'time_block'\")\n"
        "        if self.bootstrap_block_seconds < 300:\n"
        '            raise ModelingError("bootstrap_block_seconds must be at least 300")\n'
        "        if self.bootstrap_repeats < 100:\n",
        "ExperimentConfig bootstrap validation",
    )

    source = replace_once(
        source,
        "def _feature_columns(config: ExperimentConfig) -> tuple[list[str], list[str]]:\n"
        '    categorical = list(config.categorical_columns) + ["bank_pair"]\n',
        "def _feature_columns(config: ExperimentConfig) -> tuple[list[str], list[str]]:\n"
        '    categorical = list(config.categorical_columns) + ["bank_pair"]\n'
        "    if config.exclude_bank_identity_categoricals:\n"
        "        bank_identity_columns = {\n"
        "            config.corridor_from_column,\n"
        "            config.corridor_to_column,\n"
        '            "bank_pair",\n'
        "        }\n"
        "        categorical = [\n"
        "            column for column in categorical if column not in bank_identity_columns\n"
        "        ]\n",
        "bank identity ablation feature contract",
    )

    split_replacement = """def split_chronologically(frame: Any, config: ExperimentConfig) -> tuple[Any, Any, Any]:
    # Split by whole timestamp groups so a timestamp never crosses partitions.

    total = len(frame)
    requested_train_end = int(total * config.train_fraction)
    requested_validation_end = int(
        total * (config.train_fraction + config.validation_fraction)
    )
    if total < 200:
        raise ModelingError("dataset is too small for chronological train/validation/test splits")

    def group_end(target_position: int) -> int:
        target_position = max(1, min(target_position, total - 1))
        boundary_time = frame["event_time"].iat[target_position - 1]
        return int(frame["event_time"].searchsorted(boundary_time, side="right"))

    train_end = group_end(requested_train_end)
    validation_target = max(requested_validation_end, train_end + 1)
    validation_end = group_end(validation_target)
    if validation_end >= total:
        raise ModelingError("timestamp-group split leaves no locked test partition")
    if train_end < 100 or validation_end - train_end < 50 or total - validation_end < 50:
        raise ModelingError(
            "timestamp-group chronological split is too small for train/validation/test"
        )

    train = frame.iloc[:train_end].copy()
    validation = frame.iloc[train_end:validation_end].copy()
    test = frame.iloc[validation_end:].copy()
    if not (train["event_time"].max() < validation["event_time"].min() < test["event_time"].min()):
        raise ModelingError("timestamp groups leaked across chronological partitions")
    for name, partition in (("train", train), ("validation", validation), ("test", test)):
        if partition["target"].nunique() != 2:
            raise ModelingError(f"{name} split does not contain both classes")
    return train, validation, test


def split_validation_for_tuning_and_calibration"""
    source = replace_regex_once(
        source,
        r"def split_chronologically\(frame: Any, config: ExperimentConfig\) -> tuple\[Any, Any, Any\]:.*?\n\n\ndef split_validation_for_tuning_and_calibration",
        split_replacement,
        "strict timestamp-group split",
    )

    source = replace_once(
        source,
        '                "task_type": task_type,\n            }\n',
        '                "task_type": task_type,\n            }\n'
        "            if config.use_gpu and config.gpu_devices:\n"
        '                parameters["devices"] = config.gpu_devices\n',
        "explicit CatBoost GPU device",
    )

    bootstrap_replacement = """def _bootstrap_capture_delta(
    labels: Any,
    baseline_probabilities: Any,
    candidate_probabilities: Any,
    review_fraction: float,
    repeats: int,
    seed: int,
    sample_size: int | None = None,
    *,
    event_times: Any | None = None,
    method: Literal["iid_row", "time_block"] = "iid_row",
    block_seconds: int = 21600,
) -> tuple[float, float]:
    # Paired capture uplift interval using IID rows or temporal blocks.
    # Temporal blocks are sampled intact with replacement.

    (_, np, *_rest) = _require_modeling_dependencies()
    labels = np.asarray(labels, dtype=int)
    baseline_probabilities = np.asarray(baseline_probabilities, dtype=float)
    candidate_probabilities = np.asarray(candidate_probabilities, dtype=float)
    if not (len(labels) == len(baseline_probabilities) == len(candidate_probabilities)):
        raise ModelingError("bootstrap arrays must have identical length")
    generator = np.random.default_rng(seed)
    draw_size = min(len(labels), sample_size) if sample_size is not None else len(labels)

    block_rows: list[Any] | None = None
    block_draws = 0
    if method == "time_block":
        if event_times is None:
            raise ModelingError("time_block bootstrap requires event_times")
        time_values = np.asarray(event_times, dtype="datetime64[ns]")
        if len(time_values) != len(labels):
            raise ModelingError("event_times length must match bootstrap labels")
        nanoseconds = time_values.astype("int64")
        if (nanoseconds == np.iinfo("int64").min).any():
            raise ModelingError("event_times contains invalid datetime values")
        block_ids = nanoseconds // int(block_seconds * 1_000_000_000)
        _, inverse = np.unique(block_ids, return_inverse=True)
        block_rows = [np.flatnonzero(inverse == index) for index in range(int(inverse.max()) + 1)]
        mean_block_size = float(np.mean([len(indices) for indices in block_rows]))
        block_draws = max(1, int(np.ceil(draw_size / mean_block_size)))

    deltas: list[float] = []
    for _ in range(repeats):
        if method == "iid_row":
            indices = generator.integers(0, len(labels), size=draw_size)
        else:
            assert block_rows is not None
            selected_blocks = generator.integers(0, len(block_rows), size=block_draws)
            indices = np.concatenate([block_rows[int(index)] for index in selected_blocks])
        sample_labels = labels[indices]
        if sample_labels.sum() == 0:
            continue
        baseline = _metrics(sample_labels, baseline_probabilities[indices], review_fraction)
        candidate = _metrics(sample_labels, candidate_probabilities[indices], review_fraction)
        deltas.append(candidate.capture_at_budget - baseline.capture_at_budget)
    if len(deltas) < max(50, repeats // 2):
        raise ModelingError("bootstrap produced insufficient positive resamples")
    return (float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975)))


def _promotion_decision"""
    source = replace_regex_once(
        source,
        r"def _bootstrap_capture_delta\(.*?\n\n\ndef _promotion_decision",
        bootstrap_replacement,
        "temporal-block paired bootstrap",
    )

    source = replace_once(
        source,
        "        config.random_seed,\n        config.bootstrap_sample_size,\n    )\n",
        "        config.random_seed,\n"
        "        config.bootstrap_sample_size,\n"
        '        event_times=test["event_time"].to_numpy(dtype="datetime64[ns]"),\n'
        "        method=config.bootstrap_method,\n"
        "        block_seconds=config.bootstrap_block_seconds,\n"
        "    )\n",
        "bootstrap call metadata",
    )

    persistence_old = """    joblib.dump(baseline, model_dir / "baseline_logistic.joblib")
    joblib.dump(baseline_calibrator, model_dir / "baseline_sigmoid_calibrator.joblib")
    joblib.dump(candidate_calibrator, model_dir / "candidate_sigmoid_calibrator.joblib")
    if backend == "catboost":
        candidate.save_model(str(model_dir / "candidate_catboost.cbm"))
    else:
        joblib.dump(candidate, model_dir / "candidate_extra_trees.joblib")
"""
    persistence_new = """    joblib.dump(baseline, model_dir / "baseline_logistic.joblib")
    joblib.dump(baseline_calibrator, model_dir / "baseline_sigmoid_calibrator.joblib")
    joblib.dump(candidate_calibrator, model_dir / "candidate_sigmoid_calibrator.joblib")
    backend_family, _, selected_weight_mode = backend.partition(":")
    if backend_family == "catboost":
        candidate_artifact_path = model_dir / "candidate_catboost.cbm"
        candidate.save_model(str(candidate_artifact_path))
    else:
        candidate_artifact_path = model_dir / "candidate_extra_trees.joblib"
        joblib.dump(candidate, candidate_artifact_path)
    observed_parameters: dict[str, Any] = {}
    if backend_family == "catboost" and hasattr(candidate, "get_all_params"):
        observed_parameters = candidate.get_all_params()
    candidate_provenance = {
        "backend_family": backend_family,
        "selected_weight_mode": selected_weight_mode or None,
        "runtime_class": f"{type(candidate).__module__}.{type(candidate).__name__}",
        "artifact": str(candidate_artifact_path.relative_to(destination)).replace("\\\\", "/"),
        "task_type_requested": "GPU" if config.use_gpu else "CPU",
        "devices_requested": config.gpu_devices if config.use_gpu else None,
        "task_type_observed": observed_parameters.get("task_type"),
        "devices_observed": observed_parameters.get("devices"),
        "iterations_observed": observed_parameters.get("iterations"),
        "random_seed_observed": observed_parameters.get("random_seed"),
        "has_time_observed": observed_parameters.get("has_time"),
    }
    (report_dir / "candidate_model_metadata.json").write_text(
        json.dumps(candidate_provenance, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
"""
    source = replace_once(
        source, persistence_old, persistence_new, "candidate artifact provenance"
    )

    source = replace_once(
        source,
        '        "candidate_backend": backend,\n        "review_fraction": config.review_fraction,\n'
        '        "bootstrap": {\n            "repeats": config.bootstrap_repeats,\n            "sample_size_cap": config.bootstrap_sample_size,\n        },\n',
        '        "candidate_backend": backend,\n'
        '        "candidate_backend_family": backend_family,\n'
        '        "candidate_weight_mode": selected_weight_mode or None,\n'
        '        "candidate_provenance": candidate_provenance,\n'
        '        "review_fraction": config.review_fraction,\n'
        '        "bootstrap": {\n'
        '            "method": (\n'
        '                "paired_temporal_block_bootstrap"\n'
        '                if config.bootstrap_method == "time_block"\n'
        '                else "paired_iid_row_bootstrap"\n'
        "            ),\n"
        '            "unit": "time_block" if config.bootstrap_method == "time_block" else "row",\n'
        '            "block_seconds": (\n'
        "                config.bootstrap_block_seconds\n"
        '                if config.bootstrap_method == "time_block"\n'
        "                else None\n"
        "            ),\n"
        '            "repeats": config.bootstrap_repeats,\n'
        '            "sample_size_target": config.bootstrap_sample_size,\n'
        "        },\n",
        "provenance and bootstrap report",
    )

    source += "\n\n" + MARKER + "\n"
    path.write_text(source, encoding="utf-8")
    print(f"Patched: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()
    repo = Path(args.repo_root).expanduser().resolve()
    path = repo / "src" / "payment_ops_hardening" / "modeling.py"
    if not path.exists():
        raise FileNotFoundError(path)
    patch_modeling(path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"v0.9.3 hardening patch failed: {exc}", file=sys.stderr)
        raise
