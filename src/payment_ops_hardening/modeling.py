"""Leakage-aware champion-challenger experiments for payment-risk ranking.

This module intentionally keeps the release-security package lightweight: pandas,
scikit-learn, joblib and CatBoost are optional dependencies installed through the
``modeling`` extra.  The experiment runner refuses known post-outcome columns and
only derives historical features from events strictly earlier than each row.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import json
import math


class ModelingError(RuntimeError):
    """Raised when experiment data, configuration, or evidence is invalid."""


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for a chronological payment-risk experiment."""

    timestamp_column: str = "TxDateTime"
    target_column: str = "TxSts"
    positive_values: tuple[str, ...] = ("RJCT",)
    amount_column: str = "InstdAmt"
    categorical_columns: tuple[str, ...] = (
        "Currency",
        "DbtrAgtBIC",
        "CdtrAgtBIC",
        "DbtrCountry",
        "CdtrCountry",
        "PurposeCode",
    )
    payer_column: str = "DbtrAgtBIC"
    payee_column: str = "CdtrAgtBIC"
    # Optional corridor columns separate routing context from account-level entities.
    # When omitted, payer/payee columns are used for both roles.
    corridor_from_column: str | None = None
    corridor_to_column: str | None = None
    # Optional chronological prefix for an inexpensive local smoke/probe run.
    max_rows: int | None = None
    # Caps per-replicate bootstrap size for multi-million-row held-out sets.
    bootstrap_sample_size: int | None = None
    bootstrap_method: Literal["iid_row", "time_block"] = "iid_row"
    bootstrap_block_seconds: int = 21600
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    review_fraction: float = 0.05
    bootstrap_repeats: int = 500
    random_seed: int = 20260620
    candidate_backend: Literal["catboost", "extra_trees"] = "catboost"
    gpu_devices: str | None = "0"
    exclude_bank_identity_categoricals: bool = False
    use_gpu: bool = False
    max_iterations: int = 600
    min_capture_delta: float = 0.0
    max_brier_regression: float = 0.002
    target_maturity_seconds: int = 0
    # Hold back part of validation for independent probability calibration.
    calibration_fraction: float = 0.50
    # Pre-registered CatBoost imbalance treatments selected only on the tune split.
    candidate_weight_modes: tuple[str, ...] = ("none",)

    @staticmethod
    def from_mapping(value: dict[str, Any]) -> "ExperimentConfig":
        known = set(ExperimentConfig.__dataclass_fields__)
        unknown = set(value) - known
        if unknown:
            raise ModelingError(
                f"unknown experiment configuration fields: {sorted(unknown)}"
            )
        normalized = dict(value)
        for name in (
            "positive_values",
            "categorical_columns",
            "candidate_weight_modes",
        ):
            if name in normalized:
                normalized[name] = tuple(normalized[name])
        config = ExperimentConfig(**normalized)
        config.validate()
        return config

    def validate(self) -> None:
        if not 0 < self.train_fraction < 1:
            raise ModelingError("train_fraction must be in (0, 1)")
        if not 0 < self.validation_fraction < 1:
            raise ModelingError("validation_fraction must be in (0, 1)")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ModelingError(
                "train_fraction + validation_fraction must be less than 1"
            )
        if not 0 < self.review_fraction <= 1:
            raise ModelingError("review_fraction must be in (0, 1]")
        if self.bootstrap_method not in {"iid_row", "time_block"}:
            raise ModelingError("bootstrap_method must be 'iid_row' or 'time_block'")
        if self.bootstrap_block_seconds < 300:
            raise ModelingError("bootstrap_block_seconds must be at least 300")
        if self.bootstrap_repeats < 100:
            raise ModelingError("bootstrap_repeats must be at least 100")
        if self.max_iterations < 50:
            raise ModelingError("max_iterations must be at least 50")
        if self.max_rows is not None and self.max_rows < 500:
            raise ModelingError("max_rows must be at least 500 when supplied")
        if self.bootstrap_sample_size is not None and self.bootstrap_sample_size < 500:
            raise ModelingError(
                "bootstrap_sample_size must be at least 500 when supplied"
            )
        if (self.corridor_from_column is None) != (self.corridor_to_column is None):
            raise ModelingError(
                "corridor_from_column and corridor_to_column must be supplied together"
            )
        if self.target_maturity_seconds < 0:
            raise ModelingError("target_maturity_seconds must be non-negative")
        if not 0 < self.calibration_fraction < 1:
            raise ModelingError("calibration_fraction must be in (0, 1)")
        allowed_weight_modes = {"none", "SqrtBalanced", "Balanced"}
        if not self.candidate_weight_modes:
            raise ModelingError("candidate_weight_modes cannot be empty")
        unknown_weight_modes = set(self.candidate_weight_modes) - allowed_weight_modes
        if unknown_weight_modes:
            raise ModelingError(
                f"unknown candidate_weight_modes: {sorted(unknown_weight_modes)}"
            )
        if not self.positive_values:
            raise ModelingError("positive_values cannot be empty")


@dataclass(frozen=True)
class ModelMetrics:
    average_precision: float
    roc_auc: float | None
    brier: float
    capture_at_budget: float
    precision_at_budget: float
    lift_at_budget: float
    positives_captured: int
    review_count: int
    positive_count: int


@dataclass(frozen=True)
class PromotionDecision:
    champion: Literal["baseline", "candidate"]
    reason: str
    capture_delta: float
    capture_delta_ci_low: float
    capture_delta_ci_high: float
    brier_delta: float
    primary_metric: str


def _require_modeling_dependencies() -> tuple[Any, ...]:
    try:
        import joblib
        import numpy as np
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            average_precision_score,
            brier_score_loss,
            roc_auc_score,
        )
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
    except ImportError as exc:  # pragma: no cover - depends on installation extras
        raise ModelingError(
            "modeling dependencies are missing. Install with: "
            "python -m pip install '.[modeling]'"
        ) from exc
    return (
        joblib,
        np,
        pd,
        ColumnTransformer,
        HistGradientBoostingClassifier,
        SimpleImputer,
        LogisticRegression,
        average_precision_score,
        brier_score_loss,
        roc_auc_score,
        Pipeline,
        OneHotEncoder,
        OrdinalEncoder,
        StandardScaler,
    )


def load_config(path: str | Path | None) -> ExperimentConfig:
    if path is None:
        config = ExperimentConfig()
        config.validate()
        return config
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelingError(f"cannot read experiment config {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ModelingError("experiment config must be a JSON object")
    return ExperimentConfig.from_mapping(value)


def _assert_required_columns(frame: Any, config: ExperimentConfig) -> None:
    required = {
        config.timestamp_column,
        config.target_column,
        config.amount_column,
        config.payer_column,
        config.payee_column,
        *(
            (config.corridor_from_column, config.corridor_to_column)
            if config.corridor_from_column is not None
            else ()
        ),
        *config.categorical_columns,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ModelingError(f"input dataset is missing required columns: {missing}")


def _online_window_count(
    times: Iterable[Any], keys: Iterable[str], window_seconds: int
) -> list[int]:
    queues: dict[str, deque[Any]] = defaultdict(deque)
    output: list[int] = []
    for timestamp, key in zip(times, keys, strict=True):
        queue = queues[key]
        cutoff = timestamp.value // 10**9 - window_seconds
        while queue and queue[0] < cutoff:
            queue.popleft()
        output.append(len(queue))
        queue.append(timestamp.value // 10**9)
    return output


def _online_prior_mean(values: Iterable[float], keys: Iterable[str]) -> list[float]:
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    output: list[float] = []
    global_sum = 0.0
    global_count = 0
    for value, key in zip(values, keys, strict=True):
        fallback = global_sum / global_count if global_count else value
        output.append(sums[key] / counts[key] if counts[key] else fallback)
        sums[key] += value
        counts[key] += 1
        global_sum += value
        global_count += 1
    return output


def _normalized_string_column(result: Any, column: str) -> Any:
    """Return a compact, non-null string column once per feature source."""

    value = result[column].astype("string").fillna("__MISSING__")
    return value


def _entity_key(left: Any, right: Any, separator: str = "::") -> Any:
    """Create a stable entity key without repeatedly coercing raw columns."""

    return left.str.cat(right, sep=separator)


def prepare_leakage_safe_frame(frame: Any, config: ExperimentConfig) -> Any:
    """Create point-in-time features without post-outcome information.

    Historical aggregations are calculated strictly in chronological order.  In the
    IBM AML schema, source and destination keys include both bank and account so
    account IDs from different institutions cannot collide.
    """

    (_, np, pd, *_rest) = _require_modeling_dependencies()
    _assert_required_columns(frame, config)
    protected = {
        config.target_column,
        "RejectionReason",
        "StatusTimestamp",
        "ProcessingTimeSecs",
    }
    result = frame.copy()
    result["event_time"] = pd.to_datetime(
        result[config.timestamp_column], utc=True, errors="coerce"
    )
    if result["event_time"].isna().any():
        raise ModelingError("timestamp column contains invalid values")
    result["target"] = (
        result[config.target_column]
        .astype(str)
        .isin(config.positive_values)
        .astype(int)
    )
    if result["target"].nunique() != 2:
        raise ModelingError("target must contain both positive and negative records")
    result[config.amount_column] = pd.to_numeric(
        result[config.amount_column], errors="coerce"
    )
    if result[config.amount_column].isna().any():
        raise ModelingError("amount column contains non-numeric or missing values")
    if (result[config.amount_column] < 0).any():
        raise ModelingError("amount column cannot contain negative values")
    result = result.sort_values(["event_time"], kind="mergesort").reset_index(drop=True)

    # Normalize string columns once; repeated astype(str) calls are expensive at the
    # 6M+ row IBM AML scale and were the source of avoidable local latency.
    normalized = {
        column: _normalized_string_column(result, column)
        for column in set(
            list(config.categorical_columns)
            + [
                config.payer_column,
                config.payee_column,
                config.corridor_from_column or config.payer_column,
                config.corridor_to_column or config.payee_column,
            ]
        )
    }
    for column in config.categorical_columns:
        result[column] = normalized[column]

    corridor_from = config.corridor_from_column or config.payer_column
    corridor_to = config.corridor_to_column or config.payee_column
    source_bank = normalized[corridor_from]
    destination_bank = normalized[corridor_to]
    payer_value = normalized[config.payer_column]
    payee_value = normalized[config.payee_column]

    result["is_cross_border"] = (source_bank != destination_bank).astype("int8")
    result["bank_pair"] = _entity_key(source_bank, destination_bank, "__")
    result["source_entity"] = _entity_key(source_bank, payer_value)
    result["destination_entity"] = _entity_key(destination_bank, payee_value)
    result["account_pair"] = _entity_key(
        result["source_entity"], result["destination_entity"], "__"
    )

    result["amount_log1p"] = np.log1p(result[config.amount_column].astype(float))
    result["hour_of_day"] = result["event_time"].dt.hour.astype("int8")
    result["day_of_week"] = result["event_time"].dt.dayofweek.astype("int8")
    result["is_weekend"] = (result["day_of_week"] >= 5).astype("int8")
    result["minute_bucket"] = (result["event_time"].dt.minute // 15).astype("int8")

    # When an AML schema exposes both paid and received amount, use the contemporaneous
    # transaction conversion ratio. This is not an outcome field and is available at
    # the decision event; the ratio is robustly clipped after numeric coercion.
    if "Amount Received" in result.columns:
        received = pd.to_numeric(result["Amount Received"], errors="coerce")
        if received.notna().all() and (received >= 0).all():
            result["received_amount_log1p"] = np.log1p(received.astype(float))
            denominator = received.replace(0.0, np.nan)
            result["paid_to_received_ratio"] = (
                result[config.amount_column].astype(float) / denominator
            ).replace([np.inf, -np.inf], np.nan)

    effective_time = result["event_time"] - pd.to_timedelta(
        config.target_maturity_seconds, unit="s"
    )
    amounts = result[config.amount_column].astype(float)

    # Distinct source and destination histories are important for AML flows. The
    # account-pair count captures repeated source-to-destination relationships.
    history_specs = {
        "payer": result["source_entity"],
        "payee": result["destination_entity"],
        "account_pair": result["account_pair"],
        "bank_pair": result["bank_pair"],
    }
    window_map = {
        "payer": (("1h", 3600), ("24h", 86400), ("7d", 604800)),
        "payee": (("1h", 3600), ("24h", 86400), ("7d", 604800)),
        "account_pair": (("24h", 86400), ("7d", 604800)),
        "bank_pair": (("24h", 86400),),
    }
    for key_name, key_values in history_specs.items():
        for label, seconds in window_map[key_name]:
            result[f"{key_name}_prior_count_{label}"] = _online_window_count(
                effective_time, key_values, seconds
            )
        prior_mean = _online_prior_mean(amounts, key_values)
        result[f"{key_name}_prior_mean_amount"] = prior_mean
        denominator = result[f"{key_name}_prior_mean_amount"].replace(0.0, np.nan)
        result[f"{key_name}_amount_ratio"] = (amounts / denominator).replace(
            [np.inf, -np.inf], np.nan
        )
        result[f"{key_name}_is_new"] = (
            result.get(f"{key_name}_prior_count_24h", 0) == 0
        ).astype("int8")

    # Backward-compatible aliases retain the original generic pair feature names.
    for label in ("1h", "24h", "7d"):
        source = f"bank_pair_prior_count_{label}"
        if source in result:
            result[f"pair_prior_count_{label}"] = result[source]
    result["pair_prior_mean_amount"] = result["bank_pair_prior_mean_amount"]
    result["pair_amount_ratio"] = result["bank_pair_amount_ratio"]

    result.attrs["protected_columns"] = sorted(protected)
    return result


def split_chronologically(frame: Any, config: ExperimentConfig) -> tuple[Any, Any, Any]:
    # Split by whole timestamp groups so a timestamp never crosses partitions.

    total = len(frame)
    requested_train_end = int(total * config.train_fraction)
    requested_validation_end = int(
        total * (config.train_fraction + config.validation_fraction)
    )
    if total < 200:
        raise ModelingError(
            "dataset is too small for chronological train/validation/test splits"
        )

    def group_end(target_position: int) -> int:
        target_position = max(1, min(target_position, total - 1))
        boundary_time = frame["event_time"].iat[target_position - 1]
        return int(frame["event_time"].searchsorted(boundary_time, side="right"))

    train_end = group_end(requested_train_end)
    validation_target = max(requested_validation_end, train_end + 1)
    validation_end = group_end(validation_target)
    if validation_end >= total:
        raise ModelingError("timestamp-group split leaves no locked test partition")
    if (
        train_end < 100
        or validation_end - train_end < 50
        or total - validation_end < 50
    ):
        raise ModelingError(
            "timestamp-group chronological split is too small for train/validation/test"
        )

    train = frame.iloc[:train_end].copy()
    validation = frame.iloc[train_end:validation_end].copy()
    test = frame.iloc[validation_end:].copy()
    if not (
        train["event_time"].max()
        < validation["event_time"].min()
        < test["event_time"].min()
    ):
        raise ModelingError("timestamp groups leaked across chronological partitions")
    for name, partition in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        if partition["target"].nunique() != 2:
            raise ModelingError(f"{name} split does not contain both classes")
    return train, validation, test


def split_validation_for_tuning_and_calibration(
    validation: Any, config: ExperimentConfig
) -> tuple[Any, Any]:
    """Split validation chronologically: tune/early-stop first, calibrate second."""

    cut = int(len(validation) * (1 - config.calibration_fraction))
    tuning = validation.iloc[:cut].copy()
    calibration = validation.iloc[cut:].copy()
    for name, partition in (("tuning", tuning), ("calibration", calibration)):
        if len(partition) < 50 or partition["target"].nunique() != 2:
            raise ModelingError(
                f"{name} validation partition must contain both classes and at least 50 rows"
            )
    return tuning, calibration


def _feature_columns(config: ExperimentConfig) -> tuple[list[str], list[str]]:
    categorical = list(config.categorical_columns) + ["bank_pair"]
    if config.exclude_bank_identity_categoricals:
        bank_identity_columns = {
            config.corridor_from_column,
            config.corridor_to_column,
            "bank_pair",
        }
        categorical = [
            column for column in categorical if column not in bank_identity_columns
        ]
    numeric = [
        "amount_log1p",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "minute_bucket",
        "is_cross_border",
        "payer_prior_count_1h",
        "payer_prior_count_24h",
        "payer_prior_count_7d",
        "payee_prior_count_1h",
        "payee_prior_count_24h",
        "payee_prior_count_7d",
        "account_pair_prior_count_24h",
        "account_pair_prior_count_7d",
        "bank_pair_prior_count_24h",
        "payer_amount_ratio",
        "payee_amount_ratio",
        "account_pair_amount_ratio",
        "bank_pair_amount_ratio",
        "payer_is_new",
        "payee_is_new",
        "account_pair_is_new",
        "bank_pair_is_new",
    ]
    if config.amount_column == "Amount Paid":
        numeric.extend(["received_amount_log1p", "paid_to_received_ratio"])
    return categorical, numeric


def _fit_baseline(train: Any, categorical: list[str], numeric: list[str]) -> Any:
    (
        _,
        _,
        _,
        ColumnTransformer,
        _,
        SimpleImputer,
        LogisticRegression,
        _,
        _,
        _,
        Pipeline,
        OneHotEncoder,
        _,
        StandardScaler,
    ) = _require_modeling_dependencies()
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    model = Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", LogisticRegression(C=0.3, max_iter=5000, random_state=20260620)),
        ]
    )
    model.fit(train[categorical + numeric], train["target"])
    return model


def _fit_candidate(
    train: Any,
    tuning: Any,
    categorical: list[str],
    numeric: list[str],
    config: ExperimentConfig,
) -> tuple[Any, str]:
    """Fit pre-registered challenger variants and select on tune Capture@K only."""

    dependencies = _require_modeling_dependencies()
    ColumnTransformer = dependencies[3]
    SimpleImputer = dependencies[5]
    Pipeline = dependencies[10]
    OneHotEncoder = dependencies[11]
    columns = categorical + numeric
    if config.candidate_backend == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ModelingError(
                "CatBoost backend requested but catboost is not installed. "
                "Install with: python -m pip install '.[modeling]'"
            ) from exc
        task_type = "GPU" if config.use_gpu else "CPU"
        candidates: list[tuple[float, float, Any, str]] = []
        for weight_mode in config.candidate_weight_modes:
            parameters: dict[str, Any] = {
                "iterations": config.max_iterations,
                "depth": 8,
                "learning_rate": 0.05,
                "loss_function": "Logloss",
                # PRAUC is computed offline; AUC keeps GPU early stopping entirely on GPU.
                "eval_metric": "AUC",
                "random_seed": config.random_seed,
                "l2_leaf_reg": 12.0,
                "random_strength": 0.25,
                "has_time": True,
                "allow_writing_files": False,
                "verbose": False,
                "thread_count": 4,
                "task_type": task_type,
            }
            if config.use_gpu and config.gpu_devices:
                parameters["devices"] = config.gpu_devices
            if weight_mode != "none":
                parameters["auto_class_weights"] = weight_mode
            try:
                model = CatBoostClassifier(**parameters)
                model.fit(
                    train[columns],
                    train["target"],
                    cat_features=categorical,
                    eval_set=(tuning[columns], tuning["target"]),
                    early_stopping_rounds=100,
                )
            except Exception as exc:
                if config.use_gpu:
                    raise ModelingError(f"CatBoost GPU training failed: {exc}") from exc
                raise ModelingError(f"CatBoost training failed: {exc}") from exc
            validation_probability = _predict_probability(model, tuning, columns)
            validation_metrics = _metrics(
                tuning["target"], validation_probability, config.review_fraction
            )
            candidates.append(
                (
                    validation_metrics.capture_at_budget,
                    validation_metrics.average_precision,
                    model,
                    weight_mode,
                )
            )
        # Deterministic tie-break: Capture@K, then AP, then configuration order.
        best = max(candidates, key=lambda item: (item[0], item[1]))
        return best[2], f"catboost:{best[3]}"

    from sklearn.ensemble import ExtraTreesClassifier

    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("impute", SimpleImputer(strategy="median"))]),
                numeric,
            ),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    model = Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=max(250, min(config.max_iterations, 600)),
                    max_features="sqrt",
                    min_samples_leaf=3,
                    class_weight="balanced_subsample",
                    n_jobs=4,
                    random_state=config.random_seed,
                ),
            ),
        ]
    )
    model.fit(train[columns], train["target"])
    return model, "extra_trees"


def _predict_probability(model: Any, frame: Any, columns: list[str]) -> Any:
    return model.predict_proba(frame[columns])[:, 1]


def _fit_sigmoid_calibrator(probabilities: Any, labels: Any) -> Any:
    (
        _,
        np,
        _,
        _,
        _,
        _,
        LogisticRegression,
        *_rest,
    ) = _require_modeling_dependencies()
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(C=100.0, max_iter=2000, random_state=20260620)
    calibrator.fit(logit, labels)
    return calibrator


def _apply_sigmoid_calibrator(calibrator: Any, probabilities: Any) -> Any:
    (_, np, *_rest) = _require_modeling_dependencies()
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return calibrator.predict_proba(logit)[:, 1]


def _metrics(labels: Any, probabilities: Any, review_fraction: float) -> ModelMetrics:
    (
        _,
        np,
        _,
        _,
        _,
        _,
        _,
        average_precision_score,
        brier_score_loss,
        roc_auc_score,
        *_rest,
    ) = _require_modeling_dependencies()
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    review_count = max(1, int(math.ceil(len(labels) * review_fraction)))
    ranked = np.argsort(-probabilities, kind="mergesort")[:review_count]
    positives = int(labels.sum())
    captured = int(labels[ranked].sum())
    precision = float(captured / review_count)
    base_rate = float(labels.mean())
    lift = float(precision / base_rate) if base_rate > 0 else 0.0
    return ModelMetrics(
        average_precision=float(average_precision_score(labels, probabilities)),
        roc_auc=float(roc_auc_score(labels, probabilities))
        if labels.min() != labels.max()
        else None,
        brier=float(brier_score_loss(labels, probabilities)),
        capture_at_budget=float(captured / positives) if positives else 0.0,
        precision_at_budget=precision,
        lift_at_budget=lift,
        positives_captured=captured,
        review_count=review_count,
        positive_count=positives,
    )


def _bootstrap_capture_delta(
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
    draw_size = (
        min(len(labels), sample_size) if sample_size is not None else len(labels)
    )

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
        block_rows = [
            np.flatnonzero(inverse == index) for index in range(int(inverse.max()) + 1)
        ]
        mean_block_size = float(np.mean([len(indices) for indices in block_rows]))
        block_draws = max(1, int(np.ceil(draw_size / mean_block_size)))

    deltas: list[float] = []
    for _ in range(repeats):
        if method == "iid_row":
            indices = generator.integers(0, len(labels), size=draw_size)
        else:
            assert block_rows is not None
            selected_blocks = generator.integers(0, len(block_rows), size=block_draws)
            indices = np.concatenate(
                [block_rows[int(index)] for index in selected_blocks]
            )
        sample_labels = labels[indices]
        if sample_labels.sum() == 0:
            continue
        baseline = _metrics(
            sample_labels, baseline_probabilities[indices], review_fraction
        )
        candidate = _metrics(
            sample_labels, candidate_probabilities[indices], review_fraction
        )
        deltas.append(candidate.capture_at_budget - baseline.capture_at_budget)
    if len(deltas) < max(50, repeats // 2):
        raise ModelingError("bootstrap produced insufficient positive resamples")
    return (float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975)))


def _promotion_decision(
    baseline: ModelMetrics,
    candidate: ModelMetrics,
    ci_low: float,
    ci_high: float,
    config: ExperimentConfig,
) -> PromotionDecision:
    capture_delta = candidate.capture_at_budget - baseline.capture_at_budget
    brier_delta = candidate.brier - baseline.brier
    if capture_delta < config.min_capture_delta:
        reason = (
            "candidate did not meet the pre-specified capture improvement threshold"
        )
        champion: Literal["baseline", "candidate"] = "baseline"
    elif ci_low <= 0:
        reason = "capture improvement is not statistically directional under paired bootstrap"
        champion = "baseline"
    elif brier_delta > config.max_brier_regression:
        reason = "candidate capture improved but calibration regressed beyond the allowed Brier tolerance"
        champion = "baseline"
    else:
        reason = "candidate improved capture at the fixed review budget with directional bootstrap evidence and acceptable calibration"
        champion = "candidate"
    return PromotionDecision(
        champion=champion,
        reason=reason,
        capture_delta=float(capture_delta),
        capture_delta_ci_low=ci_low,
        capture_delta_ci_high=ci_high,
        brier_delta=float(brier_delta),
        primary_metric=f"capture_at_{config.review_fraction:.2%}_review_budget",
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_summary(path: Path, result: dict[str, Any]) -> None:
    baseline = result["metrics"]["baseline"]
    candidate = result["metrics"]["candidate"]
    decision = result["promotion_decision"]
    lines = [
        "# PaymentOps Champion–Challenger Result",
        "",
        f"- Input rows: {result['split_sizes']['total']}",
        f"- Review budget: {result['review_fraction']:.2%}",
        f"- Candidate backend: {result['candidate_backend']}",
        f"- Champion: **{decision['champion']}**",
        f"- Decision: {decision['reason']}",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "|---|---:|---:|---:|",
        f"| Average precision | {baseline['average_precision']:.6f} | {candidate['average_precision']:.6f} | {candidate['average_precision'] - baseline['average_precision']:+.6f} |",
        f"| ROC-AUC | {baseline['roc_auc'] if baseline['roc_auc'] is not None else 'n/a'} | {candidate['roc_auc'] if candidate['roc_auc'] is not None else 'n/a'} | — |",
        f"| Brier score (lower better) | {baseline['brier']:.6f} | {candidate['brier']:.6f} | {decision['brier_delta']:+.6f} |",
        f"| Capture at review budget | {baseline['capture_at_budget']:.4%} | {candidate['capture_at_budget']:.4%} | {decision['capture_delta']:+.4%} |",
        f"| Lift at review budget | {baseline['lift_at_budget']:.4f} | {candidate['lift_at_budget']:.4f} | {candidate['lift_at_budget'] - baseline['lift_at_budget']:+.4f} |",
        "",
        "## Promotion rule",
        "",
        "Candidate promotion requires a positive paired-bootstrap lower confidence bound for capture uplift and a Brier-score regression no greater than the configured tolerance.",
        "",
        f"- Capture uplift 95% bootstrap CI: [{decision['capture_delta_ci_low']:+.4%}, {decision['capture_delta_ci_high']:+.4%}]",
        "- Result applies only to the supplied dataset and the declared chronological split.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_champion_challenger(
    input_path: str | Path,
    output_dir: str | Path,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Run a leakage-aware baseline-vs-candidate comparison and write evidence."""

    (
        joblib,
        np,
        pd,
        *_rest,
    ) = _require_modeling_dependencies()
    source = Path(input_path)
    destination = Path(output_dir)
    if source.suffix.lower() != ".csv":
        raise ModelingError("only CSV input is currently supported")
    try:
        raw = pd.read_csv(source, nrows=config.max_rows)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ModelingError(f"cannot read input CSV {source}: {exc}") from exc
    frame = prepare_leakage_safe_frame(raw, config)
    train, validation, test = split_chronologically(frame, config)
    tuning, calibration = split_validation_for_tuning_and_calibration(
        validation, config
    )
    categorical, numeric = _feature_columns(config)
    columns = categorical + numeric
    missing_features = sorted(set(columns) - set(frame.columns))
    if missing_features:
        raise ModelingError(f"feature engineering failed to create: {missing_features}")
    baseline = _fit_baseline(train, categorical, numeric)
    candidate, backend = _fit_candidate(train, tuning, categorical, numeric, config)
    baseline_calibration_raw = _predict_probability(baseline, calibration, columns)
    candidate_calibration_raw = _predict_probability(candidate, calibration, columns)
    baseline_calibrator = _fit_sigmoid_calibrator(
        baseline_calibration_raw, calibration["target"]
    )
    candidate_calibrator = _fit_sigmoid_calibrator(
        candidate_calibration_raw, calibration["target"]
    )
    baseline_test_raw = _predict_probability(baseline, test, columns)
    candidate_test_raw = _predict_probability(candidate, test, columns)
    baseline_test = _apply_sigmoid_calibrator(baseline_calibrator, baseline_test_raw)
    candidate_test = _apply_sigmoid_calibrator(candidate_calibrator, candidate_test_raw)
    baseline_metrics = _metrics(test["target"], baseline_test, config.review_fraction)
    candidate_metrics = _metrics(test["target"], candidate_test, config.review_fraction)
    ci_low, ci_high = _bootstrap_capture_delta(
        test["target"],
        baseline_test,
        candidate_test,
        config.review_fraction,
        config.bootstrap_repeats,
        config.random_seed,
        config.bootstrap_sample_size,
        event_times=test["event_time"].to_numpy(dtype="datetime64[ns]"),
        method=config.bootstrap_method,
        block_seconds=config.bootstrap_block_seconds,
    )
    decision = _promotion_decision(
        baseline_metrics, candidate_metrics, ci_low, ci_high, config
    )
    destination.mkdir(parents=True, exist_ok=True)
    model_dir = destination / "models"
    report_dir = destination / "reports"
    model_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)
    joblib.dump(baseline, model_dir / "baseline_logistic.joblib")
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
        "artifact": str(candidate_artifact_path.relative_to(destination)).replace(
            "\\", "/"
        ),
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
    prediction_frame = test[[config.timestamp_column, "target", "event_time"]].copy()
    prediction_frame["baseline_probability"] = baseline_test
    prediction_frame["candidate_probability_raw"] = candidate_test_raw
    prediction_frame["candidate_probability"] = candidate_test
    review_count = baseline_metrics.review_count
    prediction_frame["baseline_review"] = False
    prediction_frame.loc[
        prediction_frame["baseline_probability"].rank(method="first", ascending=False)
        <= review_count,
        "baseline_review",
    ] = True
    prediction_frame["candidate_review"] = False
    prediction_frame.loc[
        prediction_frame["candidate_probability"].rank(method="first", ascending=False)
        <= review_count,
        "candidate_review",
    ] = True
    prediction_frame.to_csv(report_dir / "test_predictions.csv", index=False)

    result = {
        "schema_version": "1.0",
        "status": "PASS",
        "input_path": str(source.resolve()),
        "candidate_backend": backend,
        "candidate_backend_family": backend_family,
        "candidate_weight_mode": selected_weight_mode or None,
        "candidate_provenance": candidate_provenance,
        "review_fraction": config.review_fraction,
        "bootstrap": {
            "method": (
                "paired_temporal_block_bootstrap"
                if config.bootstrap_method == "time_block"
                else "paired_iid_row_bootstrap"
            ),
            "unit": "time_block" if config.bootstrap_method == "time_block" else "row",
            "block_seconds": (
                config.bootstrap_block_seconds
                if config.bootstrap_method == "time_block"
                else None
            ),
            "repeats": config.bootstrap_repeats,
            "sample_size_target": config.bootstrap_sample_size,
        },
        "config": asdict(config),
        "split_sizes": {
            "train": len(train),
            "validation": len(validation),
            "tuning": len(tuning),
            "calibration": len(calibration),
            "test": len(test),
            "total": len(frame),
        },
        "time_ranges": {
            "train": [str(train["event_time"].min()), str(train["event_time"].max())],
            "validation": [
                str(validation["event_time"].min()),
                str(validation["event_time"].max()),
            ],
            "test": [str(test["event_time"].min()), str(test["event_time"].max())],
        },
        "feature_schema": {"categorical": categorical, "numeric": numeric},
        "excluded_outcome_columns": [
            name
            for name in ("RejectionReason", "StatusTimestamp", "ProcessingTimeSecs")
            if name in raw.columns
        ],
        "metrics": {
            "baseline": asdict(baseline_metrics),
            "candidate": asdict(candidate_metrics),
        },
        "promotion_decision": asdict(decision),
    }
    _write_json(report_dir / "experiment_metrics.json", result)
    _write_json(report_dir / "promotion_decision.json", asdict(decision))
    _write_json(report_dir / "feature_schema.json", result["feature_schema"])
    _write_summary(report_dir / "uplift_summary.md", result)
    return result


# PAYMENTOPS_V093_HARDENING
