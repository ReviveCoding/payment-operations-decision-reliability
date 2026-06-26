from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "run_champion_challenger_v094.py"
    spec = importlib.util.spec_from_file_location("paymentops_v094_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_transaction_mass_blocks_preserve_timestamp_groups() -> None:
    module = _load_module()
    group_count = 1_500
    event_times = np.repeat(
        np.datetime64("2022-01-01T00:00")
        + np.arange(group_count).astype("timedelta64[m]"),
        2,
    ).astype("datetime64[ns]")
    blocks = module._build_transaction_mass_blocks(
        event_times,
        target_block_rows=1_000,
        np=np,
    )
    assigned = np.concatenate(blocks)
    assert np.array_equal(assigned, np.arange(len(event_times)))
    assert len(blocks) >= 2
    for block in blocks:
        start = int(block[0])
        stop = int(block[-1]) + 1
        if start > 0:
            assert event_times[start - 1] != event_times[start]
        if stop < len(event_times):
            assert event_times[stop - 1] != event_times[stop]


def test_transaction_mass_bootstrap_is_deterministic_and_paired() -> None:
    module = _load_module()
    rows = 4_000
    times = np.repeat(
        np.datetime64("2022-01-01T00:00")
        + np.arange(rows // 4).astype("timedelta64[m]"),
        4,
    ).astype("datetime64[ns]")
    labels = np.zeros(rows, dtype=int)
    labels[::25] = 1
    baseline = np.linspace(0.0, 1.0, rows)
    candidate = baseline.copy()
    candidate[labels == 1] += 1.0

    class Metric:
        def __init__(self, capture_at_budget: float):
            self.capture_at_budget = capture_at_budget

    def metrics_fn(y, scores, review_fraction):
        count = max(1, int(np.ceil(len(y) * review_fraction)))
        order = np.argsort(-scores, kind="stable")[:count]
        positives = int(y.sum())
        capture = 0.0 if positives == 0 else float(y[order].sum() / positives)
        return Metric(capture)

    first = module._transaction_mass_bootstrap(
        labels,
        baseline,
        candidate,
        times,
        review_fraction=0.05,
        repeats=100,
        seed=17,
        sample_size_target=1_000,
        target_block_rows=1_000,
        metrics_fn=metrics_fn,
        np=np,
    )
    second = module._transaction_mass_bootstrap(
        labels,
        baseline,
        candidate,
        times,
        review_fraction=0.05,
        repeats=100,
        seed=17,
        sample_size_target=1_000,
        target_block_rows=1_000,
        metrics_fn=metrics_fn,
        np=np,
    )
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert first[2]["repeats_used"] == 100
    assert first[0] >= 0.0
