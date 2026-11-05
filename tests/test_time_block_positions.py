import numpy as np
import pytest

from payment_ops_hardening import modeling


@pytest.mark.parametrize("seed", [0, 51])
@pytest.mark.parametrize("sample_size", [None, 41])
def test_time_blocks_keep_draws_and_within_block_row_order(
    seed, sample_size, monkeypatch
):
    count, draws = 97, 60
    rng = np.random.default_rng(17)
    labels = np.arange(count) % 2
    baseline, candidate = rng.random((2, count))
    times = np.datetime64("2026-01-01", "ns") + (np.arange(count) % 13).astype(
        "timedelta64[h]"
    )
    codes = times.astype("int64") // (3600 * 1_000_000_000)
    _, inverse = np.unique(codes, return_inverse=True)
    blocks = [
        np.flatnonzero(inverse == index) for index in range(int(inverse.max()) + 1)
    ]
    draw_size = count if sample_size is None else sample_size
    block_draws = max(
        1, int(np.ceil(draw_size / np.mean([len(block) for block in blocks])))
    )
    draw_rng = np.random.default_rng(seed)
    metric = modeling._metrics
    expected_calls, differences = [], []
    for _ in range(draws):
        selected = draw_rng.integers(0, len(blocks), size=block_draws)
        positions = np.concatenate([blocks[index] for index in selected])
        if labels[positions].sum() == 0:
            continue
        left = metric(labels[positions], baseline[positions], 0.1)
        right = metric(labels[positions], candidate[positions], 0.1)
        differences.append(right.capture_at_budget - left.capture_at_budget)
        expected_calls.extend(
            [
                (labels[positions], baseline[positions]),
                (labels[positions], candidate[positions]),
            ]
        )
    calls = iter(expected_calls)

    def verify_rows(targets, scores, fraction):
        expected_targets, expected_scores = next(calls)
        np.testing.assert_array_equal(targets, expected_targets)
        np.testing.assert_array_equal(scores, expected_scores)
        return metric(targets, scores, fraction)

    monkeypatch.setattr(modeling, "_metrics", verify_rows)
    result = modeling._bootstrap_capture_delta(
        labels,
        baseline,
        candidate,
        0.1,
        draws,
        seed,
        sample_size,
        event_times=times,
        method="time_block",
        block_seconds=3600,
    )
    assert result == (
        float(np.quantile(differences, 0.025)),
        float(np.quantile(differences, 0.975)),
    )
    assert next(calls, None) is None
