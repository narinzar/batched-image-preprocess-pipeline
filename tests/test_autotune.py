"""Tests for the adaptive worker-count tuner using synthetic throughput curves."""

from src.autotune import autotune_workers


def test_selects_peak_at_four_workers():
    # A curve that peaks exactly at 4 workers then degrades (oversubscription).
    curve = {1: 100.0, 2: 180.0, 4: 300.0, 8: 220.0, 16: 150.0}

    def throughput_fn(w: int) -> float:
        return curve[w]

    result = autotune_workers(throughput_fn, max_workers=16)
    assert result.best_workers == 4
    assert result.best_throughput == 300.0


def test_finds_peak_between_powers_of_two():
    # True peak is 6, which lies between the doubling probes 4 and 8.
    def throughput_fn(w: int) -> float:
        # Inverted parabola centred at 6.
        return -( (w - 6) ** 2 ) + 100.0

    result = autotune_workers(throughput_fn, max_workers=16)
    assert result.best_workers == 6


def test_monotonic_increasing_picks_the_cap():
    # Throughput keeps rising; the tuner should climb to the max worker count.
    def throughput_fn(w: int) -> float:
        return float(w)

    result = autotune_workers(throughput_fn, max_workers=8)
    assert result.best_workers == 8


def test_single_worker_when_more_is_worse():
    # More workers always hurts; the tuner should stay at 1.
    def throughput_fn(w: int) -> float:
        return 100.0 / w

    result = autotune_workers(throughput_fn, max_workers=16)
    assert result.best_workers == 1


def test_curve_is_recorded_and_cached():
    calls = []

    def throughput_fn(w: int) -> float:
        calls.append(w)
        return {1: 10.0, 2: 20.0, 4: 15.0}.get(w, 5.0)

    result = autotune_workers(throughput_fn, max_workers=4)
    # Each distinct worker count is measured at most once (caching).
    assert len(calls) == len(set(calls))
    # The curve holds every probed point.
    assert dict(result.curve)[2] == 20.0
