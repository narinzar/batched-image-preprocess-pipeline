"""Adaptive worker-count tuner.

Instead of sweeping every worker count, this hill-climbs to a good one:

1. Start at 1 worker and measure throughput.
2. Double the worker count while throughput keeps improving (exponential probe).
   Stop doubling once a step does not beat the best-so-far, or once the CPU
   count cap is reached.
3. Do a bounded local search around the best doubling point, checking the
   integer neighbours between the last two probes, to catch a peak that sits
   between two powers of two (e.g. the true best is 6 when we probed 4 and 8).

The tuner takes a ``throughput_fn`` callable so it can be driven either by the
real pipeline or by a synthetic function in tests. It returns the chosen worker
count and the full measured curve (worker_count -> images/sec) in probe order.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

ThroughputFn = Callable[[int], float]


@dataclass
class AutotuneResult:
    """Outcome of an adaptive tuning run."""

    best_workers: int
    best_throughput: float
    # Curve in the order points were probed: list of (workers, images_per_sec).
    curve: List[Tuple[int, float]] = field(default_factory=list)

    @property
    def measured(self) -> Dict[int, float]:
        """Curve as a dict for convenient lookup."""
        return {w: t for w, t in self.curve}


def autotune_workers(
    throughput_fn: ThroughputFn,
    max_workers: Optional[int] = None,
    min_gain: float = 1.0,
) -> AutotuneResult:
    """Hill-climb to the best worker count.

    Args:
        throughput_fn: maps a worker count to measured images/sec. Called once
            per distinct worker count (results are cached).
        max_workers: upper bound on workers; defaults to os.cpu_count().
        min_gain: multiplicative threshold a new point must beat to count as an
            improvement. 1.0 means "any improvement counts"; use e.g. 1.02 to
            require at least a 2% gain before continuing to double.

    Returns:
        AutotuneResult with the chosen worker count and the probe curve.
    """
    cap = max_workers or os.cpu_count() or 1
    cache: Dict[int, float] = {}
    curve: List[Tuple[int, float]] = []

    def measure(w: int) -> float:
        w = max(1, min(w, cap))
        if w not in cache:
            value = throughput_fn(w)
            cache[w] = value
            curve.append((w, value))
        return cache[w]

    # Step 1 + 2: exponential probe starting at 1 worker.
    current = 1
    best_workers = 1
    best_tp = measure(1)

    prev = 1
    while current < cap:
        nxt = min(current * 2, cap)
        if nxt == current:
            break
        tp = measure(nxt)
        if tp > best_tp * min_gain:
            best_tp = tp
            best_workers = nxt
            prev = current
            current = nxt
        else:
            # Throughput stopped improving; the peak is at or below nxt.
            prev = current
            current = nxt
            break

    # Step 3: local search between prev and current (inclusive), which brackets
    # the best doubling point. This finds peaks that lie between powers of two.
    low = max(1, min(prev, best_workers) - 1)
    high = min(cap, max(current, best_workers) + 1)
    for w in range(low, high + 1):
        tp = measure(w)
        if tp > best_tp:
            best_tp = tp
            best_workers = w

    return AutotuneResult(
        best_workers=best_workers,
        best_throughput=best_tp,
        curve=curve,
    )


def autotune_pipeline(
    paths,
    batch_size: int = 64,
    target_size=(224, 224),
    augment: bool = True,
    move_to_device: bool = False,
    max_workers: Optional[int] = None,
) -> AutotuneResult:
    """Convenience wrapper that tunes against the real pipeline.

    Builds a throughput function that runs one full pass of the pipeline per
    worker count and returns images/sec, then hands it to ``autotune_workers``.
    """
    from .bench import measure_throughput

    def throughput_fn(workers: int) -> float:
        row = measure_throughput(
            paths,
            num_workers=workers,
            batch_size=batch_size,
            target_size=target_size,
            augment=augment,
            move_to_device=move_to_device,
            show_progress=False,
        )
        return row.images_per_sec

    return autotune_workers(throughput_fn, max_workers=max_workers)
