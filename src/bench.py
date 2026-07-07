"""Worker-count sweep: measure images/sec at several worker counts.

Given a fixed set of image paths, run the full pipeline once per worker count in
a grid and record throughput. Returns a plain list of result rows so callers can
print a table, dump JSON, or plot it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Callable, List, Optional, Sequence

from .pipeline import PipelineConfig, default_worker_grid, run_pipeline


@dataclass
class BenchRow:
    """One measured point in the worker-count sweep."""

    num_workers: int
    images: int
    seconds: float
    images_per_sec: float


def measure_throughput(
    paths: Sequence[str],
    num_workers: int,
    batch_size: int = 64,
    target_size=(224, 224),
    augment: bool = True,
    move_to_device: bool = False,
    show_progress: bool = True,
) -> BenchRow:
    """Time a single full pass of the pipeline at a given worker count."""
    config = PipelineConfig(
        target_size=target_size,
        batch_size=batch_size,
        num_workers=num_workers,
        augment=augment,
    )
    start = time.perf_counter()
    images = run_pipeline(
        paths,
        config=config,
        move_to_device=move_to_device,
        show_progress=show_progress,
    )
    elapsed = time.perf_counter() - start
    ips = images / elapsed if elapsed > 0 else 0.0
    return BenchRow(
        num_workers=num_workers,
        images=images,
        seconds=elapsed,
        images_per_sec=ips,
    )


def sweep_workers(
    paths: Sequence[str],
    worker_grid: Optional[Sequence[int]] = None,
    batch_size: int = 64,
    target_size=(224, 224),
    augment: bool = True,
    move_to_device: bool = False,
    show_progress: bool = True,
) -> List[BenchRow]:
    """Run the pipeline once per worker count and return one BenchRow each."""
    grid = list(worker_grid) if worker_grid is not None else default_worker_grid()
    rows: List[BenchRow] = []
    for workers in grid:
        row = measure_throughput(
            paths,
            num_workers=workers,
            batch_size=batch_size,
            target_size=target_size,
            augment=augment,
            move_to_device=move_to_device,
            show_progress=show_progress,
        )
        rows.append(row)
    return rows


def rows_to_dicts(rows: Sequence[BenchRow]) -> List[dict]:
    """Convert BenchRow records to plain dicts for JSON serialization."""
    return [asdict(r) for r in rows]


def format_table(rows: Sequence[BenchRow]) -> str:
    """Render a small fixed-width table of the sweep results."""
    header = f"{'workers':>8} | {'images':>7} | {'seconds':>8} | {'images/sec':>11}"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r.num_workers:>8} | {r.images:>7} | {r.seconds:>8.3f} | {r.images_per_sec:>11.1f}"
        )
    return "\n".join(lines)


def best_row(rows: Sequence[BenchRow]) -> BenchRow:
    """Return the row with the highest measured throughput."""
    return max(rows, key=lambda r: r.images_per_sec)


# Type alias for a throughput function used by tests / autotuner injection.
ThroughputFn = Callable[[int], float]
