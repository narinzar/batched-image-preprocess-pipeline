"""Multiprocessing image preprocessing pipeline with a pinned-memory prefetch.

The pipeline maps the CPU-bound transform chain in ``src.transforms`` over a
list of image paths using a process pool, groups the results into batches, and
copies each finished batch into a page-locked (pinned) torch tensor. A pinned
host tensor can be moved to the GPU with ``non_blocking=True`` so the copy
overlaps with compute, which is the point of the prefetch queue.

Design notes:
- Work is chunked so each task returns one image tensor; ``ProcessPoolExecutor``
  keeps ``num_workers`` processes busy.
- Batches are assembled in submission order to keep results reproducible.
- If torch is unavailable or CUDA is absent, the pinned-memory step degrades to
  a plain CPU tensor / numpy array so the pipeline still runs for benchmarking
  the CPU transform throughput.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from .transforms import preprocess

try:  # torch is optional for the pure-CPU path.
    import torch

    _HAVE_TORCH = True
except Exception:  # pragma: no cover - exercised only without torch installed
    torch = None  # type: ignore
    _HAVE_TORCH = False


@dataclass
class PipelineConfig:
    """Configuration for a preprocessing run."""

    target_size: Tuple[int, int] = (224, 224)
    batch_size: int = 64
    num_workers: int = 4
    augment: bool = True
    seed: int = 0
    device: str = "cuda"


def _worker_task(args: Tuple[int, str, Tuple[int, int], bool, int]) -> Tuple[int, np.ndarray]:
    """Top-level worker function (must be picklable for the process pool).

    Returns (index, chw_float32_array) so the parent can reassemble batches in
    order.
    """
    index, path, target_size, augment, seed = args
    # Derive a per-image seed so augmentation is deterministic yet varied.
    arr = preprocess(
        path,
        target_size=target_size,
        augment=augment,
        seed=seed + index,
    )
    return index, arr


def _to_pinned_tensor(batch: np.ndarray):
    """Copy an NCHW float32 numpy batch into a pinned-memory torch tensor.

    Falls back to a normal tensor (or the raw numpy array) when torch or CUDA is
    not available, so benchmarking still works everywhere.
    """
    if not _HAVE_TORCH:
        return batch
    n, c, h, w = batch.shape
    can_pin = torch.cuda.is_available()
    pinned = torch.empty((n, c, h, w), dtype=torch.float32, pin_memory=can_pin)
    pinned.copy_(torch.from_numpy(batch))
    return pinned


def iter_batches(
    paths: Sequence[str],
    config: Optional[PipelineConfig] = None,
    show_progress: bool = True,
) -> Iterator["object"]:
    """Yield preprocessed batches as pinned-memory tensors.

    Each yielded batch is a pinned torch tensor of shape
    (batch_size, 3, H, W) (the last batch may be smaller). Consumers can move it
    to the GPU with ``batch.to(device, non_blocking=True)``.
    """
    if config is None:
        config = PipelineConfig()

    tasks = [
        (i, path, config.target_size, config.augment, config.seed)
        for i, path in enumerate(paths)
    ]

    num_batches = (len(tasks) + config.batch_size - 1) // config.batch_size
    buffer: List[Tuple[int, np.ndarray]] = []

    with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
        results = executor.map(_worker_task, tasks, chunksize=8)
        progress = tqdm(
            results,
            total=len(tasks),
            disable=not show_progress,
            desc=f"preprocess (workers={config.num_workers})",
            unit="img",
        )
        for index, arr in progress:
            buffer.append((index, arr))
            if len(buffer) == config.batch_size:
                yield _assemble_batch(buffer)
                buffer = []
        if buffer:
            yield _assemble_batch(buffer)

    _ = num_batches  # kept for readability of expected batch count.


def _assemble_batch(buffer: List[Tuple[int, np.ndarray]]):
    """Sort a buffer by original index, stack to NCHW, pin the memory."""
    buffer.sort(key=lambda pair: pair[0])
    batch = np.stack([arr for _, arr in buffer], axis=0)
    return _to_pinned_tensor(batch)


def run_pipeline(
    paths: Sequence[str],
    config: Optional[PipelineConfig] = None,
    move_to_device: bool = False,
    show_progress: bool = True,
) -> int:
    """Consume the whole pipeline and return the number of images processed.

    When ``move_to_device`` is set and CUDA is available, each batch is pushed to
    the GPU with a non-blocking transfer to exercise the pinned-memory path.
    """
    if config is None:
        config = PipelineConfig()

    total = 0
    use_cuda = (
        move_to_device
        and _HAVE_TORCH
        and torch.cuda.is_available()
        and config.device.startswith("cuda")
    )
    for batch in iter_batches(paths, config=config, show_progress=show_progress):
        if use_cuda:
            gpu_batch = batch.to(config.device, non_blocking=True)
            total += int(gpu_batch.shape[0])
        else:
            total += int(batch.shape[0])
    if use_cuda:
        torch.cuda.synchronize()
    return total


def default_worker_grid(max_workers: Optional[int] = None) -> List[int]:
    """Return the sweep 1, 2, 4, 8, ... capped at the CPU count."""
    cap = max_workers or os.cpu_count() or 1
    grid: List[int] = []
    w = 1
    while w < cap:
        grid.append(w)
        w *= 2
    grid.append(cap)
    # De-duplicate while preserving order (e.g. when cap is a power of two).
    seen = set()
    unique = []
    for w in grid:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return unique
