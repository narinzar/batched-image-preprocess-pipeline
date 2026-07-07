# batched-image-preprocess-pipeline

A high-throughput image preprocessing pipeline (resize -> normalize -> augment)
that runs the CPU transforms across a process pool and hands finished batches to
a pinned-memory prefetch queue for non-blocking GPU transfer. It includes a
worker-count benchmark and an adaptive tuner that finds a good worker count
automatically.

## Problem

Feeding a fast GPU (an RTX 5090 here) from CPU-side image decode + resize +
normalize is a classic bottleneck: a single Python process cannot decode images
fast enough to keep the accelerator busy, and copying each batch from pageable
host memory serializes the host-to-device transfer. The right worker count is
also machine-specific - too few workers starve the GPU, too many oversubscribe
the cores and thrash. This repo addresses both: it parallelizes the transforms
and overlaps the GPU copy, and it measures throughput to pick the worker count
instead of guessing.

## Approach

- Pure, deterministic transforms (`src/transforms.py`): load -> resize (PIL
  bilinear) -> optional random flip -> normalize to float32 CHW with mean/std.
  Seeded so a run reproduces exactly, and side-effect-free so they run safely in
  worker processes.
- Process-pool pipeline (`src/pipeline.py`) using
  `concurrent.futures.ProcessPoolExecutor` to map transforms over image paths,
  reassembling results into batches in submission order.
- Pinned-memory prefetch: each finished batch is copied into a
  `torch.empty(..., pin_memory=True)` tensor so it can be pushed to the GPU with
  `.to(device, non_blocking=True)`, letting the copy overlap with compute.
- Worker-count sweep (`src/bench.py`): times one full pass per worker count in
  the grid `1, 2, 4, 8, ...` up to `os.cpu_count()` and reports images/sec.
- Adaptive tuner (`src/autotune.py`): starts at 1 worker, doubles while
  throughput improves, then does a bounded local search to catch a peak that
  sits between two powers of two (for example a true best of 6 when it probed 4
  and 8). This is the original contribution over a plain sweep.

## Setup

```
# Create and activate a virtual environment (either tool works).
uv venv --python 3.12 .venv
# or: python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate

# Install torch from the CUDA 12.8 wheel index for an RTX 5090 (sm_120):
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Then the rest of the dependencies:
pip install -r requirements.txt

# No secrets are needed, but copy the example env for consistency:
cp .env.example .env
```

A CPU-only torch build also runs everything except the actual CUDA transfer
path; in that case the pipeline falls back to a plain CPU tensor so benchmarks
of the CPU transform stage still work.

## How to run

```
# 1. Generate synthetic sample images into data/images/ (SYNTHETIC data).
python scripts/00_make_sample_images.py --count 4000 --size 256

# 2. Sweep worker counts, save outputs/throughput.json and outputs/throughput.png.
python scripts/01_bench_workers.py --images-dir data/images
#    add --gpu to exercise the pinned-memory non_blocking transfer to CUDA.

# 3. Run the adaptive tuner and print the chosen worker count.
python scripts/02_autotune.py --images-dir data/images
#    add --gpu to tune against the GPU transfer path.

# Run the tests.
pytest -q
```

## Results

Numbers below are produced by running the commands above; this repo ships the
code, run it to populate them.

Reproduction:

```
python scripts/00_make_sample_images.py --count 4000 --size 256
python scripts/01_bench_workers.py --images-dir data/images --gpu
python scripts/02_autotune.py --images-dir data/images --gpu
```

Expected qualitative behavior:

- Throughput (images/sec) should rise with worker count up to roughly the
  physical core count, then plateau or dip as more workers oversubscribe the
  cores and add scheduling and IPC overhead.
- Enabling `--gpu` (pinned host memory + `non_blocking=True` transfer) should
  raise effective end-to-end throughput versus a pageable copy, because the
  host-to-device copy overlaps with ongoing CPU preprocessing instead of
  blocking on it.
- The adaptive tuner in step 3 should converge to a worker count at or near the
  peak of the step-2 sweep while probing far fewer points than the full sweep.

Results table (populated by running step 2):

| workers | images/sec |
| ------- | ---------- |
| 1       | TBD (run)  |
| 2       | TBD (run)  |
| 4       | TBD (run)  |
| 8       | TBD (run)  |
| ...     | TBD (run)  |

The sweep also writes `outputs/throughput.png` (images/sec vs worker count).

## What I would do next at larger scale

Move decode off the Python workers by using an image backend that releases the
GIL or a GPU decoder (nvJPEG / DALI) so the CPU only shuffles bytes, and replace
the per-run sweep with a persistent shared-memory ring buffer feeding a
double-buffered CUDA stream. The autotuner would then re-tune periodically to
track load and thermal changes rather than picking a worker count once.
