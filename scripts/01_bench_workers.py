"""Run the worker-count sweep and save a JSON table plus a matplotlib plot.

Usage:
    python scripts/01_bench_workers.py --images-dir data/images --gpu

Writes:
    outputs/throughput.json  (list of {num_workers, images, seconds, images_per_sec})
    outputs/throughput.png   (images/sec vs worker count)
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from dotenv import load_dotenv

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for saving PNGs.
import matplotlib.pyplot as plt

from src.bench import best_row, format_table, rows_to_dicts, sweep_workers
from src.pipeline import default_worker_grid


def collect_paths(images_dir: str) -> list:
    """Return a sorted list of PNG paths under images_dir."""
    paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    if not paths:
        raise SystemExit(
            f"no PNG images found in {images_dir}; run "
            "scripts/00_make_sample_images.py first."
        )
    return paths


def main() -> None:
    load_dotenv()  # no secrets required; loads optional overrides if present.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=str, default=os.path.join("data", "images"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--target-size", type=int, default=224)
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="move batches to CUDA with non_blocking transfer (needs a GPU).",
    )
    parser.add_argument("--out-dir", type=str, default="outputs")
    args = parser.parse_args()

    paths = collect_paths(args.images_dir)
    grid = default_worker_grid()
    print(f"benchmarking {len(paths)} images over worker grid {grid}")

    rows = sweep_workers(
        paths,
        worker_grid=grid,
        batch_size=args.batch_size,
        target_size=(args.target_size, args.target_size),
        move_to_device=args.gpu,
        show_progress=True,
    )

    print("\n" + format_table(rows))
    top = best_row(rows)
    print(f"\nbest: {top.num_workers} workers at {top.images_per_sec:.1f} images/sec")

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "throughput.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(rows_to_dicts(rows), fh, indent=2)
    print(f"wrote {json_path}")

    png_path = os.path.join(args.out_dir, "throughput.png")
    workers = [r.num_workers for r in rows]
    ips = [r.images_per_sec for r in rows]
    plt.figure(figsize=(7, 4.5))
    plt.plot(workers, ips, marker="o")
    plt.xlabel("worker count")
    plt.ylabel("images / sec")
    plt.title("Preprocessing throughput vs worker count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=120)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
