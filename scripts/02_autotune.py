"""Run the adaptive worker-count tuner against the real pipeline and print it.

Usage:
    python scripts/02_autotune.py --images-dir data/images --gpu

Prints the chosen worker count and the measured probe curve. The tuner probes
far fewer worker counts than the full sweep in scripts/01_bench_workers.py.
"""

from __future__ import annotations

import argparse
import glob
import os

from dotenv import load_dotenv

from src.autotune import autotune_pipeline


def collect_paths(images_dir: str) -> list:
    paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    if not paths:
        raise SystemExit(
            f"no PNG images found in {images_dir}; run "
            "scripts/00_make_sample_images.py first."
        )
    return paths


def main() -> None:
    load_dotenv()  # no secrets required.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=str, default=os.path.join("data", "images"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--target-size", type=int, default=224)
    parser.add_argument("--gpu", action="store_true", help="use CUDA non_blocking transfer.")
    args = parser.parse_args()

    paths = collect_paths(args.images_dir)
    print(f"autotuning worker count over {len(paths)} images...")

    result = autotune_pipeline(
        paths,
        batch_size=args.batch_size,
        target_size=(args.target_size, args.target_size),
        move_to_device=args.gpu,
    )

    print("\nprobe curve (workers -> images/sec):")
    for workers, ips in result.curve:
        print(f"  {workers:>3} workers : {ips:>10.1f} images/sec")
    print(
        f"\nchosen: {result.best_workers} workers "
        f"({result.best_throughput:.1f} images/sec)"
    )


if __name__ == "__main__":
    main()
