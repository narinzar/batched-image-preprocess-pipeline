"""Generate synthetic RGB images so the pipeline has real files to process.

These are SYNTHETIC images (random noise blended with smooth gradients). They
carry no license concerns and stand in for a real dataset when benchmarking the
preprocessing pipeline. Run this once before the benchmark scripts.

Usage:
    python scripts/00_make_sample_images.py --count 4000 --size 256
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

DEFAULT_OUT = os.path.join("data", "images")


def make_synthetic_image(size: int, seed: int) -> np.ndarray:
    """Create one HWC uint8 RGB image: a color gradient plus random noise."""
    rng = np.random.default_rng(seed)
    ys = np.linspace(0, 1, size, dtype=np.float32).reshape(size, 1)
    xs = np.linspace(0, 1, size, dtype=np.float32).reshape(1, size)
    # Per-channel smooth gradients with random weights.
    wr, wg, wb = rng.random(3)
    r = (ys * (1 - wr) + xs * wr)
    g = (ys * wg + xs * (1 - wg))
    b = ((ys + xs) * 0.5 * wb + (1 - wb) * (1 - ys))
    grad = np.stack([r, g, b], axis=2)
    noise = rng.random((size, size, 3), dtype=np.float32) * 0.25
    blended = np.clip(grad * 0.75 + noise, 0.0, 1.0)
    return (blended * 255.0).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=4000, help="number of images")
    parser.add_argument("--size", type=int, default=256, help="square image side in px")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT, help="output directory")
    parser.add_argument("--seed", type=int, default=0, help="base RNG seed")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for i in tqdm(range(args.count), desc="generating synthetic images", unit="img"):
        arr = make_synthetic_image(args.size, seed=args.seed + i)
        Image.fromarray(arr, mode="RGB").save(
            os.path.join(args.out, f"synthetic_{i:06d}.png")
        )
    print(f"wrote {args.count} synthetic images to {args.out}")


if __name__ == "__main__":
    main()
