"""Pure-function image transforms built on numpy and PIL.

Every transform is deterministic given an explicit seed so that a run can be
reproduced exactly. Transforms operate on numpy arrays in HWC uint8 layout as
loaded from disk and produce a normalized float32 tensor in CHW layout.

The functions here have no side effects and no global state, which makes them
safe to call from worker processes.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from PIL import Image

# ImageNet-style defaults; callers may pass their own mean/std.
DEFAULT_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
DEFAULT_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def load_image(path: str) -> np.ndarray:
    """Load an image from disk as an HWC uint8 RGB numpy array."""
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        return np.asarray(rgb, dtype=np.uint8)


def resize(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize an HWC uint8 array to target_size (height, width).

    Uses bilinear resampling via PIL. The result stays HWC uint8.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HWC RGB image, got shape {image.shape}")
    target_h, target_w = target_size
    pil = Image.fromarray(image, mode="RGB")
    # PIL size is (width, height).
    resized = pil.resize((target_w, target_h), resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def normalize(
    image: np.ndarray,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
) -> np.ndarray:
    """Scale an HWC uint8 image to [0, 1], normalize, return float32 CHW.

    Output layout is CHW so it can be stacked directly into an NCHW batch that
    torch expects.
    """
    if image.dtype != np.uint8:
        raise ValueError(f"expected uint8 input, got {image.dtype}")
    scaled = image.astype(np.float32) / 255.0
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    normed = (scaled - mean_arr) / std_arr
    # HWC -> CHW
    return np.ascontiguousarray(normed.transpose(2, 0, 1))


def random_flip(image: np.ndarray, seed: int) -> np.ndarray:
    """Randomly flip an HWC image horizontally, deterministic given seed.

    Returns a flipped copy with 50% probability, otherwise the input unchanged.
    Applying the same seed twice is idempotent, and a flipped image flipped
    again with a forced flip recovers the original (see tests).
    """
    rng = np.random.default_rng(seed)
    if rng.random() < 0.5:
        return np.ascontiguousarray(image[:, ::-1, :])
    return image


def flip_horizontal(image: np.ndarray) -> np.ndarray:
    """Unconditionally flip an HWC image horizontally. Its own inverse."""
    return np.ascontiguousarray(image[:, ::-1, :])


def random_crop(image: np.ndarray, crop_size: Tuple[int, int], seed: int) -> np.ndarray:
    """Deterministically crop an HWC image to crop_size (height, width).

    If the image is smaller than the crop in a dimension, it is first resized up
    so a valid crop exists.
    """
    crop_h, crop_w = crop_size
    h, w = image.shape[:2]
    if h < crop_h or w < crop_w:
        image = resize(image, (max(h, crop_h), max(w, crop_w)))
        h, w = image.shape[:2]
    rng = np.random.default_rng(seed)
    top = int(rng.integers(0, h - crop_h + 1))
    left = int(rng.integers(0, w - crop_w + 1))
    return np.ascontiguousarray(image[top : top + crop_h, left : left + crop_w, :])


def preprocess(
    path: str,
    target_size: Tuple[int, int] = (224, 224),
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
    augment: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """Full transform chain: load -> resize -> augment -> normalize.

    Returns a float32 CHW array of shape (3, target_h, target_w). Deterministic
    for a fixed path and seed. Augmentation (random flip) is applied before
    normalization so value statistics are unchanged.
    """
    image = load_image(path)
    image = resize(image, target_size)
    if augment:
        image = random_flip(image, seed=seed)
    return normalize(image, mean=mean, std=std)
