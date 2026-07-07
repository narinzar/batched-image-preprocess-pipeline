"""Tests for the pure-function transforms: shapes, dtypes, ranges, determinism."""

import numpy as np

from src.transforms import (
    DEFAULT_MEAN,
    DEFAULT_STD,
    flip_horizontal,
    normalize,
    random_crop,
    random_flip,
    resize,
)


def make_image(h=64, w=48, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_resize_shape_and_dtype():
    img = make_image(64, 48)
    out = resize(img, (32, 40))  # (height, width)
    assert out.shape == (32, 40, 3)
    assert out.dtype == np.uint8


def test_normalize_shape_dtype_and_values():
    img = make_image(16, 16)
    out = normalize(img, mean=DEFAULT_MEAN, std=DEFAULT_STD)
    # HWC uint8 -> CHW float32.
    assert out.shape == (3, 16, 16)
    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]

    # A zero image maps to -mean/std; a full-white image to (1-mean)/std.
    zeros = np.zeros((4, 4, 3), dtype=np.uint8)
    ones = np.full((4, 4, 3), 255, dtype=np.uint8)
    z = normalize(zeros)
    o = normalize(ones)
    for c in range(3):
        expected_low = -DEFAULT_MEAN[c] / DEFAULT_STD[c]
        expected_high = (1.0 - DEFAULT_MEAN[c]) / DEFAULT_STD[c]
        assert np.allclose(z[c], expected_low, atol=1e-5)
        assert np.allclose(o[c], expected_high, atol=1e-5)


def test_normalize_value_range_is_bounded():
    img = make_image(32, 32)
    out = normalize(img)
    # With ImageNet stats, every normalized value stays inside these bounds.
    lo = min(-m / s for m, s in zip(DEFAULT_MEAN, DEFAULT_STD))
    hi = max((1 - m) / s for m, s in zip(DEFAULT_MEAN, DEFAULT_STD))
    assert out.min() >= lo - 1e-4
    assert out.max() <= hi + 1e-4


def test_flip_horizontal_is_its_own_inverse():
    img = make_image(20, 24, seed=7)
    twice = flip_horizontal(flip_horizontal(img))
    assert np.array_equal(twice, img)


def test_flip_horizontal_actually_flips():
    img = make_image(8, 8, seed=3)
    flipped = flip_horizontal(img)
    assert np.array_equal(flipped[:, 0, :], img[:, -1, :])


def test_random_flip_is_deterministic_given_seed():
    img = make_image(12, 12, seed=1)
    a = random_flip(img, seed=123)
    b = random_flip(img, seed=123)
    assert np.array_equal(a, b)


def test_random_crop_shape_and_determinism():
    img = make_image(64, 64, seed=5)
    a = random_crop(img, (32, 32), seed=42)
    b = random_crop(img, (32, 32), seed=42)
    assert a.shape == (32, 32, 3)
    assert np.array_equal(a, b)


def test_random_crop_upscales_when_too_small():
    img = make_image(10, 10, seed=9)
    out = random_crop(img, (32, 32), seed=0)
    assert out.shape == (32, 32, 3)
