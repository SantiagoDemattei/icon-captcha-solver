import cv2
import numpy as np
import pytest

from icon_solver.segmentation import region_from_mask
from icon_solver.shapes import CROP_SIZE, encode_regions, region_crop, square_pad, to_input, to_polar


def _shape() -> np.ndarray:
    mask = np.zeros((CROP_SIZE, CROP_SIZE), np.uint8)
    cv2.fillPoly(mask, [np.array([[10, 12], [36, 8], [30, 20], [38, 38], [14, 30]])], 255)
    return mask


@pytest.mark.parametrize("quarter_turns", [1, 2, 3])
def test_rotating_the_mask_is_a_roll_of_the_angular_axis(quarter_turns):
    mask = _shape()
    rotated = np.ascontiguousarray(np.rot90(mask, quarter_turns))
    rows_per_quarter = CROP_SIZE // 4

    assert np.array_equal(to_polar(rotated), np.roll(to_polar(mask), -quarter_turns * rows_per_quarter, axis=0))


def test_input_is_polar_uint8_divided_by_255():
    polar = to_polar(_shape())
    x = to_input(polar[None])

    assert x.shape == (1, 1, CROP_SIZE, CROP_SIZE)
    assert x.dtype.is_floating_point
    assert np.array_equal(x[0, 0].numpy(), polar.astype(np.float32) / np.float32(255))


def test_encode_regions_matches_the_step_by_step_encoding():
    mask = np.zeros((200, 300), np.uint8)
    cv2.circle(mask, (80, 60), 15, 255, cv2.FILLED)
    region = region_from_mask(mask)
    assert region is not None

    x = encode_regions([region, region])

    expected = to_input(to_polar(region_crop(region.contour))[None])[0]
    assert x.shape == (2, 1, CROP_SIZE, CROP_SIZE)
    assert all(np.array_equal(x[i].numpy(), expected.numpy()) for i in range(2))


def test_square_pad_centers_without_stretching():
    padded = square_pad(np.ones((2, 6), np.uint8))

    assert padded.shape == (6, 6)
    assert padded.sum() == 12
    assert padded[2:4].all() and not padded[:2].any() and not padded[4:].any()
