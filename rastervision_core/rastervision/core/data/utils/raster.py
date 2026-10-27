import math

import numpy as np

from rastervision.core.box import Box


def fill_overflow(bbox: Box,
                  window: Box,
                  chip: np.ndarray,
                  fill_value: int = 0) -> np.ndarray:
    """Where ``chip``'s ``window`` overflows bbox, fill with ``fill_value``.

    The chip may have been resized, i.e., its height and width do not need to
    match those of the window. A pixel of the chip is filled if its center
    lies outside the bbox.

    Args:
        bbox (Box): Bounding box.
        window (Box): Window corresponding to the ``chip``.
        chip (np.ndarray): (..., H, W, C) array.
        fill_value (int): Value to set overflowing pixels to.
            Defaults to 0.

    Returns:
        np.ndarray: Chip with overflowing regions filled with ``fill_value``.
    """
    top_overflow = max(0, bbox.ymin - window.ymin)
    bottom_overflow = max(0, window.ymax - bbox.ymax)
    left_overflow = max(0, bbox.xmin - window.xmin)
    right_overflow = max(0, window.xmax - bbox.xmax)

    *_, h, w, _ = chip.shape
    win_h, win_w = window.size
    if win_h <= 0 or win_w <= 0:
        return chip
    # convert overflows from window pixels to chip pixels
    scale_y, scale_x = h / win_h, w / win_w

    def num_overflowing_pixels(overflow: float) -> int:
        # number of pixels whose centers lie within the overflow
        return max(0, math.ceil(overflow - 0.5))

    ymin = num_overflowing_pixels(top_overflow * scale_y)
    ymax = max(0, h - num_overflowing_pixels(bottom_overflow * scale_y))
    xmin = num_overflowing_pixels(left_overflow * scale_x)
    xmax = max(0, w - num_overflowing_pixels(right_overflow * scale_x))

    chip[..., :ymin, :, :] = fill_value
    chip[..., ymax:, :, :] = fill_value
    chip[..., :, :xmin, :] = fill_value
    chip[..., :, xmax:, :] = fill_value
    return chip


def pad_to_window_size(chip: np.ndarray, window: Box, bbox: Box,
                       fill_value: int) -> np.ndarray:
    """Where chip's window overflows bbox, pad chip with fill_value.

    Args:
        chip (np.ndarray): (H, W[, C]) array.
        bbox (Box): Bounding box.
        window (Box): Window corresponding to the ``chip``.
        fill_value (int): Value to pad with. Defaults to 0.

    Returns:
        np.ndarray: Chip of size equal to window size with edges padded with
        fill_value where needed.
    """
    if window in bbox:
        return chip

    top_overflow = max(0, bbox.ymin - window.ymin)
    bottom_overflow = max(0, window.ymax - bbox.ymax)
    left_overflow = max(0, bbox.xmin - window.xmin)
    right_overflow = max(0, window.xmax - bbox.xmax)

    h, w = window.size
    ymin, ymax = top_overflow, (h - bottom_overflow)
    xmin, xmax = left_overflow, (w - right_overflow)

    if chip.ndim == 2:
        out_shape = (h, w)
        out = np.full(out_shape, fill_value)
        out[ymin:ymax, xmin:xmax] = chip
    else:
        *dims, _, _, c = chip.shape
        out_shape = (*dims, h, w, c)
        out = np.full(out_shape, fill_value)
        out[..., ymin:ymax, xmin:xmax, :] = chip
    return out
