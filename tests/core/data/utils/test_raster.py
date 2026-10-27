import unittest

import numpy as np

from rastervision.core.box import Box
from rastervision.core.data.utils.raster import (pad_to_window_size,
                                                 fill_overflow)


class TestRasterUtils(unittest.TestCase):
    def test_fill_overflow(self):
        bbox = Box(10, 10, 90, 90)
        window = Box(0, 0, 100, 100)
        chip_in = np.ones((2, 100, 100, 4), dtype=np.uint8)
        chip_out = fill_overflow(bbox, window, chip_in)
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:90, 10:90] = 1
        self.assertTrue(np.all(chip_out[:, mask] == 1))
        self.assertTrue(np.all(chip_out[:, ~mask] == 0))

    def test_fill_overflow_resized_chip(self):
        # 100x100 window overflowing the bbox by 20 px at the top and 10 px
        # on the right, resized to 50x40
        bbox = Box(0, 0, 100, 100)
        window = Box(-20, 10, 80, 110)
        chip_in = np.ones((50, 40, 1), dtype=np.uint8)
        chip_out = fill_overflow(bbox, window, chip_in)
        mask = np.zeros((50, 40), dtype=bool)
        # 20 * 50 / 100 = 10 rows; 10 * 40 / 100 = 4 cols
        mask[10:, :-4] = 1
        self.assertTrue(np.all(chip_out[mask] == 1))
        self.assertTrue(np.all(chip_out[~mask] == 0))

        # overflow that is not a whole number of chip pixels: a chip pixel is
        # filled if its center lies outside the bbox
        bbox = Box(0, 0, 100, 100)
        window = Box(-15, -25, 85, 75)
        chip_in = np.ones((30, 30, 1), dtype=np.uint8)
        chip_out = fill_overflow(bbox, window, chip_in)
        # top: 15 * 0.3 = 4.5 -> 4 rows; left: 25 * 0.3 = 7.5 -> 7 cols
        mask = np.zeros((30, 30), dtype=bool)
        mask[4:, 7:] = 1
        self.assertTrue(np.all(chip_out[mask] == 1))
        self.assertTrue(np.all(chip_out[~mask] == 0))

        # window completely outside the bbox
        window = Box(150, 0, 250, 100)
        chip_in = np.ones((50, 50, 1), dtype=np.uint8)
        chip_out = fill_overflow(bbox, window, chip_in)
        self.assertTrue(np.all(chip_out == 0))

    def test_pad_to_window_size(self):
        bbox = Box(10, 10, 90, 90)
        window = Box(0, 0, 100, 100)

        # ndim == 2
        chip_in = np.ones((80, 80), dtype=np.uint8)
        chip_out = pad_to_window_size(chip_in, window, bbox, fill_value=0)
        self.assertEqual(chip_out.shape, (100, 100))
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:90, 10:90] = 1
        self.assertTrue(np.all(chip_out[mask] == 1))
        self.assertTrue(np.all(chip_out[~mask] == 0))

        # ndim > 2
        chip_in = np.ones((2, 80, 80, 4), dtype=np.uint8)
        chip_out = pad_to_window_size(chip_in, window, bbox, fill_value=0)
        self.assertEqual(chip_out.shape, (2, 100, 100, 4))
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:90, 10:90] = 1
        self.assertTrue(np.all(chip_out[:, mask] == 1))
        self.assertTrue(np.all(chip_out[:, ~mask] == 0))


if __name__ == '__main__':
    unittest.main()
