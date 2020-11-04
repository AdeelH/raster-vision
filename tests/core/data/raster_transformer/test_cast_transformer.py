import unittest

import numpy as np

from rastervision.core.data.raster_transformer import CastTransformerConfig


class TestCastRasterTransformer(unittest.TestCase):
    def test_cast_transformer(self):
        chip = np.empty((10, 10, 3), dtype=np.float32)
        tf = CastTransformerConfig(to_dtype='uint8').build()
        self.assertEqual(tf(chip).dtype, np.uint8)
        self.assertEqual(str(tf), 'CastTransformer(to_dtype=uint8)')

        chip = np.empty((10, 10, 3), dtype=np.uint16)
        tf = CastTransformerConfig(to_dtype='float32').build()
        self.assertEqual(tf(chip).dtype, np.float32)
        self.assertEqual(str(tf), 'CastTransformer(to_dtype=float32)')


if __name__ == '__main__':
    unittest.main()
