import unittest

import numpy as np

from rastervision.core.data.raster_transformer import ChannelTransformerConfig


class TestChannelRasterTransformer(unittest.TestCase):
    def test_channel_transformer(self):
        # without denom
        cfg = ChannelTransformerConfig(numer_coeffs=[(1, 1), (-1, 1)])
        tf = cfg.build()
        in_chip = np.empty((10, 10, 2))
        in_chip[:, :5, 0] = 1
        in_chip[:, 5:, 0] = 2
        in_chip[:, :5, 1] = 1
        in_chip[:, 5:, 1] = 2
        in_chip = in_chip.astype(np.uint8)
        print(in_chip[..., 0])
        print(in_chip[..., 1])

        out_chip = tf.transform(in_chip)
        print()
        print(out_chip[..., 0])
        print(out_chip[..., 1])
        self.assertEqual(out_chip.shape, (10, 10, 2))
        self.assertEqual(out_chip.dtype, np.uint8)
        self.assertTrue(np.all(out_chip[:, :5, 0] == 0))
        self.assertTrue(np.all(out_chip[:, 5:, 0] == 255))
        self.assertTrue(np.all(out_chip[:, :5, 1] == 0))
        self.assertTrue(np.all(out_chip[:, 5:, 1] == 0))


if __name__ == '__main__':
    unittest.main()
