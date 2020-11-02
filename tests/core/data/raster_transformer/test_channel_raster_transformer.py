import unittest

import numpy as np

from rastervision.core.data.raster_transformer import ChannelTransformerConfig


class TestChannelRasterTransformer(unittest.TestCase):
    def test_channel_transformer(self):
        # without denom
        cfg = ChannelTransformerConfig(numer_coeffs=[(1, 1, 1), (1, 1, -1)])
        tf = cfg.build()
        in_chip = np.array([1, 2, 3])[np.newaxis, np.newaxis, :] * np.ones(
            (100, 100, 3))
        in_chip = in_chip.astype(np.uint8)

        out_chip = tf.transform(in_chip)
        self.assertEqual(out_chip.shape, (100, 100, 2))
        self.assertEqual(out_chip.dtype, np.uint8)
        self.assertTrue(np.all(out_chip[..., 0] == 6))
        self.assertTrue(np.all(out_chip[..., 1] == 0))

        # with denom
        cfg = ChannelTransformerConfig(
            numer_coeffs=[(1, 1, 1)], denom_coeffs=[(1, 1, -1)])
        tf = cfg.build()
        out_chip = tf.transform(in_chip)
        self.assertEqual(out_chip.shape, (100, 100, 1))
        self.assertEqual(out_chip.dtype, np.uint8)
        self.assertTrue(np.all(out_chip[..., 0] == 0))


if __name__ == '__main__':
    unittest.main()
