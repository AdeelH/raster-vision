from typing import Optional, Sequence

import numpy as np

from rastervision.core.data.raster_transformer.raster_transformer \
    import RasterTransformer


class ChannelTransformer(RasterTransformer):
    """Compute new image channels as a ratio of linear combinations of existing
    channels. Useful for creating bands like NDVI, EVI, etc.

    For example, numer_coeffs = [(-1, 0, 0, 1)] and
    denom_coeffs = [(1, 0, 0, 1)] will transform an (R, G, B, NIR) image to an
    NDVI image.
    """

    def __init__(self,
                 numer_coeffs: Sequence[Sequence[float]],
                 denom_coeffs: Optional[Sequence[Sequence[float]]] = None):
        """Construct a new ChannelTransformer.

        Args:
            numer_coeffs (Sequence[Sequence[float]]): Coefficients to use for
                computing the numerator linear combination.
            denom_coeffs (Sequence[Sequence[float]], optional): Coefficients to
                use for computing the denominator linear combination. If not
                provided, no division takes place and the numerator linear
                combination becomes the output. Defaults to None.
        """
        if denom_coeffs is not None and len(numer_coeffs) != len(denom_coeffs):
            raise ValueError('len(numer_coeffs) != len(denom_coeffs)')

        self.out_channels = len(numer_coeffs)
        self.numer_coeffs = [
            np.array(c)[np.newaxis, np.newaxis, :] for c in numer_coeffs
        ]
        if denom_coeffs is None:
            self.denom_coeffs = [None] * len(numer_coeffs)
        else:
            self.denom_coeffs = [
                np.array(c)[np.newaxis, np.newaxis, :] for c in denom_coeffs
            ]

    def transform(self,
                  chip: np.ndarray,
                  channel_order: Optional[Sequence[int]] = None):
        """Transform a chip.

        Args:
            chip (np.ndarray): uint8 array of shape [height, width, channels].

        Returns:
            [height, width, channels] numpy array

        """
        chip = chip.astype(np.float32) / 255

        h, w = chip.shape[:2]
        out_chip = np.empty((h, w, self.out_channels), dtype=np.float32)
        numer_coeffs, denom_coeffs = self.numer_coeffs, self.denom_coeffs
        for i, (nc, dc) in enumerate(zip(numer_coeffs, denom_coeffs)):
            numer = (nc * chip).sum(axis=-1)
            denom = (dc * chip).sum(axis=-1) if dc is not None else 1.
            out_slice = numer / (denom + 1e-6)
            # normalize to [0, 1]
            out_slice = (out_slice - out_slice.min()) / (
                out_slice.max() - out_slice.min())
            out_chip[..., i] = out_slice

        out_chip *= 255
        out_chip = out_chip.astype(np.uint8)
        return out_chip
