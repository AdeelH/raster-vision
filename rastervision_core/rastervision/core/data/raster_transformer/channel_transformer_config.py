from typing import Optional, Sequence

from rastervision.pipeline.config import register_config, Field
from rastervision.core.data.raster_transformer import (RasterTransformerConfig,
                                                       ChannelTransformer)


@register_config('channel_transformer')
class ChannelTransformerConfig(RasterTransformerConfig):
    """Compute new image channels as a ratio of linear combinations of existing
    channels. Useful for creating bands like NDVI, EVI, etc.

    For example, numer_coeffs = [(-1, 0, 0, 1)] and
    denom_coeffs = [(1, 0, 0, 1)] will transform an (R, G, B, NIR) image to an
    NDVI image.
    """
    numer_coeffs: Sequence[Sequence[float]] = Field(
        ...,
        description='Coefficients to use for computing the numerator linear '
        'combination.')
    denom_coeffs: Optional[Sequence[Sequence[float]]] = Field(
        None,
        description='Coefficients to use for computing the denominator linear '
        'combination. If not provided, no division takes place and the '
        'numerator linear combination becomes the output. Defaults to None.')

    def build(self):
        return ChannelTransformer(
            numer_coeffs=self.numer_coeffs, denom_coeffs=self.denom_coeffs)
