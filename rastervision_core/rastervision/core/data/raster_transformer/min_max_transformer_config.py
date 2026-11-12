from rastervision.core.data.raster_transformer import (
    MinMaxTransformer,
    RasterTransformerConfig,
)
from rastervision.pipeline.config import register_config


@register_config('min_max_transformer')
class MinMaxTransformerConfig(RasterTransformerConfig):
    """Configure a :class:`.MinMaxTransformer`."""

    def build(self, channel_order: list[int] | None = None):
        return MinMaxTransformer()
