from rastervision.core.data.raster_transformer.cast_transformer import (
    CastTransformer,
)
from rastervision.core.data.raster_transformer.raster_transformer_config import (  # noqa
    RasterTransformerConfig,
)
from rastervision.pipeline.config import Field, register_config


@register_config('cast_transformer')
class CastTransformerConfig(RasterTransformerConfig):
    """Configure a :class:`.CastTransformer`."""

    to_dtype: str = Field(
        ...,
        description='dtype to cast raster to. Must be a valid Numpy dtype '
        'e.g. "uint8", "float32", etc.',
    )

    def build(self, channel_order: list[int] | None = None):
        return CastTransformer(to_dtype=self.to_dtype)
