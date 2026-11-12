from pydantic import Field

from rastervision.core.data.raster_transformer.nan_transformer import (  # noqa
    NanTransformer,
)
from rastervision.core.data.raster_transformer.raster_transformer_config import (  # noqa
    RasterTransformerConfig,
)
from rastervision.pipeline.config import register_config


@register_config('nan_transformer')
class NanTransformerConfig(RasterTransformerConfig):
    """Configure a :class:`.NanTransformer`."""

    to_value: float = Field(
        default=0.0, description=('Turn all NaN values into this value.')
    )

    def build(self, channel_order: list[int] | None = None):
        return NanTransformer(to_value=self.to_value)
