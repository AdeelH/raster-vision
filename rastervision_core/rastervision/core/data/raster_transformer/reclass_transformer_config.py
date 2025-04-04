from rastervision.core.data.raster_transformer import (
    RasterTransformerConfig,
    ReclassTransformer,
)
from rastervision.pipeline.config import Field, register_config


@register_config('reclass_transformer')
class ReclassTransformerConfig(RasterTransformerConfig):
    """Configure a :class:`.ReclassTransformer`."""

    mapping: dict[int, int] = Field(
        ..., description=('The reclassification mapping.')
    )

    def build(
        self, channel_order: list[int] | None = None
    ) -> ReclassTransformer:
        return ReclassTransformer(mapping=self.mapping)
