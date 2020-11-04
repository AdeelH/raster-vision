from typing import Optional

from rastervision.pipeline.config import register_config, Field
from rastervision.core.data.raster_transformer.raster_transformer_config import (  # noqa
    RasterTransformerConfig)
from rastervision.core.data.raster_transformer.cast_transformer import (  # noqa
    CastTransformer)


@register_config('cast_transformer')
class CastTransformerConfig(RasterTransformerConfig):
    to_dtype: Optional[str] = Field(
        'uint8', description='dtype to cast raster to.')

    def build(self):
        return CastTransformer(to_dtype=self.to_dtype)
