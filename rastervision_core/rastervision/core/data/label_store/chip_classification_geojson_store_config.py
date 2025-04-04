from os.path import join

from rastervision.core.data.label_store import (
    ChipClassificationGeoJSONStore,
    LabelStoreConfig,
)
from rastervision.pipeline.config import Field, register_config


@register_config('chip_classification_geojson_store')
class ChipClassificationGeoJSONStoreConfig(LabelStoreConfig):
    """Configure a :class:`.ChipClassificationGeoJSONStore`."""

    uri: str | None = Field(
        None,
        description=(
            'URI of GeoJSON file with predictions. If None, and this Config is part of '
            'a SceneConfig inside an RVPipelineConfig, it will be auto-generated.'
        ),
    )

    def build(self, class_config, crs_transformer, bbox=None, tmp_dir=None):
        return ChipClassificationGeoJSONStore(
            self.uri, class_config, crs_transformer, bbox=bbox
        )

    def update(self, pipeline=None, scene=None):
        if self.uri is None and pipeline is not None and scene is not None:
            self.uri = join(pipeline.predict_uri, f'{scene.id}.json')
