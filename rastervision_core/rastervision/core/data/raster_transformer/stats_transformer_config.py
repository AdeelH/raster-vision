from typing import TYPE_CHECKING, Optional
from os.path import join, basename

from rastervision.pipeline.config import register_config, Field
from rastervision.core.data.raster_transformer import (RasterTransformerConfig,
                                                       StatsTransformer)
from rastervision.core.raster_stats import RasterStats

if TYPE_CHECKING:
    from rastervision.core.rv_pipeline import RVPipelineConfig
    from rastervision.core.data import SceneConfig


@register_config('stats_transformer')
class StatsTransformerConfig(RasterTransformerConfig):
    stats_uri: Optional[str] = Field(
        None,
        description='URI of directory where stats will be saved. '
        'Stats for the scene-group will be save in a JSON file at '
        '<stats_uri>/<scene-group-name>/stats.json. If None, and this Config '
        'is part of an RVPipeline, this field will be auto-generated.')
    scene_group: str = Field(
        'train_scenes',
        description='Name of the group of scenes whose stats to use.')

    def update(self,
               pipeline: Optional['RVPipelineConfig'] = None,
               scene: Optional['SceneConfig'] = None) -> None:
        if pipeline is not None and self.stats_uri is None:
            self.stats_uri = join(pipeline.analyze_uri, 'stats',
                                  self.scene_group, 'stats.json')

    def build(self):
        return StatsTransformer(RasterStats.load(self.stats_uri))

    def update_root(self, root_dir):
        self.stats_uri = join(root_dir, basename(self.stats_uri))
