from typing import TYPE_CHECKING, Iterable, Optional, Tuple
from os.path import join

from rastervision.pipeline.config import register_config, Config, Field

if TYPE_CHECKING:
    from rastervision.core.data import ClassConfig


@register_config('evaluator')
class EvaluatorConfig(Config):
    output_uri: Optional[str] = Field(
        None,
        description='URI of directory where evaluator output will be saved. '
        'Evaluations for each scene-group will be save in a JSON file at '
        '<output_uri>/<scene-group-name>/eval.json. If None, and this Config '
        'is part of an RVPipeline, this field will be auto-generated.')

    def build(self, class_config: 'ClassConfig',
              scene_group: Tuple[str, Iterable[str]]) -> 'EvaluatorConfig':
        pass

    def get_output_uri(self, scene_group_name: Optional[str] = None) -> str:
        if scene_group_name is None:
            return join(self.output_uri, 'eval.json')
        return join(self.output_uri, scene_group_name, 'eval.json')

    def update(self, pipeline=None):
        if pipeline is not None and self.output_uri is None:
            self.output_uri = pipeline.eval_uri
