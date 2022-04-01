from typing import TYPE_CHECKING, Iterable, Tuple

from rastervision.pipeline.config import register_config
from rastervision.core.evaluation.classification_evaluator_config import (
    ClassificationEvaluatorConfig)
from rastervision.core.evaluation.object_detection_evaluator import (
    ObjectDetectionEvaluator)

if TYPE_CHECKING:
    from rastervision.core.data import ClassConfig


@register_config('object_detection_evaluator')
class ObjectDetectionEvaluatorConfig(ClassificationEvaluatorConfig):
    def build(self, class_config: 'ClassConfig',
              scene_group: Tuple[str, Iterable[str]]
              ) -> ObjectDetectionEvaluator:
        group_name, _ = scene_group
        output_uri = self.get_output_uri(group_name)
        return ObjectDetectionEvaluator(class_config, output_uri)
