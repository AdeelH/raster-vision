from rastervision.core.evaluation.evaluator_config import EvaluatorConfig
from rastervision.pipeline.config import register_config


@register_config('classification_evaluator')
class ClassificationEvaluatorConfig(EvaluatorConfig):
    """Configure a :class:`.ClassificationEvaluator`."""
