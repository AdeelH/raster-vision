from typing import TYPE_CHECKING

from rastervision.pipeline.pipeline import Pipeline
from rastervision.pipeline.pipeline_config import PipelineConfig
from rastervision.pytorch_learner import LearnerConfig

if TYPE_CHECKING:
    from rastervision.core.data import Labels, Scene
    from rastervision.pytorch_learner import LearnerPipelineConfig


class LearnerPipeline(Pipeline):
    """Simple Pipeline that is a wrapper around Learner.main()

    This supports the ability to use the pytorch_learner package to train models using
    the RV pipeline package and its runner functionality without the rest of RV.
    """
    commands = ['train']
    gpu_commands = ['train']

    def __init__(self, config: PipelineConfig, tmp_dir: str):
        super().__init__(config, tmp_dir)
        self.config: 'LearnerPipelineConfig'

    def train(self):
        learner_cfg: LearnerConfig = self.config.learner
        learner = learner_cfg.build(learner_cfg, self.tmp_dir)
        learner.main()

    def predict_scene(self, scene: 'Scene') -> 'Labels':
        raise NotImplementedError()
