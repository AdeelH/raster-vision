# flake8: noqa
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rastervision.pipeline import Registry


def register_plugin(registry: 'Registry'):
    registry.set_plugin_version('rastervision.pytorch_learner', 7)
    registry.register_renamed_type_hints('geo_data_window', 'window_sampling')


import rastervision.pipeline
from rastervision.pytorch_learner.learner_config import *
from rastervision.pytorch_learner.learner import *
from rastervision.pytorch_learner.classification_learner_config import *
from rastervision.pytorch_learner.classification_learner import *
from rastervision.pytorch_learner.regression_learner_config import *
from rastervision.pytorch_learner.regression_learner import *
from rastervision.pytorch_learner.semantic_segmentation_learner_config import *
from rastervision.pytorch_learner.semantic_segmentation_learner import *
from rastervision.pytorch_learner.object_detection_learner_config import *
from rastervision.pytorch_learner.object_detection_learner import *
from rastervision.pytorch_learner.dataset import *

__all__ = [
    # Learner
    'Learner',
    'SemanticSegmentationLearner',
    'ClassificationLearner',
    'ObjectDetectionLearner',
    'RegressionLearner',
    # LearnerConfig
    'LearnerConfig',
    'SemanticSegmentationLearnerConfig',
    'ClassificationLearnerConfig',
    'ObjectDetectionLearnerConfig',
    'RegressionLearnerConfig',
    # DataConfig
    'DataConfig',
    'GeoDataConfig',
    'PlotOptions',
    'ImageDataConfig',
    'SemanticSegmentationDataConfig',
    'SemanticSegmentationGeoDataConfig',
    'SemanticSegmentationImageDataConfig',
    'ClassificationDataConfig',
    'ClassificationGeoDataConfig',
    'ClassificationImageDataConfig',
    'ObjectDetectionDataConfig',
    'ObjectDetectionGeoDataConfig',
    'ObjectDetectionImageDataConfig',
    'RegressionDataConfig',
    'RegressionGeoDataConfig',
    'RegressionImageDataConfig',
    'PlotOptions',
    # DataFormat
    'SemanticSegmentationDataFormat',
    'ClassificationDataFormat',
    'ObjectDetectionDataFormat',
    'RegressionDataFormat',
    # ModelConfig
    'ModelConfig',
    'ExternalModuleConfig',
    'Backbone',
    'SemanticSegmentationModelConfig',
    'ClassificationModelConfig',
    'ObjectDetectionModelConfig',
    'RegressionModelConfig',
    # SolverConfig
    'SolverConfig',
    # Dataset
    'AlbumentationsDataset',
    'GeoDataset',
    'ImageDataset',
    'SlidingWindowGeoDataset',
    'RandomWindowGeoDataset',
    'SemanticSegmentationSlidingWindowGeoDataset',
    'SemanticSegmentationRandomWindowGeoDataset',
    'SemanticSegmentationImageDataset',
    'SemanticSegmentationDataReader',
    'ClassificationSlidingWindowGeoDataset',
    'ClassificationRandomWindowGeoDataset',
    'ClassificationImageDataset',
    'ObjectDetectionSlidingWindowGeoDataset',
    'ObjectDetectionRandomWindowGeoDataset',
    'ObjectDetectionImageDataset',
    'CocoDataset',
    'RegressionSlidingWindowGeoDataset',
    'RegressionRandomWindowGeoDataset',
    'RegressionImageDataset',
    'RegressionDataReader',
    'TransformType',
    'DatasetError',
    'ImageDatasetError',
    'GeoDatasetError',
    # Visualizer
    'Visualizer',
    'SemanticSegmentationVisualizer',
    'ClassificationVisualizer',
    'ObjectDetectionVisualizer',
    'RegressionVisualizer',
]
