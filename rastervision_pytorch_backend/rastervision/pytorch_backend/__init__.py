def register_plugin(registry):
    registry.set_plugin_version('rastervision.pytorch_backend', 2)


import rastervision.pipeline
from rastervision.pytorch_backend.pytorch_chip_classification import *
from rastervision.pytorch_backend.pytorch_chip_classification_config import *
from rastervision.pytorch_backend.pytorch_object_detection import *
from rastervision.pytorch_backend.pytorch_object_detection_config import *
from rastervision.pytorch_backend.pytorch_semantic_segmentation import *
from rastervision.pytorch_backend.pytorch_semantic_segmentation_config import *
from rastervision.pytorch_backend.utils import *

__all__ = [
    'PyTorchChipClassification',
    'PyTorchChipClassificationConfig',
    'PyTorchSemanticSegmentation',
    'PyTorchSemanticSegmentationConfig',
    'PyTorchObjectDetection',
    'PyTorchObjectDetectionConfig',
]
