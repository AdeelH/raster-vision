import albumentations as A

from rastervision.core.rv_pipeline import ChipClassificationConfig
from rastervision.core.data import ClassConfig, DatasetConfig
from rastervision.pytorch_backend import PyTorchChipClassificationConfig
from rastervision.pytorch_learner import (
    ClassificationImageDataConfig, ClassificationModelConfig, SolverConfig)

CLASS_NAMES = [
    'AnnualCrop',
    'HerbaceousVegetation',
    'Industrial',
    'PermanentCrop',
    'River',
    'Forest',
    'Highway',
    'Pasture',
    'Residential',
    'SeaLake',
]


def get_config(runner, root_uri: str,
               chip_uri: str) -> ChipClassificationConfig:
    class_config = ClassConfig(names=CLASS_NAMES)
    img_sz = 256
    aug_tf = A.Compose([
        A.Flip(),
        A.ShiftScaleRotate(),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=10),
            A.RandomBrightness(),
            A.RandomGamma(),
        ]),
        A.CoarseDropout(max_height=8, max_width=8, max_holes=5),
    ])
    data_cfg = ClassificationImageDataConfig(
        class_names=class_config.names,
        class_colors=class_config.colors,
        img_channels=3,
        img_sz=img_sz,
        aug_transform=A.to_dict(aug_tf),
        num_workers=6,
    )
    model_cfg = ClassificationModelConfig(backbone='resnet18', pretrained=True)
    solver_cfg = SolverConfig(
        batch_sz=32,
        lr=3e-4,
        num_epochs=5,
    )
    backend = PyTorchChipClassificationConfig(
        data=data_cfg, model=model_cfg, solver=solver_cfg)
    pipeline = ChipClassificationConfig(
        root_uri=root_uri,
        chip_uri=chip_uri,
        dataset=DatasetConfig(
            class_config=class_config, train_scenes=[], validation_scenes=[]),
        backend=backend,
        predict_chip_sz=64,
    )
    return pipeline
