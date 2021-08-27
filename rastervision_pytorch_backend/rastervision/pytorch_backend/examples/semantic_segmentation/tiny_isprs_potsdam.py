# flake8: noqa

from rastervision.pipeline.runner import *
from rastervision.core.rv_pipeline import *
from rastervision.core.data import *
from rastervision.pytorch_backend import *
from rastervision.pytorch_learner import *


def get_config(runner: Runner, root_uri: str) -> SemanticSegmentationConfig:
    # specify class names and corresponding colors in the label rasters
    class_config = ClassConfig(
        names=[
            'Car', 'Building', 'Low Vegetation', 'Tree', 'Impervious',
            'Clutter'
        ],
        colors=[
            '#ffff00', '#0000ff', '#00ffff', '#00ff00', '#ffffff', '#ff0000'
        ])

    # configure the training scene
    train_scene = SceneConfig(
        id='2_10',
        # configure the imagery raster data
        raster_source=RasterioSourceConfig(
            uris=['img/2_10_RGBIR.tif'], channel_order=[0, 1, 2, 3]),
        # configure the ground truth label source - here, the labels are
        # specified as RGB images with colors identifying the class for each
        # pixel
        label_source=SemanticSegmentationLabelSourceConfig(
            rgb_class_config=class_config,
            raster_source=RasterioSourceConfig(uris=['label/2_10_label.tif'])))

    # configure the validation scene
    val_scene = SceneConfig(
        id='6_12',
        raster_source=RasterioSourceConfig(
            uris=['img/6_12_RGBIR.tif'], channel_order=[0, 1, 2, 3]),
        label_source=SemanticSegmentationLabelSourceConfig(
            rgb_class_config=class_config,
            raster_source=RasterioSourceConfig(uris=['label/6_12_label.tif'])),
        # configure how the predicted labels will be outputted - here, we want
        # our output labels to be RGB rasters just like the ground truth
        # labels, so we set rgb=True
        label_store=SemanticSegmentationLabelStoreConfig(rgb=True))

    # the training/validation split - we'll skip the test set in this example
    dataset_config = DatasetConfig(
        class_config=class_config,
        train_scenes=[train_scene],
        validation_scenes=[val_scene])

    # configure how the data will be read
    data_config = SemanticSegmentationGeoDataConfig(
        scene_dataset=dataset_config,
        # read 300x300 chips from the scene using a sliding window with a
        # stride of 300 pixels
        # Tip: you can also specify different window opts for each scene!
        window_opts=GeoDataWindowConfig(
            method=GeoDataWindowMethod.sliding, size=300, stride=300),
        channel_display_groups={
            'RGB': [0, 1, 2],
            'IR': [3]
        })

    # configure the pytorch backend
    backend_config = PyTorchSemanticSegmentationConfig(
        data=data_config,
        # configure the model architecture - by default, RV uses DeepLabV3
        # for semantic segmentation
        model=SemanticSegmentationModelConfig(backbone=Backbone.resnet50),
        # configure training params
        solver=SolverConfig(lr=1e-4, num_epochs=3, batch_sz=8, one_cycle=True),
        # save but not run tensorboard log
        log_tensorboard=True,
        run_tensorboard=False)

    # bring it all together
    pipeline_config = SemanticSegmentationConfig(
        # the output directory
        root_uri=root_uri,
        dataset=dataset_config,
        backend=backend_config,
        # the size of chips to use for prediction
        predict_chip_sz=300)

    return pipeline_config


# # Two ways of reading chips
# data_config = SemanticSegmentationGeoDataConfig(
#     scene_dataset=dataset_config,
#     # read 100 randomly located chips of size 300x300 from the scene
#     window_opts=GeoDataWindowConfig(
#         method=GeoDataWindowMethod.random,
#         size=300,
#         size_lims=(200, 400),
#         max_windows=100))

# # Handling AOIs
# train_scene = SceneConfig(
#     id='2_10',
#     raster_source=RasterioSourceConfig(uris=['2_10_RGBIR.tif']),
#     label_source=SemanticSegmentationLabelSourceConfig(
#         rgb_class_config=class_config,
#         raster_source=RasterioSourceConfig(uris=['2_10_label.tif'])),
#     aoi_uris=['2_10_aoi.json'])

# # data augmentation
# aug_transform = A.Compose([
#     A.Flip(),
#     A.ShiftScaleRotate(),
#     A.OneOf([
#         A.RGBShift(),
#         A.ToGray(),
#         A.ToSepia(),
#     ]),
#     A.OneOf([
#         A.RandomBrightness(),
#         A.RandomGamma(),
#     ]),
#     A.OneOf([
#         A.Blur(),
#         A.Downscale(),
#         A.GridDistortion(),
#     ]),
#     A.CoarseDropout(max_height=32, max_width=32, max_holes=5)
# ])

# data_config = SemanticSegmentationGeoDataConfig(
#     scene_dataset=dataset_config,
#     window_opts=GeoDataWindowConfig(
#         method=GeoDataWindowMethod.sliding, size=300, stride=300),
#     aug_transform=A.to_dict(aug_transform))

# # external model, loss
# backend_config = PyTorchSemanticSegmentationConfig(
#     data=data_config,
#     model=SemanticSegmentationModelConfig(
#         external_def=ExternalModuleConfig(
#             github_repo='AdeelH/pytorch-fpn:0.2',
#             name='fpn',
#             entrypoint='make_fpn_resnet',
#             entrypoint_kwargs={
#                 'name': 'resnet18',
#                 'fpn_type': 'panoptic',
#                 'num_classes': 7,
#                 'fpn_channels': 128,
#                 'out_size': (300, 300)
#             })),
#     solver=SolverConfig(
#         lr=1e-4,
#         num_epochs=3,
#         batch_sz=8,
#         one_cycle=True,
#         external_loss_def=ExternalModuleConfig(
#             github_repo='AdeelH/pytorch-multi-class-focal-loss:1.0',
#             name='focal_loss',
#             entrypoint='focal_loss',
#             force_reload=False,
#             entrypoint_kwargs={
#                 'alpha': [.75, .25],
#                 'gamma': 2
#             })))

# # multiband
# train_scene = SceneConfig(
#     id='2_10',
#     raster_source=RasterioSourceConfig(
#         uris=['2_10_RGBIR.tif'], channel_order=[0, 1, 2, 3]),
#     label_source=SemanticSegmentationLabelSourceConfig(
#         rgb_class_config=class_config,
#         raster_source=RasterioSourceConfig(uris=['2_10_label.tif'])))
