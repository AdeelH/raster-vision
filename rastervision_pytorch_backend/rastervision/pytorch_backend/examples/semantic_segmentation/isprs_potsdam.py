# flake8: noqa

import os
from os.path import join

from rastervision.core.rv_pipeline import *
from rastervision.core.backend import *
from rastervision.core.data import *
from rastervision.core.analyzer import *
from rastervision.pytorch_backend import *
from rastervision.pytorch_learner import *
from rastervision.pytorch_backend.examples.utils import get_scene_info, save_image_crop


def get_config(runner,
               raw_uri,
               processed_uri,
               root_uri,
               analyze_uri=None,
               chip_uri=None,
               bundle_uri=None,
               multiband=False,
               augment=False,
               external_model=False,
               test=False):
    train_ids = [
        '2-10', '2-11', '3-10', '3-11', '4-10', '4-11', '4-12', '5-10', '5-11',
        '5-12', '6-10', '6-11', '6-7', '6-9', '7-10', '7-11', '7-12', '7-7',
        '7-8', '7-9'
    ]
    val_ids = ['2-12', '3-12', '6-12']

    if test:
        train_ids = train_ids[0:2]
        val_ids = val_ids[0:2]

    class_config = ClassConfig(
        names=[
            'Car', 'Building', 'Low Vegetation', 'Tree', 'Impervious',
            'Clutter'
        ],
        colors=[
            '#ffff00', '#0000ff', '#00ffff', '#00ff00', '#ffffff', '#ff0000'
        ])

    if multiband:
        # use all 4 channels
        channel_order = [0, 1, 2, 3]
    else:
        # use infrared, red, & green channels only
        channel_order = [3, 0, 1]

    def make_scene(id):
        id = id.replace('-', '_')
        raster_uri = '{}/4_Ortho_RGBIR/top_potsdam_{}_RGBIR.tif'.format(
            raw_uri, id)
        label_uri = '{}/5_Labels_for_participants/top_potsdam_{}_label.tif'.format(
            raw_uri, id)

        if test:
            crop_uri = join(processed_uri, 'crops',
                            os.path.basename(raster_uri))
            label_crop_uri = join(processed_uri, 'crops',
                                  os.path.basename(label_uri))
            save_image_crop(
                raster_uri,
                crop_uri,
                label_uri=label_uri,
                label_crop_uri=label_crop_uri,
                size=600,
                vector_labels=False)
            raster_uri = crop_uri
            label_uri = label_crop_uri

        raster_source = RasterioSourceConfig(
            uris=[raster_uri], channel_order=channel_order)

        # Using with_rgb_class_map because label TIFFs have classes encoded as
        # RGB colors.
        label_source = SemanticSegmentationLabelSourceConfig(
            rgb_class_config=class_config,
            raster_source=RasterioSourceConfig(uris=[label_uri]))

        # URI will be injected by scene config.
        # Using rgb=True because we want prediction TIFFs to be in
        # RGB format.
        label_store = SemanticSegmentationLabelStoreConfig(
            rgb=True, vector_output=[PolygonVectorOutputConfig(class_id=0)])

        scene = SceneConfig(
            id=id,
            raster_source=raster_source,
            label_source=label_source,
            label_store=label_store)

        return scene

    dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=[make_scene(id) for id in train_ids],
        validation_scenes=[make_scene(id) for id in val_ids])
    chip_sz = 300
    chip_options = SemanticSegmentationChipOptions(
        window_method=SemanticSegmentationWindowMethod.sliding, stride=chip_sz)

    img_sz = 256
    batch_sz = 8
    if external_model:
        model = SemanticSegmentationModelConfig(
            external_def=ExternalModuleConfig(
                # github_repo='AdeelH/pytorch-efficientnet-deeplabv3',
                # github_repo='AdeelH/pytorch-fpn',
                # uri='/home/adeel/pytorch-efficientnet-deeplabv3',
                uri='/home/adeel/pytorch-fpn',
                force_reload=True,
                name='pytorch-fpn',
                # entrypoint='make_segmentation_model',
                # entrypoint='make_segm_fpn_efficientnet',
                entrypoint='make_segm_fpn_resnet',
                entrypoint_kwargs={
                    'name': 'resnet18',
                    # 'backbone_name': 'efficientnet_b4',
                    # 'name': 'resnet50',
                    'fpn_type': 'panoptic',
                    'num_classes': len(class_config.names) + 1,
                    # 'pretrained': 'imagenet',
                    # 'pretrained': False,
                    'fpn_channels': 256,
                    'out_size': (img_sz, img_sz)
                }))
    else:
        model = SemanticSegmentationModelConfig(backbone=Backbone.resnet50)

    if augment:
        aug_transform = A.Compose([
            A.Flip(),
            A.Transpose(),
            A.RandomRotate90(),
            A.ShiftScaleRotate(),
            A.CLAHE(p=.2),
            A.FancyPCA(p=.2),
            A.ColorJitter(p=.5),
            A.OneOf([
                A.GaussNoise(),
                A.ISONoise(),
                A.RandomFog(),
                A.Blur(),
                A.MotionBlur(),
                A.ImageCompression(),
                A.Downscale(),
            ]),
            A.CoarseDropout(max_height=8, max_width=8, max_holes=5)
        ])

        aug_transform = A.to_dict(aug_transform)
        base_transform = None
        plot_transform = None
    else:
        aug_transform = None
        base_transform = None
        plot_transform = None

    backend = PyTorchSemanticSegmentationConfig(
        model=model,
        solver=SolverConfig(
            lr=1e-14,
            num_epochs=1,
            # test_num_epochs=1,
            batch_sz=2,
            # test_batch_sz=2,
            one_cycle=True),
        log_tensorboard=True,
        run_tensorboard=False,
        test_mode=test,
        img_sz=img_sz,
        base_transform=base_transform,
        aug_transform=aug_transform,
        plot_options=PlotOptions(transform=plot_transform),
        group_uris=[
            'rvexp/data/ss/output/chip/c855e319-134f-4d9c-85e2-93923a7470f0.zip',
            'rvexp/data/ss/output/chip/fedd08d8-0098-40f1-9fb1-915d7ef43d3a.zip'
        ],
        group_train_sz_rel=[.25, .5])

    if multiband:
        channel_display_groups = {'RGB': (0, 1, 2), 'IR': (3, )}
    else:
        channel_display_groups = None

    return SemanticSegmentationConfig(
        root_uri=root_uri,
        analyze_uri=analyze_uri,
        chip_uri=chip_uri,
        dataset=dataset,
        backend=backend,
        channel_display_groups=channel_display_groups,
        train_chip_sz=chip_sz,
        predict_chip_sz=chip_sz,
        chip_options=chip_options,
        chip_nodata_threshold=.5)
