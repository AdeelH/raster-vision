# flake8: noqa

import os
from os.path import join

from rastervision.core.rv_pipeline import *
from rastervision.core.backend import *
from rastervision.core.data import *
from rastervision.core.analyzer import *
from rastervision.pytorch_backend import *
from rastervision.pytorch_learner import *
from rastervision.pytorch_backend.examples.utils import (get_scene_info,
                                                         save_image_crop)

TRAIN_IDS = ['2_10']
UNLABELED_IDS = [
    '2_11', '2_12', '2_14', '3_11', '3_13', '4_10', '5_10', '6_7', '6_9'
]
VAL_IDS = ['2_13', '6_8', '3_10']


def get_config(runner, raw_uri: str, processed_uri: str, root_uri: str,
               **kwargs: dict) -> ChipClassificationConfig:
    """Generate the pipeline config for this task. This function will be called
    by RV, with arguments from the command line, when this example is run.

    Args:
        runner (Runner): Runner for the pipeline. Will be provided by RV.
        raw_uri (str): Directory where the raw data resides
        processed_uri (str): Directory for storing processed data. 
                             E.g. crops for testing.
        root_uri (str): Directory where all the output will be written.

    Returns:
        ChipClassificationConfig: A pipeline config.
    """
    # extract params
    epochs = int(kwargs.get('epochs', 2))
    lr = float(kwargs.get('lr', 1e-4))
    batch_size = int(kwargs.get('batch_size', 32))
    one_cycle = kwargs.get('one_cycle', True)
    external_loss = kwargs.get('external_loss', False)

    num_workers = int(kwargs.get('batch_size', 4))

    train_ids = kwargs.get('train_ids', TRAIN_IDS)
    if isinstance(train_ids, str):
        train_ids = train_ids.split(',')
    val_ids = kwargs.get('val_ids', TRAIN_IDS)
    if isinstance(val_ids, str):
        val_ids = val_ids.split(',')

    aoi_uris = kwargs.get('aoi_uris', '').split(',')

    chip_sz = int(kwargs.get('batch_size', 300))
    img_sz = chip_sz

    def make_scene(id: str, **kwargs) -> SceneConfig:
        raster_uri = join(raw_uri, f'4_Ortho_RGBIR/top_potsdam_{id}_RGBIR.tif')
        label_uri = join(processed_uri, 'labels', 'all',
                         f'top_potsdam_{id}_RGBIR.json')

        raster_source = RasterioSourceConfig(
            uris=[raster_uri],
            channel_order=[0, 1, 2],
            extent_crop=kwargs.get('extent_crop', None))

        vector_source = GeoJSONVectorSourceConfig(
            uri=label_uri, default_class_id=1, ignore_crs_field=True)
        label_source = ChipClassificationLabelSourceConfig(
            vector_source=vector_source,
            infer_cells=True,
            ioa_thresh=0.33,
            use_intersection_over_cell=False,
            pick_min_class_id=False,
            background_class_id=0)

        return SceneConfig(
            id=id,
            raster_source=raster_source,
            label_source=label_source,
            aoi_uris=kwargs.get('aoi_uris', []))

    class_config = ClassConfig(names=['background', 'vehicle'])

    train_scenes = []
    for i, _id in enumerate(train_ids):
        try:
            scene = make_scene(_id, aoi_uris=[aoi_uris[i]])
        except:
            scene = make_scene(_id)
        train_scenes.append(scene)

    scene_dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=train_scenes,
        validation_scenes=[make_scene(id) for id in val_ids])

    window_opts = {}
    # set window configs for training scenes
    for s in scene_dataset.train_scenes:
        window_opts[s.id] = GeoDataWindowConfig(
            method=GeoDataWindowMethod.random,
            size=chip_sz,
            size_lims=(chip_sz, chip_sz + 1),
            max_windows=100)
    # set window configs for validation scenes
    for s in scene_dataset.validation_scenes:
        window_opts[s.id] = GeoDataWindowConfig(
            method=GeoDataWindowMethod.sliding,
            size=chip_sz,
            stride=chip_sz // 2)

    data = ClassificationGeoDataConfig(
        scene_dataset=scene_dataset,
        window_opts=window_opts,
        img_sz=img_sz,
        num_workers=num_workers)

    if external_loss:
        external_loss_def = ExternalModuleConfig(
            github_repo='AdeelH/pytorch-multi-class-focal-loss',
            name='focal_loss',
            entrypoint='focal_loss',
            force_reload=False,
            entrypoint_kwargs={
                'alpha': [.25, .75],
                'gamma': 2
            })
    else:
        external_loss_def = None

    model = ClassificationModelConfig(backbone=Backbone.resnet18)
    solver = SolverConfig(
        lr=lr,
        num_epochs=epochs,
        batch_sz=batch_size,
        one_cycle=one_cycle,
        external_loss_def=external_loss_def)
    backend = PyTorchChipClassificationConfig(
        data=data,
        model=model,
        solver=solver,
        log_tensorboard=False,
        run_tensorboard=False)

    pipeline = ChipClassificationConfig(
        root_uri=root_uri,
        dataset=scene_dataset,
        backend=backend,
        train_chip_sz=chip_sz,
        predict_chip_sz=chip_sz)

    return pipeline
