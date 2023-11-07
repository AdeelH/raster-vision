import pystac_client
from shapely.geometry import mapping

from torch.utils.data import DataLoader

from rastervision.pipeline.file_system.utils import json_to_file
from rastervision.core.box import Box
from rastervision.core.data import XarraySourceConfig, STACItemConfig, Scene
from rastervision.pytorch_learner.dataset import SemanticSegmentationSlidingWindowGeoDataset

if __name__ == '__main__':
    bbox = Box(ymin=48.8155755, xmin=2.224122, ymax=48.902156, xmax=2.4697602)
    raster_source_config = XarraySourceConfig(
        stac=STACItemConfig(
            uri='xarray_source_config/item.json',
            assets=[
                'coastal',  # B01
            ]),
        bbox_map_coords=tuple(bbox),
    )
    rs = raster_source_config.build()
    scene = Scene(id='', raster_source=rs)
    ds = SemanticSegmentationSlidingWindowGeoDataset(
        scene=scene,
        size=100,
        stride=100,
    )
    dl = DataLoader(ds, batch_size=1, num_workers=2)
    print(f'len(dl) = {len(dl)}')
    print('Sampling from DataLoader...')
    for _, (x, _) in zip(range(4), dl):
        print(x.shape)
