from collections.abc import Callable
from os.path import join
import unittest
from unittest.mock import patch

import numpy as np
from xarray import DataArray
from pystac import Item
from pyproj import Transformer
import rasterio
from rasterio.transform import from_origin

from rastervision.pipeline.file_system import get_tmp_dir
from rastervision.core.box import Box
from rastervision.core.data import (
    RasterioSourceConfig, MultiRasterSource, MultiRasterSourceConfig,
    ReclassTransformerConfig, CastTransformerConfig, XarraySource,
    IdentityCRSTransformer, TemporalMultiRasterSource, RasterioSource,
    RasterioCRSTransformer)
from rastervision.core.data.raster_source.multi_raster_source_config import (
    multi_rs_config_upgrader)

from tests import data_file_path


def make_cfg(img_path: str = 'small-rgb-tile.tif',
             **kwargs) -> MultiRasterSourceConfig:
    img_path = data_file_path(img_path)
    r_source = RasterioSourceConfig(uris=[img_path])
    g_source = RasterioSourceConfig(uris=[img_path])
    b_source = RasterioSourceConfig(uris=[img_path])

    cfg = MultiRasterSourceConfig(
        raster_sources=[r_source, g_source, b_source], **kwargs)
    return cfg


def make_cfg_diverse(diff_dtypes: bool = False,
                     **kwargs) -> MultiRasterSourceConfig:
    img_paths = [
        data_file_path('multi_raster_source/const_100_600x600.tiff'),
        data_file_path('multi_raster_source/const_175_60x60.tiff'),
        data_file_path('multi_raster_source/const_250_6x6.tiff')
    ]
    transformers = [[]] * 3
    if diff_dtypes:
        transformers = [
            [],
            [CastTransformerConfig(to_dtype='float32')],
            [CastTransformerConfig(to_dtype='int')],
        ]
    rs_cfgs = [
        RasterioSourceConfig(uris=[path], transformers=tfs)
        for path, tfs in zip(img_paths, transformers)
    ]
    cfg = MultiRasterSourceConfig(raster_sources=rs_cfgs, **kwargs)
    return cfg


class TestMultiRasterSourceConfig(unittest.TestCase):
    def assertNoError(self, fn: Callable, msg: str = ''):
        try:
            fn()
        except Exception:
            self.fail(msg)

    def test_validate_primary_source_idx(self):
        with self.assertRaises(IndexError):
            _ = make_cfg(primary_source_idx=10)
        self.assertNoError(lambda: make_cfg(primary_source_idx=0))
        self.assertNoError(lambda: make_cfg(primary_source_idx=1))
        self.assertNoError(lambda: make_cfg(primary_source_idx=2))

    def test_validate_temporal(self):
        with self.assertRaises(ValueError):
            _ = make_cfg(temporal=True, channel_order=[0, 1, 2])
        self.assertNoError(lambda: make_cfg(temporal=True))

    def test_build(self):
        cfg = make_cfg()
        self.assertNoError(lambda: cfg.build(tmp_dir=get_tmp_dir()))

    def test_build_with_bbox(self):
        cfg = make_cfg(bbox=(0, 0, 1, 1))
        rs = cfg.build(tmp_dir=get_tmp_dir())
        self.assertEqual(rs.bbox, Box(0, 0, 1, 1))

    def test_build_temporal(self):
        cfg = make_cfg(temporal=True)
        rs = cfg.build(tmp_dir=get_tmp_dir())
        self.assertIsInstance(rs, TemporalMultiRasterSource)
        self.assertEqual(rs.shape, (3, 256, 256, 3))

    def test_upgrader_v14(self):
        cfg = make_cfg()
        cfg_dict_old = cfg.dict()
        cfg_dict_old['force_same_dtype'] = True
        cfg_dict_new = multi_rs_config_upgrader(cfg_dict_old, 13)
        self.assertNotIn('force_same_dtype', cfg_dict_new)


class TestMultiRasterSource(unittest.TestCase):
    def assertNoError(self, fn: Callable, msg: str = ''):
        try:
            fn()
        except Exception:
            self.fail(msg)

    def setUp(self):
        self.tmp_dir_obj = get_tmp_dir()
        self.tmp_dir = self.tmp_dir_obj.name

    def tearDown(self):
        self.tmp_dir_obj.cleanup()

    def test_extent(self):
        cfg = make_cfg('small-rgb-tile.tif')
        rs = cfg.build(tmp_dir=self.tmp_dir)
        extent = rs.extent
        h, w = extent.size
        self.assertEqual(h, 256)
        self.assertEqual(w, 256)
        ymin, xmin, ymax, xmax = extent
        self.assertEqual(ymin, 0)
        self.assertEqual(xmin, 0)
        self.assertEqual(ymax, 256)
        self.assertEqual(xmax, 256)

    def test_primary_source_idx(self):
        primary_source_idx = 2

        cfg = make_cfg_diverse(primary_source_idx=primary_source_idx)
        rs = cfg.build(tmp_dir=self.tmp_dir)
        primary_rs = rs.raster_sources[primary_source_idx]

        self.assertEqual(rs.extent, primary_rs.extent)
        self.assertEqual(rs.dtype, primary_rs.dtype)
        self.assertEqual(rs.crs_transformer.transform,
                         primary_rs.crs_transformer.transform)

    def test_dtype_validation(self):
        cfg = make_cfg_diverse(diff_dtypes=True)
        self.assertRaises(ValueError, lambda: cfg.build(tmp_dir=self.tmp_dir))

    def test_bbox(self):
        # /wo user specified extent
        cfg = make_cfg('small-rgb-tile.tif')
        rs = cfg.build(tmp_dir=self.tmp_dir)
        self.assertEqual(rs.bbox, Box(0, 0, 256, 256))
        self.assertEqual(rs.extent, Box(0, 0, 256, 256))

        # /w user specified extent
        cfg_crop = make_cfg('small-rgb-tile.tif', bbox=(64, 64, 192, 192))
        rs_crop = cfg_crop.build(tmp_dir=self.tmp_dir)

        # test extent box
        self.assertEqual(rs_crop.bbox, Box(64, 64, 192, 192))
        self.assertEqual(rs_crop.extent, Box(0, 0, 128, 128))

    def test_get_chip(self):
        # create a 3-channel raster from a 1-channel raster
        # using a ReclassTransformer to give each channel a different value
        # (100, 175, and 250 respectively)
        path = data_file_path('multi_raster_source/const_100_600x600.tiff')
        source_1 = RasterioSourceConfig(uris=[path], channel_order=[0])
        source_2 = RasterioSourceConfig(
            uris=[path],
            channel_order=[0],
            transformers=[ReclassTransformerConfig(mapping={100: 175})])
        source_3 = RasterioSourceConfig(
            uris=[path],
            channel_order=[0],
            transformers=[ReclassTransformerConfig(mapping={100: 250})])

        cfg = MultiRasterSourceConfig(
            raster_sources=[source_1, source_2, source_3],
            channel_order=[2, 1, 0],
            transformers=[
                ReclassTransformerConfig(mapping={
                    100: 10,
                    175: 17,
                    250: 25
                })
            ])
        rs = cfg.build(tmp_dir=self.tmp_dir)

        window = Box(0, 0, 100, 100)

        # sub transformers and channel_order applied
        chip = rs._get_chip(window)
        self.assertEqual(
            tuple(chip.reshape(-1, 3).mean(axis=0)), (100, 175, 250))

        # sub transformers, channel_order, and transformer applied
        chip = rs.get_chip(window)
        self.assertEqual(tuple(chip.reshape(-1, 3).mean(axis=0)), (25, 17, 10))

    def test_nonidentical_extents_and_resolutions(self):
        cfg = make_cfg_diverse(diff_dtypes=False)
        rs = cfg.build(tmp_dir=self.tmp_dir)
        for get_chip_fn in [rs._get_chip, rs.get_chip]:
            ch_1_only = get_chip_fn(Box(0, 0, 10, 10))
            self.assertEqual(
                tuple(ch_1_only.reshape(-1, 3).mean(axis=0)), (100, 0, 0))
            ch_2_only = get_chip_fn(Box(0, 600, 10, 600 + 10))
            self.assertEqual(
                tuple(ch_2_only.reshape(-1, 3).mean(axis=0)), (0, 175, 0))
            ch_3_only = get_chip_fn(Box(600, 0, 600 + 10, 10))
            self.assertEqual(
                tuple(ch_3_only.reshape(-1, 3).mean(axis=0)), (0, 0, 250))
            full_img = get_chip_fn(Box(0, 0, 600, 600))
            self.assertEqual(set(np.unique(full_img[..., 0])), {100})
            self.assertEqual(set(np.unique(full_img[..., 1])), {0, 175})
            self.assertEqual(set(np.unique(full_img[..., 2])), {0, 250})

    def test_temporal_sub_raster_sources(self):
        dtype = np.uint8
        arr = np.ones((2, 5, 5, 4), dtype=dtype)
        arr *= np.arange(4, dtype=np.uint8)
        da = DataArray(arr, dims=['time', 'x', 'y', 'band'])
        rs = XarraySource(da, IdentityCRSTransformer(), temporal=True)
        mrs = MultiRasterSource([rs, rs])
        self.assertEqual(mrs.shape, (2, 5, 5, 8))

        chip = mrs.get_chip(Box(0, 0, 2, 2))
        chip_expected = np.ones((2, 2, 2, 8), dtype=dtype)
        chip_expected[..., :4] *= np.arange(4, dtype=np.uint8)
        chip_expected[..., 4:] *= np.arange(4, dtype=np.uint8)
        np.testing.assert_array_equal(chip, chip_expected)

        chip = mrs.get_chip(Box(0, 0, 2, 2), out_shape=(1, 1))
        chip_expected = np.ones((2, 1, 1, 8), dtype=dtype)
        chip_expected[..., :4] *= np.arange(4, dtype=np.uint8)
        chip_expected[..., 4:] *= np.arange(4, dtype=np.uint8)
        np.testing.assert_array_equal(chip, chip_expected)

    def test_from_stac(self):
        item = Item.from_file(data_file_path('stac/item.json'))

        # avoid reading actual remote files
        mock_raster_uri = data_file_path('ones.tif')
        item.assets['red'].__setattr__('href', mock_raster_uri)
        item.assets['green'].__setattr__('href', mock_raster_uri)

        # test bbox
        bbox = Box(ymin=0, xmin=0, ymax=100, xmax=100)
        rs = MultiRasterSource.from_stac(
            item, assets=['red', 'green'], bbox=bbox)
        self.assertEqual(rs.bbox, bbox)

        # test bbox_map_coords
        bbox_map_coords = Box(
            ymin=29.977309,
            xmin=31.134949,
            ymax=29.978710,
            xmax=31.136567,
        )
        rs = MultiRasterSource.from_stac(
            item, assets=['red', 'green'], bbox_map_coords=bbox_map_coords)
        self.assertEqual(rs.bbox, Box(ymin=51, xmin=50, ymax=207, xmax=206))

        # test error if both bbox and bbox_map_coords specified
        args = dict(
            item=item,
            assets=['red', 'green'],
            bbox=bbox,
            bbox_map_coords=bbox_map_coords)
        self.assertRaises(ValueError,
                          lambda: MultiRasterSource.from_stac(**args))


# The primary raster in TestMultiRasterSourceAlignment: a 200x200 1m grid in
# UTM zone 17N, close to the eastern edge of the zone.
PRIMARY_CRS = 'EPSG:32617'
PRIMARY_X0, PRIMARY_Y0, PRIMARY_SIZE = 740000., 4430000., 200


def write_coord_raster(path: str, crs: str, transform, height: int,
                       width: int) -> None:
    """Write a 2-band raster where each pixel holds the (x, y) coordinates of
    its center in PRIMARY_CRS, relative to the primary raster's origin.

    The difference between the values of two rasters at the same location of
    a chip is therefore the misalignment between them in meters (which is
    also pixels of the primary raster).
    """
    rows, cols = np.mgrid[0:height, 0:width]
    xs, ys = rasterio.transform.xy(
        transform, rows.ravel(), cols.ravel(), offset='center')
    tf = Transformer.from_crs(crs, PRIMARY_CRS, always_xy=True)
    x, y = tf.transform(np.array(xs), np.array(ys))
    arr = np.stack([
        np.reshape(x, (height, width)) - PRIMARY_X0,
        np.reshape(y, (height, width)) - PRIMARY_Y0
    ]).astype(np.float32)
    with rasterio.open(
            path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=2,
            dtype='float32',
            crs=crs,
            transform=transform) as ds:
        ds.write(arr)


def write_projected_coord_raster(path: str,
                                 crs: str,
                                 res: float,
                                 margin: float = 20.) -> None:
    """Write a coordinate raster in the given CRS that covers the primary
    raster's extent plus a margin (in meters)."""
    xmin, xmax = PRIMARY_X0 - margin, PRIMARY_X0 + PRIMARY_SIZE + margin
    ymin, ymax = PRIMARY_Y0 - PRIMARY_SIZE - margin, PRIMARY_Y0 + margin
    tf = Transformer.from_crs(PRIMARY_CRS, crs, always_xy=True)
    xs, ys = tf.transform([xmin, xmin, xmax, xmax], [ymin, ymax, ymin, ymax])
    width = int(np.ceil((max(xs) - min(xs)) / res))
    height = int(np.ceil((max(ys) - min(ys)) / res))
    transform = from_origin(min(xs), max(ys), res, res)
    write_coord_raster(path, crs, transform, height, width)


class TestMultiRasterSourceAlignment(unittest.TestCase):
    """Test that chips from sub raster sources with different pixel grids
    are correctly aligned."""

    def setUp(self):
        self.tmp_dir_obj = get_tmp_dir()
        self.tmp_dir = self.tmp_dir_obj.name
        self.primary_path = join(self.tmp_dir, 'primary.tif')
        write_coord_raster(self.primary_path, PRIMARY_CRS,
                           from_origin(PRIMARY_X0, PRIMARY_Y0, 1, 1),
                           PRIMARY_SIZE, PRIMARY_SIZE)
        self.windows = [
            Box.make_square(10, 10, 64),
            Box.make_square(100, 30, 64),
            Box.make_square(50, 130, 64)
        ]

    def tearDown(self):
        self.tmp_dir_obj.cleanup()

    def get_max_error(self, rs: MultiRasterSource, window: Box,
                      **kwargs) -> float:
        chip = rs.get_chip(window, **kwargs)
        primary, other = chip[..., :2], chip[..., 2:]
        err = np.hypot(*(other - primary).transpose(2, 0, 1))
        return float(err.max())

    def test_different_grids(self):
        cases = {
            'same CRS, offset 3m grid': (PRIMARY_CRS, 3., 0.),
            'EPSG:4326': ('EPSG:4326', 1e-5, 0.15),
            'adjacent UTM zone': ('EPSG:32618', 1., 0.15),
        }
        for name, (crs, res, tol) in cases.items():
            with self.subTest(name):
                path = join(self.tmp_dir, 'other.tif')
                write_projected_coord_raster(path, crs, res, margin=21.7)
                rs = MultiRasterSource(
                    [RasterioSource(self.primary_path),
                     RasterioSource(path)])
                for window in self.windows:
                    err = self.get_max_error(rs, window)
                    self.assertLessEqual(err, tol + 1e-3)
                    # resized chips should also be aligned
                    err = self.get_max_error(rs, window, out_shape=(32, 32))
                    self.assertLessEqual(err, tol + 1e-3)

    def test_same_grid_with_offset(self):
        # same CRS and resolution, offset by a whole number of pixels
        path = join(self.tmp_dir, 'other.tif')
        write_coord_raster(path, PRIMARY_CRS,
                           from_origin(PRIMARY_X0 - 13, PRIMARY_Y0 + 7, 1, 1),
                           PRIMARY_SIZE + 20, PRIMARY_SIZE + 20)
        rs = MultiRasterSource(
            [RasterioSource(self.primary_path),
             RasterioSource(path)])
        # this case should not need any reprojection
        with patch.object(
                RasterioSource,
                'get_chip_reprojected',
                side_effect=AssertionError('should not be called')):
            for window in self.windows:
                self.assertEqual(self.get_max_error(rs, window), 0)

    def test_primary_bbox(self):
        path = join(self.tmp_dir, 'other.tif')
        write_projected_coord_raster(path, 'EPSG:32618', 1.)
        rs = MultiRasterSource(
            [RasterioSource(self.primary_path),
             RasterioSource(path)],
            bbox=Box(30, 40, 150, 160))
        # windows within the extent (120 x 120) of the cropped raster source
        windows = [Box.make_square(10, 10, 64), Box.make_square(50, 40, 64)]
        for window in windows:
            self.assertLessEqual(self.get_max_error(rs, window), 0.15)

    def test_non_primary_bbox(self):
        # pixels outside the non-primary raster source's bbox should be 0
        path = join(self.tmp_dir, 'other.tif')
        write_coord_raster(
            path, PRIMARY_CRS,
            from_origin(PRIMARY_X0 - 10.25, PRIMARY_Y0 + 10.25, 2, 2), 110,
            110)
        # the bbox covers 40 x 50 pixels (80 x 100 primary pixels) starting
        # at pixel (25, 30), which is at (39.75, 49.75) in primary pixel
        # coords, i.e. between the centers of primary pixels 39 and 40 (rows)
        # and 49 and 50 (cols)
        other_rs = RasterioSource(path, bbox=Box(25, 30, 65, 80))
        rs = MultiRasterSource([RasterioSource(self.primary_path), other_rs])
        chip = rs.get_chip(Box(0, 0, 200, 200))
        nonzero = np.any(chip[..., 2:] != 0, axis=-1)
        rows, cols = np.nonzero(nonzero)
        self.assertEqual((rows.min(), rows.max()), (40, 119))
        self.assertEqual((cols.min(), cols.max()), (50, 149))
        self.assertTrue(nonzero[40:120, 50:150].all())

    def test_coarse_non_rasterio_source(self):
        # A non-RasterioSource that is much coarser than the primary source,
        # such that the window maps to less than a pixel.
        arr = np.full((4, 4, 2), 7, dtype=np.float32)
        da = DataArray(arr, dims=['y', 'x', 'band'])
        crs_tf = RasterioCRSTransformer(
            from_origin(PRIMARY_X0 - 3, PRIMARY_Y0 + 3, 51, 51), PRIMARY_CRS)
        xr_rs = XarraySource(da, crs_tf)
        rs = MultiRasterSource([RasterioSource(self.primary_path), xr_rs])
        chip = rs.get_chip(Box(10, 10, 20, 20))
        np.testing.assert_array_equal(chip[..., 2:], 7)


if __name__ == '__main__':
    unittest.main()
