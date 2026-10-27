import unittest
from unittest.mock import patch
import os
from os.path import join

import numpy as np
import pyproj
import rasterio as rio
from rasterio.transform import Affine, from_origin

from rastervision.pipeline.file_system.utils import get_tmp_dir
from rastervision.core.box import Box
from rastervision.core.data.utils.rasterio import (
    build_vrt, crop_geotiff, download_and_build_vrt, get_aws_session,
    write_geotiff_like_geojson, write_bbox)
from rastervision.core.data import RasterioSource, GeoJSONVectorSource
from tests import data_file_path

RIO_UTILS_LOGGER = 'rastervision.core.data.utils.rasterio'


def write_tile(path: str,
               arr: np.ndarray,
               xmin: float,
               ymax: float,
               res: float = 1.,
               crs: str = 'EPSG:32615',
               nodata: float | None = None,
               transform: Affine | None = None) -> str:
    """Write a (C, H, W) array to a GeoTIFF."""
    if transform is None:
        transform = from_origin(xmin, ymax, res, res)
    count, height, width = arr.shape
    with rio.open(
            path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=count,
            dtype=arr.dtype,
            crs=crs,
            transform=transform,
            nodata=nodata) as ds:
        ds.write(arr)
    return path


class TestRasterioUtils(unittest.TestCase):
    def test_write_bbox(self):
        bbox = Box(ymin=48.815, xmin=2.224, ymax=48.902, xmax=2.469)
        crs_wkt = pyproj.CRS('epsg:4326').to_wkt()
        r = bbox.width / bbox.height
        arr1 = np.zeros((50, int(100 * r)))
        arr2 = np.zeros((50, int(100 * r), 4))
        with get_tmp_dir() as tmp_dir:
            geotiff_path = join(tmp_dir, 'test.geotiff')
            write_bbox(geotiff_path, arr1, bbox=bbox, crs_wkt=crs_wkt)
            rs = RasterioSource(geotiff_path)
            geotiff_bbox = rs.crs_transformer.pixel_to_map(rs.extent)
            np.testing.assert_array_almost_equal(
                np.array(list(geotiff_bbox)), np.array(list(bbox)), decimal=3)
            self.assertEqual(rs.shape, (*arr1.shape, 1))

            write_bbox(geotiff_path, arr2, bbox=bbox, crs_wkt=crs_wkt)
            rs = RasterioSource(geotiff_path)
            geotiff_bbox = rs.crs_transformer.pixel_to_map(rs.extent)
            np.testing.assert_array_almost_equal(
                np.array(list(geotiff_bbox)), np.array(list(bbox)), decimal=3)
            self.assertEqual(rs.shape, arr2.shape)

    def test_crop_geotiff(self):
        src_path = data_file_path('multi_raster_source/const_100_600x600.tiff')
        window = Box(0, 0, 10, 10)
        with get_tmp_dir() as tmp_dir:
            crop_path = join(tmp_dir, 'test.tiff')
            crop_geotiff(src_path, window, crop_path)
            rs = RasterioSource(crop_path)
            self.assertEqual(rs.extent, window)

    def test_write_geotiff_like_geojson(self):
        geojson_path = data_file_path('0-aoi.geojson')
        arr = np.zeros((10, 10))
        with get_tmp_dir() as tmp_dir:
            geotiff_path = join(tmp_dir, 'test.tiff')
            write_geotiff_like_geojson(
                geotiff_path, arr, geojson_path, crs=None)
            rs = RasterioSource(geotiff_path)
            geotiff_bbox = rs.crs_transformer.pixel_to_map(rs.extent)
            vs = GeoJSONVectorSource(geojson_path, rs.crs_transformer)
            geojson_bbox = rs.crs_transformer.pixel_to_map(vs.extent)
            np.testing.assert_array_almost_equal(
                np.array(list(geotiff_bbox)),
                np.array(list(geojson_bbox)),
                decimal=3)
            self.assertEqual(rs.shape, (10, 10, 1))

    @patch.dict('os.environ', AWS_REQUEST_PAYER='requester')
    def test_get_aws_session(self):
        session = get_aws_session()
        self.assertTrue(session.requester_pays)

    @patch.dict('sys.modules', {'rastervision.aws_s3': None})
    @patch.dict('os.environ', AWS_S3_REQUESTER_PAYS='false')
    def test_get_aws_session_no_rv_aws_s3_and_not_requester_pays(self):
        session = get_aws_session()
        self.assertFalse(session.requester_pays)


class TestBuildVRT(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def _write_2x2_grid(self, tmp_dir: str) -> tuple[list[str], np.ndarray]:
        """Write a 2x2 grid of adjacent 10x12 3-band tiles.

        Returns the tile paths and the expected (C, H, W) mosaic.
        """
        h, w = 10, 12
        mosaic = self.rng.integers(1, 255, (3, 2 * h, 2 * w), dtype=np.uint8)
        paths = []
        for i in range(2):
            for j in range(2):
                tile = mosaic[:, i * h:(i + 1) * h, j * w:(j + 1) * w]
                path = join(tmp_dir, f'tile_{i}_{j}.tif')
                write_tile(path, tile, xmin=100 + j * w, ymax=500 - i * h)
                paths.append(path)
        return paths, mosaic

    def test_build_vrt_grid(self):
        with get_tmp_dir() as tmp_dir:
            paths, mosaic = self._write_2x2_grid(tmp_dir)
            vrt_path = join(tmp_dir, 'vrt', 'mosaic.vrt')
            # shuffle the order to make sure it doesn't matter
            build_vrt(vrt_path, paths[::-1])
            self.assertTrue(os.path.exists(vrt_path))
            with rio.open(vrt_path) as ds:
                self.assertEqual(ds.shape, (20, 24))
                self.assertEqual(ds.count, 3)
                self.assertEqual(ds.dtypes, ('uint8', ) * 3)
                self.assertEqual(ds.crs, rio.crs.CRS.from_epsg(32615))
                self.assertEqual(ds.transform, from_origin(100, 500, 1, 1))
                self.assertEqual(ds.nodatavals, (None, ) * 3)
                np.testing.assert_array_equal(ds.read(), mosaic)
                # quadrant-by-quadrant
                for i in range(2):
                    for j in range(2):
                        window = rio.windows.Window(j * 12, i * 10, 12, 10)
                        with rio.open(paths[i * 2 + j]) as tile_ds:
                            np.testing.assert_array_equal(
                                ds.read(window=window), tile_ds.read())

    def test_build_vrt_special_chars_in_path(self):
        with get_tmp_dir() as tmp_dir:
            tile_dir = join(tmp_dir, 'a&b <c>')
            os.makedirs(tile_dir)
            paths, mosaic = self._write_2x2_grid(tile_dir)
            vrt_path = join(tmp_dir, 'mosaic.vrt')
            build_vrt(vrt_path, paths)
            with rio.open(vrt_path) as ds:
                np.testing.assert_array_equal(ds.read(), mosaic)

    def test_build_vrt_nodata(self):
        nodata = 255
        arr1 = self.rng.integers(1, 200, (1, 10, 10), dtype=np.uint8)
        arr2 = self.rng.integers(1, 200, (1, 10, 10), dtype=np.uint8)
        # left half of tile 2 is nodata and overlaps with tile 1, so it should
        # not overwrite tile 1's values
        arr2[:, :, :5] = nodata
        with get_tmp_dir() as tmp_dir:
            paths = [
                write_tile(join(tmp_dir, '1.tif'), arr1, 0, 20, nodata=nodata),
                # overlaps with the right half of tile 1
                write_tile(join(tmp_dir, '2.tif'), arr2, 5, 20, nodata=nodata),
                # leaves a gap below tile 1 and 2
                write_tile(
                    join(tmp_dir, '3.tif'), arr1, 20, 10, nodata=nodata),
            ]
            vrt_path = join(tmp_dir, 'mosaic.vrt')
            build_vrt(vrt_path, paths)
            with rio.open(vrt_path) as ds:
                self.assertEqual(ds.shape, (20, 30))
                self.assertEqual(ds.nodatavals, (nodata, ))
                self.assertEqual(ds.transform, from_origin(0, 20, 1, 1))
                out = ds.read(1)
        expected = np.full((20, 30), nodata, dtype=np.uint8)
        expected[:10, :10] = arr1[0]
        expected[:10, 10:15] = arr2[0, :, 5:]
        expected[10:, 20:] = arr1[0]
        np.testing.assert_array_equal(out, expected)

    def test_build_vrt_different_resolutions(self):
        arr1 = np.full((1, 10, 10), 1, dtype=np.uint8)
        arr2 = np.full((1, 5, 5), 2, dtype=np.uint8)
        with get_tmp_dir() as tmp_dir:
            paths = [
                write_tile(join(tmp_dir, '1.tif'), arr1, 0, 10, res=1.),
                write_tile(join(tmp_dir, '2.tif'), arr2, 10, 10, res=2.),
            ]
            vrt_path = join(tmp_dir, 'mosaic.vrt')
            build_vrt(vrt_path, paths)
            with rio.open(vrt_path) as ds:
                # average resolution
                self.assertEqual(ds.res, (1.5, 1.5))
                self.assertEqual(ds.transform, from_origin(0, 10, 1.5, 1.5))
                # extent of 20 x 10 map units at res 1.5
                self.assertEqual(ds.shape, (7, 13))
                out = ds.read(1)
        # the boundary between the tiles falls at col 6.67
        np.testing.assert_array_equal(out[:, :6], 1)
        np.testing.assert_array_equal(out[:, 7:], 2)

    def test_build_vrt_skips_mismatched_crs(self):
        arr = np.ones((1, 10, 10), dtype=np.uint8)
        with get_tmp_dir() as tmp_dir:
            paths = [
                write_tile(join(tmp_dir, '1.tif'), arr, 100, 500),
                write_tile(
                    join(tmp_dir, '2.tif'), arr * 2, 110, 500,
                    crs='EPSG:4326'),
                write_tile(join(tmp_dir, '3.tif'), arr * 3, 100, 490),
            ]
            vrt_path = join(tmp_dir, 'mosaic.vrt')
            with self.assertLogs(RIO_UTILS_LOGGER, level='WARNING') as cm:
                build_vrt(vrt_path, paths)
            self.assertEqual(len(cm.records), 1)
            self.assertIn('2.tif', cm.output[0])
            self.assertIn('projection', cm.output[0])
            with rio.open(vrt_path) as ds:
                self.assertEqual(ds.crs, rio.crs.CRS.from_epsg(32615))
                self.assertEqual(ds.shape, (20, 10))
                out = ds.read(1)
                with open(vrt_path) as f:
                    self.assertNotIn('2.tif', f.read())
        np.testing.assert_array_equal(out[:10], 1)
        np.testing.assert_array_equal(out[10:], 3)

    def test_build_vrt_skips_mismatched_bands_dtype_and_rotated(self):
        arr = np.ones((1, 10, 10), dtype=np.uint8)
        rotated_tf = Affine(1, 0.1, 0, 0.1, -1, 10)
        with get_tmp_dir() as tmp_dir:
            paths = [
                write_tile(join(tmp_dir, 'ok.tif'), arr, 0, 10),
                write_tile(
                    join(tmp_dir, 'bands.tif'), np.ones((2, 10, 10), np.uint8),
                    0, 10),
                write_tile(
                    join(tmp_dir, 'dtype.tif'), arr.astype(np.uint16), 0, 10),
                write_tile(
                    join(tmp_dir, 'rotated.tif'),
                    arr,
                    0,
                    10,
                    transform=rotated_tf),
            ]
            vrt_path = join(tmp_dir, 'mosaic.vrt')
            with self.assertLogs(RIO_UTILS_LOGGER, level='WARNING') as cm:
                build_vrt(vrt_path, paths)
            self.assertEqual(len(cm.records), 3)
            for output, name in zip(cm.output, ['bands', 'dtype', 'rotated']):
                self.assertIn(f'{name}.tif', output)
            with rio.open(vrt_path) as ds:
                self.assertEqual(ds.shape, (10, 10))
                self.assertEqual(ds.count, 1)

    def test_build_vrt_no_usable_inputs(self):
        arr = np.ones((1, 10, 10), dtype=np.uint8)
        rotated_tf = Affine(1, 0.1, 0, 0.1, -1, 10)
        with get_tmp_dir() as tmp_dir:
            paths = [
                join(tmp_dir, 'does_not_exist.tif'),
                write_tile(
                    join(tmp_dir, 'rotated.tif'),
                    arr,
                    0,
                    10,
                    transform=rotated_tf),
            ]
            vrt_path = join(tmp_dir, 'mosaic.vrt')
            with self.assertLogs(RIO_UTILS_LOGGER, level='WARNING'):
                with self.assertRaises(ValueError):
                    build_vrt(vrt_path, paths)
            self.assertFalse(os.path.exists(vrt_path))

    def test_download_and_build_vrt(self):
        with get_tmp_dir() as tmp_dir:
            paths, mosaic = self._write_2x2_grid(tmp_dir)
            vrt_dir = join(tmp_dir, 'vrt')
            vrt_path = download_and_build_vrt(paths, vrt_dir)
            self.assertEqual(vrt_path, join(vrt_dir, 'index.vrt'))
            with rio.open(vrt_path) as ds:
                np.testing.assert_array_equal(ds.read(), mosaic)

    def test_rasterio_source_multiple_uris(self):
        with get_tmp_dir() as tmp_dir:
            paths, mosaic = self._write_2x2_grid(tmp_dir)
            mosaic_hwc = mosaic.transpose(1, 2, 0)
            rs = RasterioSource(paths, tmp_dir=join(tmp_dir, 'rs'))
            self.assertEqual(rs.shape, (20, 24, 3))
            self.assertEqual(rs.extent, Box(0, 0, 20, 24))
            # chip that spans all 4 tiles
            chip = rs.get_chip(Box(5, 6, 15, 18))
            np.testing.assert_array_equal(chip, mosaic_hwc[5:15, 6:18])
            np.testing.assert_array_equal(rs[:, :], mosaic_hwc)
            self.assertEqual(rs.image_dataset.transform,
                             from_origin(100, 500, 1, 1))


if __name__ == '__main__':
    unittest.main()
