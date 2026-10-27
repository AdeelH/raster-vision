import os
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin

from rastervision.core.utils.cog import (create_cog, DEFAULT_BLOCK_SIZE,
                                         DEFAULT_OVERVIEWS)
from rastervision.pipeline.file_system.utils import get_tmp_dir

CRS = 'EPSG:32617'
TRANSFORM = from_origin(500000.5, 4000000.25, 0.3, 0.3)


def make_raster(path: str,
                height: int = 1024,
                width: int = 1024,
                count: int = 3,
                dtype: str = 'uint8',
                nodata: float | None = None) -> np.ndarray:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(count, height, width)).astype(dtype)
    profile = dict(
        driver='GTiff',
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=CRS,
        transform=TRANSFORM,
        nodata=nodata)
    with rasterio.open(path, 'w', **profile) as ds:
        ds.write(arr)
    return arr


class TestCreateCog(unittest.TestCase):
    def setUp(self):
        self.tmp_dir_obj = get_tmp_dir()
        self.tmp_dir = self.tmp_dir_obj.name
        self.src_path = os.path.join(self.tmp_dir, 'src.tif')
        self.dst_path = os.path.join(self.tmp_dir, 'out', 'cog.tif')
        self.local_dir = os.path.join(self.tmp_dir, 'local')
        os.makedirs(self.local_dir)

    def tearDown(self):
        self.tmp_dir_obj.cleanup()

    def assert_preserved(self,
                         ds: rasterio.DatasetReader,
                         arr: np.ndarray,
                         nodata: float | None = None):
        self.assertEqual(ds.crs, rasterio.crs.CRS.from_string(CRS))
        self.assertEqual(ds.transform, TRANSFORM)
        self.assertEqual(ds.count, arr.shape[0])
        self.assertEqual((ds.height, ds.width), arr.shape[1:])
        self.assertTrue(all(dt == arr.dtype for dt in ds.dtypes))
        self.assertEqual(ds.nodata, nodata)
        np.testing.assert_array_equal(ds.read(), arr)

    def test_defaults(self):
        arr = make_raster(self.src_path)
        create_cog(self.src_path, self.dst_path, self.local_dir)
        self.assertTrue(os.path.isfile(self.dst_path))
        with rasterio.open(self.dst_path) as ds:
            self.assertEqual(
                ds.tags(ns='IMAGE_STRUCTURE').get('LAYOUT'), 'COG')
            self.assertEqual(ds.compression.value, 'DEFLATE')
            for i in ds.indexes:
                self.assertEqual(ds.overviews(i), DEFAULT_OVERVIEWS)
            self.assertEqual(ds.block_shapes,
                             [(DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_SIZE)] * 3)
            self.assert_preserved(ds, arr)
        # check overview tiling as well
        with rasterio.open(self.dst_path, overview_level=0) as ovr:
            self.assertEqual((ovr.height, ovr.width), (512, 512))
            self.assertEqual(ovr.block_shapes[0],
                             (DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_SIZE))
        # intermediate file should have been cleaned up
        self.assertFalse(
            os.path.exists(os.path.join(self.local_dir, 'src-translate.tif')))

    def test_custom_args(self):
        arr = make_raster(self.src_path, height=600, width=700, count=1)
        create_cog(
            self.src_path,
            self.dst_path,
            self.local_dir,
            block_size=256,
            resample_method='average',
            compression='lzw',
            overviews=[2, 4])
        with rasterio.open(self.dst_path) as ds:
            self.assertEqual(
                ds.tags(ns='IMAGE_STRUCTURE').get('LAYOUT'), 'COG')
            self.assertEqual(ds.compression.value, 'LZW')
            self.assertEqual(ds.overviews(1), [2, 4])
            self.assertEqual(ds.block_shapes, [(256, 256)])
            self.assert_preserved(ds, arr)
            # average resampling of the first 2x2 block
            ovr = ds.read(1, out_shape=(300, 350))
        expected = arr[0, :2, :2].astype(float).mean()
        self.assertAlmostEqual(float(ovr[0, 0]), expected, delta=1)

    def test_no_compression(self):
        arr = make_raster(
            self.src_path, height=320, width=320, dtype='float32', nodata=-1)
        create_cog(
            self.src_path, self.dst_path, self.local_dir, compression='none')
        with rasterio.open(self.dst_path) as ds:
            self.assertEqual(
                ds.tags(ns='IMAGE_STRUCTURE').get('LAYOUT'), 'COG')
            self.assertIsNone(ds.compression)
            self.assertEqual(ds.overviews(1), DEFAULT_OVERVIEWS)
            self.assert_preserved(ds, arr, nodata=-1)

    def test_invalid_resample_method(self):
        make_raster(self.src_path, height=64, width=64)
        with self.assertRaises(ValueError):
            create_cog(
                self.src_path,
                self.dst_path,
                self.local_dir,
                resample_method='foo')


if __name__ == '__main__':
    unittest.main()
