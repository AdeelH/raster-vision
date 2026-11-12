from collections.abc import Callable
import os
import unittest

import boto3
from moto import mock_aws
from shapely.geometry import Polygon

from rastervision.pipeline.file_system import get_tmp_dir
from rastervision.core.utils.stac import (S3StacIO, setup_stac_io,
                                          read_stac)

from tests import data_file_path


class TestStac(unittest.TestCase):
    def assertNoError(self, fn: Callable, msg: str = ''):
        try:
            fn()
        except Exception:
            self.fail(msg)

    def test_setup_stac_io(self):
        self.assertNoError(setup_stac_io)

    @mock_aws
    def test_setup_stac_io_s3(self):
        from pystac import StacIO
        default_stac_io = StacIO.default()
        try:
            setup_stac_io()
            stac_io = StacIO.default()
            assert isinstance(stac_io, S3StacIO)
            s3 = boto3.client('s3', region_name='us-east-1')
            s3.create_bucket(Bucket='bucket')
            uri = 's3://bucket/catalog.json'
            stac_io.write_text_to_href(uri, '{"a": 1}')
            self.assertEqual(stac_io.read_text_from_href(uri), '{"a": 1}')
        finally:
            StacIO.set_default(lambda: default_stac_io)

    def test_read_stac(self):
        zip_path = data_file_path('catalog.zip')
        expected_keys = {
            'label_uri': str,
            'image_uris': list,
            'label_bbox': Polygon,
            'image_bbox': (type(None), Polygon),
            'bboxes_intersect': bool,
            'aoi_geometry': (type(None), dict)
        }

        with get_tmp_dir() as tmp_dir:
            out = read_stac(zip_path, tmp_dir)

            # check for correctness of format
            self.assertIsInstance(out, list)
            for o in out:
                self.assertIsInstance(o, dict)
                for k, v in o.items():
                    self.assertTrue(k in expected_keys)
                    self.assertIsInstance(v, expected_keys[k])
                for uri in o['image_uris']:
                    self.assertIsInstance(uri, str)

            # check for correctness of content (WRT the test catalog)
            self.assertEqual(len(out), 1)
            self.assertEqual(len(out[0]['image_uris']), 1)

            for o in out:
                label_uri = o['label_uri']
                self.assertTrue(os.path.exists(label_uri))
                self.assertTrue(label_uri.startswith(tmp_dir))


if __name__ == '__main__':
    unittest.main()
