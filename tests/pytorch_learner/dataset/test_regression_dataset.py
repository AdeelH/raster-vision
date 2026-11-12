from os.path import join
import unittest

from rastervision.pipeline.file_system import get_tmp_dir, make_dir
from rastervision.pytorch_learner.dataset.regression_dataset import (
    RegressionDataReader,
)


class TestRegressionDataReader(unittest.TestCase):
    def test_len(self):
        with get_tmp_dir() as tmp_dir:
            make_dir(join(tmp_dir, 'img'))
            with open(join(tmp_dir, 'labels.csv'), 'w') as f:
                f.write('image, a, b\n0.png, 1.0, 2.0\n1.png, 3.0, 4.0\n')
            reader = RegressionDataReader(tmp_dir, class_names=['b'])
            self.assertEqual(len(reader), 2)
            self.assertEqual(reader.targets, [[2.0], [4.0]])
