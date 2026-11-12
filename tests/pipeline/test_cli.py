import os
import unittest
from os.path import dirname, join

from click.testing import CliRunner

from rastervision.pipeline.cli import (
    convert_bool_args,
    get_configs,
    main,
    print_error,
)
from rastervision.pipeline.config import Config
from rastervision.pipeline.file_system.utils import get_tmp_dir
from rastervision.pipeline_example_plugin1 import config1
from rastervision.pipeline_example_plugin1.sample_pipeline import (
    SamplePipelineConfig,
)


class TestCli(unittest.TestCase):
    def setUp(self):
        self.tmp_dir_obj = get_tmp_dir()
        self.root_uri = self.tmp_dir_obj.name

    def tearDown(self):
        self.tmp_dir_obj.cleanup()

    def test_rastervision_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ['--help'])
        self.assertEqual(result.exit_code, 0)

    def test_rastervision_run_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ['run', '--help'])
        self.assertEqual(result.exit_code, 0)

    def test_rastervision_run_local(self):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                'run',
                'local',
                'rastervision.pipeline_example_plugin1.config1',
                '-a',
                'root_uri',
                self.root_uri,
                '--splits',
                '2',
            ],
        )
        self.assertEqual(result.exit_code, 0)

        # from config path
        cfg_path = join(dirname(config1.__file__), 'config1.py')
        result = runner.invoke(
            main,
            [
                'run',
                'local',
                cfg_path,
                '-a',
                'root_uri',
                self.root_uri,
                '--splits',
                '2',
            ],
        )
        self.assertEqual(result.exit_code, 0)

    def test_rastervision_run_inprocess1(self):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                'run',
                'inprocess',
                'rastervision.pipeline_example_plugin1.config1',
                '-a',
                'root_uri',
                self.root_uri,
                '--splits',
                '2',
            ],
        )
        self.assertEqual(result.exit_code, 0)

    def test_rastervision_run_inprocess2(self):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                'run',
                'inprocess',
                'rastervision.pipeline_example_plugin1.config2',
                '-a',
                'root_uri',
                self.root_uri,
                '--splits',
                '2',
            ],
        )
        self.assertEqual(result.exit_code, 0)

    def test_rastervision_run_inprocess3(self):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                'run',
                'inprocess',
                'rastervision.pipeline_example_plugin2.config3',
                '-a',
                'root_uri',
                self.root_uri,
                '--splits',
                '2',
            ],
        )
        self.assertEqual(result.exit_code, 0)


class TestUtils(unittest.TestCase):
    def test_print_error(self):
        print_error('error')

    def test_convert_bool_args(self):
        args_in = dict(a='true', b='false')
        args_out = convert_bool_args(args_in)
        self.assertDictEqual(args_out, dict(a=True, b=False))

    def test_get_configs_json(self):
        cfg = SamplePipelineConfig(root_uri='abc', names=['x', 'y', 'z'])
        with get_tmp_dir() as tmp_dir:
            cfg_path = join(tmp_dir, 'cfg.json')
            cfg.to_file(cfg_path)
            cfgs = get_configs(cfg_path)
        self.assertEqual(len(cfgs), 1)
        cfg_out = cfgs[0]
        assert isinstance(cfg_out, SamplePipelineConfig)
        self.assertEqual(cfg_out.root_uri, cfg.root_uri)
        self.assertListEqual(cfg_out.names, cfg.names)

    def test_get_configs_no_func(self):
        with get_tmp_dir() as tmp_dir:
            cfg_path = join(tmp_dir, 'cfg.py')
            with open(cfg_path, 'w'):
                pass
            self.assertRaises(ImportError, lambda: get_configs(cfg_path))
            os.remove(cfg_path)

    def test_get_configs_not_pipeline_cfg(self):
        cfg = Config()
        with get_tmp_dir() as tmp_dir:
            cfg_path = join(tmp_dir, 'cfg.json')
            cfg.to_file(cfg_path)
            self.assertRaises(TypeError, lambda: get_configs(cfg_path))


if __name__ == '__main__':
    unittest.main()
