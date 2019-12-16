import os
import numpy as np
from colicoords.config import DefaultConfig, create_config, load_config, ParsedConfig
import colicoords.config as config
import unittest


class TestConfig(unittest.TestCase):
    def test_defaultconfig(self):
        default_cfg = DefaultConfig()
        cache_dir = default_cfg.CACHE_DIR

    def test_create_config(self):
        create_config()
        load_config()

        self.assertIsInstance(config.cfg, ParsedConfig)

        create_config('.')
        load_config('config.ini')

        self.assertIsInstance(config.cfg, ParsedConfig)
        self.assertEqual(config.cfg.R_DIST_STOP, 20)
        f_path = os.path.dirname(os.path.realpath(__file__))
        load_config(os.path.join(f_path, 'test_data', 'test_config.ini'))
        self.assertEqual(config.cfg.R_DIST_STOP, 30)

        load_config('asdfasdfsadf')
        self.assertIs(config.cfg, DefaultConfig)