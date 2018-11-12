import os
import configparser
import numpy as np


#https://stackoverflow.com/questions/128573/using-property-on-classmethods
class classproperty(object):
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


class DefaultConfig(object):
    # Pixel sizes are in nm, displayed units are um
    IMG_PIXELSIZE = 80
    PAD_WIDTH = 3

    # Optimization bounds defaults
    ENDCAP_RANGE = 20. # Endcap (`xl`, `xr`) are allowed to varied this many pixels from the initital guesses.

    # Plotting default values
    R_DIST_STOP = 20.
    R_DIST_STEP = 0.5
    R_DIST_SIGMA = 0.3
    R_DIST_NORM_STOP = 2
    R_DIST_NORM_STEP = 0.05
    R_DIST_NORM_SIGMA = 0.05

    L_DIST_NBINS = 100
    L_DIST_SIGMA = 0.5

    ALHPA_DIST_STOP = 180.
    ALPHA_DIST_STEP = 1.

    DEBUG = False # If True, numpy division warnings will be printed.

    #Other
    @classproperty
    def CACHE_DIR(self):
        return os.path.join(os.path.expanduser('~'), '.colicoords', 'cache')


config_sections = {
    'General': ['IMG_PIXELSIZE', 'PAD_WIDTH', 'CELL_FRACTION'],
    'Optimization': ['ENDCAP_RANGE'],
    'Plotting': ['R_DIST_STOP', 'R_DIST_STEP', 'R_DIST_SIGMA', 'R_DIST_NORM_STOP', 'R_DIST_NORM_STEP',
                 'R_DIST_NORM_SIGMA', 'L_DIST_NBINS', 'L_DIST_SIGMA', 'ALHPA_DIST_STOP', 'ALPHA_DIST_STEP'],
    'Other': ['CACHE_DIR', 'DEBUG']
}

reverse_sections = {vi: k for k, v in config_sections.items() for vi in v}


class ParsedConfig(object):
    getters = {bool: 'getboolean', float: 'getfloat', int: 'getint', str: 'get'}

    def __init__(self, config):
        self.config = config

    def __getattr__(self, name):
        _type = type(getattr(DefaultConfig, name))
        get = self.getters[_type]
        return getattr(self.config[reverse_sections[name]], get)(name)


def load_config(path=None):
    """
    Load the configuration file at `path`. If not file is given the default directory is used. If no config file is
    present, the default config is used.

    Parameters
    ----------
    path : :obj:`str`:
        Optional path.

    """
    global cfg
    path = path if path else os.path.join(os.path.expanduser('~'), '.colicoords', 'config.ini')
    if os.path.exists(path):
        config = configparser.ConfigParser()
        config.read(path)
        cfg = ParsedConfig(config)
    else:
        cfg = DefaultConfig


def create_config(path=None):
    """
    Create a config file at the specified `path` using default values. If not path is given the file is created in the
    user's home directory / .colicoords.

    Parameters
    ----------
    path : :obj:`str`
        Optional path where to create the config file.
    """

    if not path:
        home = os.path.expanduser('~')
        path = os.path.join(home, '.colicoords')
        if not os.path.isdir(path):
            os.mkdir(path)

    config = configparser.ConfigParser()
    config.optionxform = str
    for k, v in config_sections.items():
        config[k] = {vi: getattr(DefaultConfig, vi) for vi in v}

    with open(os.path.join(path, 'config.ini'), 'w') as configfile:
        config.write(configfile)


load_config()

try:
    if not cfg.DEBUG:
        np.seterr(divide='ignore', invalid='ignore')
except KeyError:
    print('Invalid config file')
