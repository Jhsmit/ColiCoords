#!/usr/bin/env python
"""
Load 3D-DAOSTORM (https://github.com/ZhuangLab/storm-analysis) format data.

Hazen 07/19
"""
import numpy
import warnings

import colicoords.fileIO as fileIO

import storm_analysis.sa_library.sa_h5py as saH5Py


def load_daostorm(file_path):
    """
    Load a HDF5 format file from 3D-DAOSTORM output.

    Parameters
    ----------
    file_path : :obj:`str`
        Target file path to 3D-DAOSTORM HDF5 file.
    """
    # At least for now, we only handle tracked files.
    with saH5Py.SAH5Py(file_path) as h5:
        if not h5.hasTracks():
            warnings.warn("No tracks found in '" + file_path + "' HDF5 file.")
        tracks = h5.getTracks(fields = ['frame_number', 'x', 'y'])

    # Change 'frame_number' to 'frame'
    names = list(tracks.keys())
    names = ['frame' if (x == 'frame_number') else x for x in names]

    # Convert to numpy data table.
    dtype = {'names': tuple(names),
             'formats': tuple(fileIO.TYPES[name] for name in names)}

    storm_table = numpy.zeros(tracks["x"].size, dtype = dtype)
    storm_table['frame'] = tracks['frame_number']
    storm_table['x'] = tracks['x']
    storm_table['y'] = tracks['y']

    return storm_table
