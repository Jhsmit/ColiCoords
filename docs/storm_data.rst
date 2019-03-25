Processing SMLM data
====================

(section under construction)

SMLM-type datasets can be processed via ColiCoords in a similar fashion as image-based data. The data flow of this type
data element is exactly the same as other data classes. The input data format is a numpy structured array, where the
required entires are `x`, `y` coordinates as well as a `frame` entry which associated the given localization with an image
frame. Optionally, the structured array can be supplemented with additional information such as number of number of
photons (intensity), parameters of fitted gaussian or chi-squared value.

A helper function is provided to load the .csv output from the ThunderSTORM_ super-resolution analysis package into a
numpy structured table which can be used in ColiCoords. Other super-resolution software will be supported in the future,
at the moment users should load the data themselves and parse to a numpy structured array using standard python functions.




The default input for STORM data in
ColiCoords is the ThunderSTORM_ output format, and this data can be loaded through
:meth:`~colicoords.fileIO.load_thunderstorm`. Since the units in the ThunderSTORM output file are typically in nanometers,
and ColiCoords internal units are pixels, the `pixelsize` kwarg is required to convert to the correct units.

The output from :meth:`~colicoords.fileIO.load_thunderstorm` is a numpy structured array with the at least the fields `x`,
`y` and `frame`. The `frame` entry is used for slicing a :class:`~colicoords.data_models.Data` object in the z or t
dimension; axis 0 for a 3D :class:`colicoords.data_models.Data` object.





.. _ThunderSTORM: http://zitmen.github.io/thunderstorm/