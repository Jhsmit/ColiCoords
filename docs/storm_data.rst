Handling of STORM data
======================

(section under construction)


With a coordinate system in place ColiCoords can also handle STORM-type data as input. The basic input data format
requires a table of x and y coordinates of localizations, as well a `frame` entry, optionally supplemented with
additional information such as number of photons (intensity), width of a fitted gaussian and chi-squared. The default input for STORM data in
ColiCoords is the ThunderSTORM_ output format, and this data can be loaded through
:meth:`~colicoords.fileIO.load_thunderstorm`. Since the units in the ThunderSTORM output file are typically in nanometers,
and ColiCoords internal units are pixels, the `pixelsize` kwarg is required to convert to the correct units. For more
information on the coordinate system in ColiCoords see sectiontahtdoesntexistyet.

The output from :meth:`~colicoords.fileIO.load_thunderstorm` is a numpy structured array with the at least the fields `x`,
`y` and `frame`. The `frame` entry is used for slicing a :class:`~colicoords.data_models.Data` object in the z or t
dimension; axis 0 for a 3D :class:`colicoords.data_models.Data` object.





.. _ThunderSTORM: http://zitmen.github.io/thunderstorm/