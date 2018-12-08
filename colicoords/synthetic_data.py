from colicoords.cell import Cell, Coordinates, CellList
from colicoords.data_models import Data
import numpy as np
import mahotas as mh
from scipy.integrate import quad
from scipy.optimize import fsolve


class SynthCell(Cell):
    """
    Generate a synthetic cell.

    Parameters
    ----------
    length : :obj:`float`
        Length of the cell.
    radius : :obj:`radius`
        Radius of the cell.
    curvature : :obj:`curvature`
        Curvature of the cell. Equal to Cell.coords.a2.
    pad_width : :obj:`int`
        Number of pixels to pad around the synthetic cell.
    name : :obj:`str`, optional
        Name of the cell object
    """

    def __init__(self, length, radius, curvature, pad_width=5, name=None):
        a2 = curvature
        xl = radius + pad_width
        xr = fsolve(calc_length, length, args=(xl, a2, length)).squeeze()
        a1 = -a2 * (xr + xl)
        r = radius
        xm = (xl + xr) / 2
        y_mid = a1*xm + a2*xm**2
        a0 = 4*radius - y_mid

        y_max = a0 + a1 * xr + a2 * xr ** 2
        shape = tuple(np.ceil([y_max + 10 + r, xr + 2 * r + 10]).astype(int))
        coords = Coordinates(None, a0=a0, a1=a1, a2=a2, xl=xl, xr=xr, r=r, shape=shape, initialize=False)
        binary = coords.rc < r
        min1, max1, min2, max2 = mh.bbox(binary)
        min1p, max1p, min2p, max2p = min1 - pad_width, max1 + pad_width, min2 - pad_width, max2 + pad_width
        res = binary[min1p:max1p, 0:max2p]

        data = Data()
        data.add_data(res.astype(int), 'binary')
        super(SynthCell, self).__init__(data, name=name)
        self.coords.a0 = a0 - min1p
        self.coords.a1 = a1
        self.coords.a2 = a2
        self.coords.xl = xl
        self.coords.xr = xr
        self.coords.r = r

    def gen_radial_model_data(self, rmodel, parameters, **kwargs):
        """
        Add a image data element based on the radial distribution model `rmodel`

        Parameters
        ----------
        rmodel : :obj:`callable`
            Radial distribution model.
        parameters : :obj:`dict`
            Parameters dict to pass to rmodel
        dclass : :obj:`str`, optional
            Output data class. Default is 'fluorescence'.
        name : :obj:`str`, optional
            Name of the data element. Default is 'fluorescence'.

        """
        #todo more catchy name for this function
        num = kwargs.pop('num', 200)
        x = np.linspace(0, np.max(self.data.shape) / 1.8, num=num)
        y = rmodel(x, **parameters)[0]
        flu = np.interp(self.coords.rc, x, y)
        return flu

    def gen_storm_membrane(self, num, r_mean, r_std=None, intensity_mean=1., intensity_std=None):
        """
        Returns a STORM data element to the ``Cell`` object with localizations randomly spaced on the membrane.

        Parameters
        ----------
        num : :obj:`int`
            Number of localizations to add.
        r_mean : :obj:`float`
            Mean radial distance of localizations.
        r_std : :obj:`std`
            Standard deviation of radial distance of localizations.
        intensity_mean : :obj:`float`
            Intensity value of the localizations
        intensity_std : :obj:`float`
            If `intensity_std` is given, the intensity values are drawn from a normal distribution with centre
            `intensity_mean` and standard deviation `intensity_std`.
        name : :obj:`str`, optional
            Name of the data element. Default is 'storm'

        Returns
        -------
        storm : :class:`~colicoords.data_models.STORMTable`
            Output STORM table.
        """
        if num <= 0:
            x_res = np.array([])
            y_res = np.array([])
        else:
            # move theese definitions
            def integrant_top(t, a1, a2, r):
                return np.sqrt(1 + (a1 + 2 * a2 * t) ** 2 + ((4 * a2 ** 2 * r ** 2) / (1 + (a1 + 2 * a2 * t) ** 2) ** 2) + (
                        (4 * a2 * r) / np.sqrt(1 + (a1 + 2 * a2 * t))))

            def integrant_bot(t, a1, a2, r):
                return np.sqrt(1 + (a1 + 2 * a2 * t) ** 2 + ((4 * a2 ** 2 * r ** 2) / (1 + (a1 + 2 * a2 * t) ** 2) ** 2) - (
                        (4 * a2 * r) / np.sqrt(1 + (a1 + 2 * a2 * t))))

            top, terr = quad(integrant_top, self.coords.xl, self.coords.xr, args=(self.coords.a1, self.coords.a2, r_mean))
            bot, berr = quad(integrant_bot, self.coords.xl, self.coords.xr, args=(self.coords.a1, self.coords.a2, r_mean))

            segments_lenghts = np.array([np.pi * r_mean, top, np.pi * r_mean, bot])
            if np.any(np.isnan(segments_lenghts)):
                raise ValueError('Some NaNs found in segment lengths')

            total = np.sum(segments_lenghts)
            cumsum = np.cumsum(segments_lenghts)
            s = np.random.uniform(0, np.nextafter(total, total + 1), num)
            i = np.digitize(s, cumsum)

            s_rel = s - np.insert(cumsum, 0, 0)[i]

            x_res = np.empty_like(s_rel)
            y_res = np.empty_like(s_rel)
            new_r = np.random.normal(loc=r_mean, scale=r_std, size=num) if r_std else r_mean*np.ones(num)

            #i == 0
            th1 = np.arctan(self.coords.p_dx(self.coords.xl))
            th2 = s_rel[i == 0] / r_mean
            x_res[i == 0] = self.coords.xl + new_r[i == 0]*np.sin(-th2 - th1)
            y_res[i == 0] = self.coords.p(self.coords.xl) + new_r[i == 0]*np.cos(-th2 - th1)

            #i == 1
            t = (s_rel[i == 1] / top) * (self.coords.xr - self.coords.xl) + self.coords.xl
            x_res[i == 1] = t + new_r[i == 1] * ((self.coords.a1 + 2 * self.coords.a2 * t) /
                                                 np.sqrt(1 + (self.coords.a1 + 2 * self.coords.a2 * t) ** 2))
            y_res[i == 1] = self.coords.a0 + self.coords.a1 * t + self.coords.a2 * (t ** 2) - new_r[i == 1] * (1 / np.sqrt(1 + (self.coords.a1 + 2 * self.coords.a2 * t) ** 2))

            #i == 2
            th1 = np.arctan(self.coords.p_dx(self.coords.xr))
            th2 = s_rel[i == 2] / r_mean

            x_res[i == 2] = self.coords.xr + new_r[i == 2]*np.sin(th1 + th2)
            y_res[i == 2] = self.coords.p(self.coords.xr) - new_r[i == 2]*np.cos(th1 + th2)

            #i == 3
            t = self.coords.xr - (s_rel[i == 3] / bot) * (self.coords.xr - self.coords.xl)
            x_res[i == 3] = t + - new_r[i == 3] * ((self.coords.a1 + 2 * self.coords.a2 * t) / np.sqrt(1 + (self.coords.a1 + 2 * self.coords.a2 * t) ** 2))
            y_res[i == 3] = self.coords.a0 + self.coords.a1 * t + self.coords.a2 * (t ** 2) + new_r[i == 3] * (1 / np.sqrt(1 + (self.coords.a1 + 2 * self.coords.a2 * t) ** 2))

        storm = np.recarray((len(x_res, )), dtype=[('x', float), ('y', float), ('frame', int), ('intensity', int)])
        storm['x'] = x_res
        storm['y'] = y_res
        storm['frame'] = np.zeros_like(x_res)
        storm['intensity'] = intensity_mean*np.ones_like(x_res) if not intensity_std else \
            np.random.normal(intensity_mean, intensity_std, len(x_res))

        return storm

    def copy(self):
        """
        Make a copy of the cell object and all its associated data elements.

        This is a deep copy meaning that all numpy data arrays are copied in memory and therefore modifying the copied
        cell object does not modify the original cell object.

        Returns
        -------
        cell : :class:`~colicoords.cell.Cell`:
            Copied cell object

        """
        # todo needs testing (this is done?) arent there more properties to copy?
        new_cell = SynthCell.__new__(SynthCell)
        super(SynthCell, new_cell).__init__(self.data.copy(), name=self.name)
        #
        #
        # new_cell = SynthCell(data_object=self.data.copy(), name=self.name)
        for par in self.coords.parameters:
            setattr(new_cell.coords, par, getattr(self.coords, par))

        return new_cell

    def gen_flu_from_storm(self, storm_name='storm', sigma=1.54, sigma_std=None):
        """
        Reverse engineers a fluorescence image from STORM data.

        The image is generated by placing a gaussian at every localization.

        Parameters
        ----------
        storm_name : :obj:`str`
            Name of the STORM data element to use.
        flu_name : :obj:`str`
            Name of the fluorescence image data element to create.
        sigma : :obj:`float`
            Sigma of the gaussians to place
        sigma_std : :obj:`float`, optional
            If `sigma_std` is supplied the value of sigma used is drawn from a normal distribution with center `sigma`
            and standard deviation `sigma_std`

        Returns
        -------
        img : :class:`~numpy.ndarry`
            Generated image with fluorescent foci

        """

        xmax = self.data.shape[1]
        ymax = self.data.shape[0]
        step = 1
        xi = np.arange(step / 2, xmax, step)
        yi = np.arange(step / 2, ymax, step)

        x_coords = np.repeat(xi, len(yi)).reshape(len(xi), len(yi)).T
        y_coords = np.repeat(yi, len(xi)).reshape(len(yi), len(xi))
        storm_table = self.data.data_dict[storm_name]
        x, y = storm_table['x'], storm_table['y']

        img = np.zeros_like(x_coords)
        intensities = storm_table['intensity']
        sigma = sigma*np.ones_like(x_coords) if not sigma_std else np.random.normal(sigma, sigma_std, size=len(x_coords))
        for _sigma, _int, _x, _y in zip(sigma, intensities, x, y):
            img += _int * np.exp(-(((_x - x_coords) / _sigma) ** 2 + ((_y - y_coords) / _sigma) ** 2) / 2)

        return img


def draw_poisson(img):
    return np.random.poisson(lam=img, size=img.shape)

#https://kmdouglass.github.io/posts/modeling-noise-for-image-simulations.html
def add_readout_noise(img, noise=2):
    return img + np.random.normal(scale=noise, size=img.shape)


def calc_length(xr, xl, a2, length):
    a1 = -a2 * (xr + xl)
    l = (1 / (4 * a2)) * (
            ((a1 + 2 * a2 * xr) * np.sqrt(1 + (a1 + 2 * a2 * xr) ** 2) + np.arcsinh((a1 + 2 * a2 * xr))) -
            ((a1 + 2 * a2 * xl) * np.sqrt(1 + (a1 + 2 * a2 * xl) ** 2) + np.arcsinh((a1 + 2 * a2 * xl)))
    )

    return length-l


class SynthCellList(CellList):
    """
    Create a list of ``SynthCell`` objects.

    Parameters
    ----------
    lengths : array_like
        Array like of lengths of cells.
    radii : array_like
        Array like ot radii of cells.
    curvatures : array_like
        Array like of curvatures of cells.
    """

    def __init__(self, lengths, radii, curvatures):
        cell_list = [SynthCell(l, r, c, name='Cell_' + str(i).zfill(int(np.ceil(np.log10(len(radii)))))) for i, (l, r, c) in enumerate(zip(lengths, radii, curvatures))]
        super(SynthCellList, self).__init__(cell_list)

    def copy(self):
        out = SynthCellList.__new__(SynthCellList)
        super(SynthCellList, out).__init__([cell.copy() for cell in self])
        return out
