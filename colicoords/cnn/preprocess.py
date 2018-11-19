import numpy as np
from keras.utils import Sequence
import scipy
from scipy.signal import medfilt, wiener
import itertools
import mahotas as mh


def identity(arr):
    """Identity operation on array"""
    return arr


def flip_horizontal(arr):
    """Flip array along horizontal axis"""
    return arr[::-1, ::]


def flip_vertical(arr):
    """Flip array along vertical axis"""
    return arr[:, ::-1]


def transpose(arr):
    """Transpose array"""
    return arr.T


def norm_minmax(arr):
    """Norm array by min / max"""
    return arr - arr.min() / arr.max() - arr.min()


def norm_zscore(arr):
    """Zero-center and norm array by standard deviation"""
    arr -= arr.mean()
    arr /= arr.std()
    return arr


def norm_hampel(arr):
    """Norm array by Hampel's method"""
    #todo reference
    #Hampel estimators
    return 0.5 * (np.tanh(0.01 * (arr - arr.mean()) / arr.std()) + 1)


def gaussian_filter(arr, sigma=1):
    """Filter array with a gaussian kernel"""
    return mh.gaussian_filter(arr, sigma=sigma)


def wiener_filter(arr, mysize=None, noise=None):
    """Apply a Wiener filter to array"""
    return wiener(arr, mysize=mysize, noise=noise)


def median_filter(arr, kernel_size=3):
    """Median filter an array"""
    return medfilt(arr, kernel_size=kernel_size)


AUGMENTATIONS = {
    'flip_horizontal': flip_horizontal,
    'flip_vertical': flip_vertical,
    'transpose': transpose
}

STANDARDIZATIONS = {
    'minmax': norm_minmax,
    'zscore': norm_zscore,
    'hampel': norm_hampel
}


class BaseSequence(Sequence):
    """
    Base object for custom data preprocessing, standardization and augmentation.

    Parameters
    ----------
    x_arr : :class:`~numpy.ndarray`
        Input training (brightfield) array
    y_arr : :class:`~numpy.ndarray`
        Input prediction (binary) array
    index_list : array_like
        List of tuples with (index, dictionary), where index is the index of `x_arr` and dictionary contains operations
        for preprocessing, standardization and augmentation.
    shuffle : :obj:`bool`
        If `True` the data will be shuffled after each epoch.
    batch_size : :obj:`int`
        Number of images per batch to feed to the neural network during training.
    """

    def __init__(self, x_arr, y_arr, index_list, shuffle=True, batch_size=8):
        self.x_arr = x_arr
        self.y_arr = y_arr
        self.index_list = np.array(index_list)
        self._shuffle = shuffle
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.index_list)/self.batch_size))

    def __getitem__(self, item):
        items = self.index_list[item*self.batch_size:(item+1)*self.batch_size]
        x, y = np.rollaxis(np.array([self._prepare_data(item) for item in items]), 1, 0)

        return np.expand_dims(x, -1), np.expand_dims(y, -1)

    def _prepare_data(self, item):
        """Apply operations and return array as specified by `item`"""
        idx, operations = item
        x = self.x_arr[idx]
        y = self.y_arr[idx]
        for op in operations['standardizaton']:
            x = op(x)
        for op in operations['augmentations']:
            x = op(x)
            y = op(y)

        return [x, y]

    def on_epoch_end(self):
        if self._shuffle:
            self.shuffle()

    def shuffle(self):
        """Shuffles the index list and thereby the data"""
        np.random.shuffle(self.index_list)


class AugmentedImgSequence(BaseSequence):
    """
    Object for custom data preprocessing, standardization and augmentation.

    Parameters
    ----------
    x_arr : :class:`~numpy.ndarray`
        Input training (brightfield) array
    y_arr : :class:`~numpy.ndarray`
        Input prediction (binary) array
    standardization : :obj:`list`
        List of standardization operations to apply. Elements must be either callable or one of 'minmax', 'zscore' or
        `hempel`
    augmentation : :obj:`list`
        List of augmentation operations to apply.  Elements must be either callable or one of 'flip_horizontal',
        'flip_vertical' or 'transpose'.
    shuffle : :obj:`bool`
        If `True` the data will be shuffled after each epoch.
    batch_size : :obj:`int`
        Number of images per batch to feed to the neural network during training.
    """
    def __init__(self, x_arr, y_arr, standardization=None,
                 augmentation=None, shuffle=True, batch_size=8):
        standardization = [] if standardization is None else standardization
        augmentation = [] if augmentation is None else augmentation
        st_list = []
        for elem in standardization:
            if type(elem) == str:
                st_list.append(STANDARDIZATIONS[elem])
            elif callable(elem):
                st_list.append(elem)
            else:
                raise TypeError('Invalid standardization type')

        ag_list = []
        for elem in augmentation:
            if type(elem) == str:
                ag_list.append(AUGMENTATIONS[elem])
            elif callable(elem):
                ag_list.append(elem)
            else:
                raise TypeError('Invalid standardization type')

        idx = range(len(x_arr))
        ag_permutations = list(itertools.product(*[[identity, aug] for aug in ag_list]))

        index_list = [(i, {'standardizaton': st_list, 'augmentations': p}) for i, p in itertools.product(idx, ag_permutations)]

        super(AugmentedImgSequence, self).__init__(x_arr, y_arr, index_list, shuffle=shuffle, batch_size=batch_size)

    #todo smart training
    #todo check rounding 20181107 example
    #https://stackoverflow.com/questions/25889637/how-to-use-k-fold-cross-validation-in-a-neural-network#25897087
    def val_split(self, frac, random=True, offset=0):
        """
        Split the data into two parts for validation and training.

        frac : :obj:`float`
            Fraction of the data to use for validation.
        random : :obj:`bool`
            If `True` the returned validation data is randomy but equidistantly selected.
        offset : :obj:`int`
            Index of where to start selecting validation data.

        val_seq : :class:`BaseSequence`
            ``BaseSequence`` object with indices selected for validation.
        train_eq : :class:`BaseSequence`
            ``BaseSequence`` object with indices selected for training.
        """
        step = int(np.round(1/frac))
        idx_val = np.arange(step-1, len(self.index_list), step) + offset
        if random:
            idx_val += np.random.random_integers(0, step-1, len(idx_val))

        if idx_val.max() > len(self.index_list):
            idx_val[-1] = len(self.index_list) - 1
            print("Warning, index out of bounds, set to last element")

        assert idx_val.max() < len(self.index_list)
        assert len(np.unique(idx_val)) == len(idx_val)

        val_indices = self.index_list[idx_val]
        train_idices = np.delete(self.index_list, idx_val, axis=0)

        val_seq = BaseSequence(self.x_arr, self.y_arr, val_indices, shuffle=self._shuffle, batch_size=self.batch_size)
        train_seq = BaseSequence(self.x_arr, self.y_arr, train_idices, shuffle=self._shuffle, batch_size=self.batch_size)

        return val_seq, train_seq


# todo wrapper for defaults
class DefaultImgSequence(AugmentedImgSequence):
    """
    Object for custom data preprocessing, standardization and augmentation.

    The object has default values for `standardizatin` and `augmentation`

    Parameters
    ----------
    x_arr : :class:`~numpy.ndarray`
        Input training (brightfield) array
    y_arr : :class:`~numpy.ndarray`
        Input prediction (binary) array
    standardization : :obj:`list`
        List of standardization operations to apply. Elements must be either callable or one of 'minmax', 'zscore' or
        'hempel'. Defaults to `'hempel'`
    augmentation : :obj:`list`
        List of augmentation operations to apply.  Elements must be either callable or one of 'flip_horizontal',
        'flip_vertical' or 'transpose'.  Defaults to `['flip_horizontal', 'flip_vertical', 'transpose']`.
    shuffle : :obj:`bool`
        If `True` the data will be shuffled after each epoch.
    batch_size : :obj:`int`
        Number of images per batch to feed to the neural network during training.
    """
    def __init__(self, x_arr, y_arr, standardization=None, augmentation=None, shuffle=True, batch_size=8):
        standardization = ['hampel'] if standardization is None else standardization
        augmentation = ['flip_horizontal', 'flip_vertical', 'transpose'] if augmentation is None else augmentation
        super(DefaultImgSequence, self).__init__(x_arr, y_arr, standardization=standardization,
                                                 augmentation=augmentation, shuffle=shuffle, batch_size=batch_size)


class ImgSequence(Sequence):
    """Deprecated, will be removed"""
    def __init__(self, x_arr, y_arr, batch_size=10, shuffle=True):
        assert x_arr.shape == y_arr.shape
        self.x_arr = x_arr
        self.y_arr = y_arr
        self.batch_size=batch_size
        self.shuffle = shuffle
        
        self.operations = [
            lambda x: x,
            lambda x: np.transpose(x, axes=(0, 2, 1)),
            lambda x: x[:, :, ::-1],
            lambda x: x[:, ::-1, :],
            lambda x: np.transpose(x, axes=(0, 2, 1))[:, :, ::-1],
            lambda x: x[:, ::-1, ::-1],
            lambda x: np.transpose(x, axes=(0, 2, 1))[:, ::-1, :],
            lambda x: np.transpose(x, axes=(0, 2, 1))[:, ::-1, ::-1]
        ]
        
    def __len__(self):
        return int(np.floor(len(self.x_arr)*8/self.batch_size))
    
    def __getitem__(self, idx):
        start = idx*self.batch_size
        stop = (idx+1)*self.batch_size
        
        input_len = len(self.x_arr)
        start_op = start // input_len
        stop_op = stop // input_len

        start_idx = start % input_len
        stop_idx = stop % input_len

        if stop_op == 8:
            x_arr = self.operations[start_op](self.x_arr[start_idx:])
            y_arr = self.operations[start_op](self.y_arr[start_idx:])
        
        elif start_op == stop_op:
            x_arr = self.operations[start_op](self.x_arr[start_idx:stop_idx])
            y_arr = self.operations[start_op](self.y_arr[start_idx:stop_idx])

        else:
            x_arr_1 = self.operations[start_op](self.x_arr[start_idx:])
            y_arr_1 = self.operations[start_op](self.y_arr[start_idx:]) 

            x_arr_2 = self.operations[stop_op](self.x_arr[0:stop_idx])
            y_arr_2 = self.operations[stop_op](self.y_arr[0:stop_idx])
            
            x_arr = np.concatenate((x_arr_1, x_arr_2))
            y_arr = np.concatenate((y_arr_1, y_arr_2))
            
        if self.shuffle:
            p = np.random.permutation(self.batch_size)
            x_arr = x_arr[p]
            y_arr = y_arr[p]

        return np.expand_dims(x_arr, -1), np.expand_dims(y_arr, -1)


def norm_stack(img_stack):
    """
    Deprecated, will soon be removed.

    """

    img_float = img_stack.astype(float)
    mins, maxes = img_float.min(axis=(1,2)), img_float.max(axis=(1,2))
    norm_stack = (img_float - mins[:, np.newaxis, np.newaxis]) / (maxes - mins)[:, np.newaxis, np.newaxis]

    return norm_stack


def resize_stack(img_stack, factor, img_type=None):
    """
    Resize a stack of images by a constant factor

    Parameters
    -----------
    img_stack : :class:`~numpy.ndarray`
        Input image stack (shape z, w, h)
    factor : :obj:`float`
        Images are resized by this factor. Width and height dimensions are increased by the value of factor.

    Returns
    -------
    resized_stack : :class:`~numpy.ndarray`
        Resized stack of images.

    """
    resized_stack = np.stack([scipy.ndimage.interpolation.zoom(img, factor) for img in img_stack])

    if img_type == 'binary':
        mins = np.min(resized_stack, axis=(1,2))
        resized_stack = (resized_stack > mins[:, np.newaxis, np.newaxis]).astype(int)

    return resized_stack
