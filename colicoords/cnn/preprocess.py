import numpy as np
from keras.utils import Sequence
import scipy


class ImgSequence(Sequence):
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
    """Normalized a stack of images between 0 and 1

    Args:
        img_stack (:class:`~numpy.ndarray`): Input image stack (shape z, w, h)

    Returns:
        :class:`~numpy.ndarray` Normalized stack of images

    """

    img_float = img_stack.astype(float)
    mins, maxes = img_float.min(axis=(1,2)), img_float.max(axis=(1,2))
    norm_stack = (img_float - mins[:, np.newaxis, np.newaxis]) / (maxes - mins)[:, np.newaxis, np.newaxis]

    return norm_stack


def resize_stack(img_stack, factor, img_type=None):
    """Resize a stack of images with

    Args:
        img_stack (:class:`~numpy.ndarray`): Input image stack (shape z, w, h)
        factor (:obj:`float`): Images are resized by this factor. Width and 
            height dimensions are increased by the value of factor.

    Returns:
        :class:`~numpy.ndarray` Resized stack of images

    """
    zoom_stack = np.stack([scipy.ndimage.interpolation.zoom(img, factor) for img in img_stack])

    if img_type == 'binary':
        mins = np.min(zoom_stack, axis=(1,2))
        zoom_stack = (zoom_stack > mins[:, np.newaxis, np.newaxis]).astype(int)

    return zoom_stack
