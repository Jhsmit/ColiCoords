import numpy as np
from keras.utils import Sequence
import scipy


class MySequence(Sequence):
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


def zoom_norm(bf_img):
    #todo refactor
    bf_zoom = np.stack([scipy.ndimage.interpolation.zoom(bf, 0.5) for bf in bf_img])

    mins, maxes = bf_zoom.min(axis=(1,2)), bf_zoom.max(axis=(1,2))
    bf_norm = (bf_zoom - mins[:, np.newaxis, np.newaxis]) / (maxes - mins)[:, np.newaxis, np.newaxis]
    
    return bf_norm


def zoom_norm_binary(bin_img):
    #todo refactor
    bin_zoom = np.stack([scipy.ndimage.interpolation.zoom(bf, 0.5) for bf in bin_img])

    mins = np.min(bin_zoom, axis=(1,2))
    bin_final = (bin_zoom > mins[:, np.newaxis, np.newaxis]).astype(int)
    
    return bin_final
