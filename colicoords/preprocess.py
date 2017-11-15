import mahotas as mh
import numpy as np
from colicoords.cell import Cell, CellList


def data_to_cells(input_data, pad_width=3, cell_frac=0.5, rotate='binary'):
    assert 'binary' in input_data.dclasses
    cell_list = CellList()
    for i, data in enumerate(input_data):
        print('storm data size before 2d slicing', data.storm_storm.size)
        binary = data.binary_img
        if (binary > 0).mean() > cell_frac or binary.mean() == 0.:
            print('Image {} {}: Too many or no cells'.format(binary.name, i))
            continue

        # Iterate over all cells in the image
        for l in np.unique(binary)[1:]:
            selected_binary = (binary == l).astype('int')
            min1, max1, min2, max2 = mh.bbox(selected_binary)
            min1p, max1p, min2p, max2p = min1 - pad_width, max1 + pad_width, min2 - pad_width, max2 + pad_width

            try:
                assert min1p > 0 and min2p > 0 and max1p < data.shape[0] and max2p < data.shape[1]
            except AssertionError:
                print('Cell {} on image {} {}: on the edge of the image'.format(l, binary.name, i))
                continue
            try:
                assert len(np.unique(binary[min1p:max1p, min2p:max2p])) == 2
            except AssertionError:
                print('Cell {} on image {} {}: multiple cells per selection'.format(l, data.binary_img.name, i))
                continue

            print('input', min1p, max1p, min2p, max2p)
            output_data = data[min1p:max1p, min2p:max2p].copy()
            output_data.binary_img //= output_data.binary_img.max()

            # Calculate rotation angle and rotate selections
            theta = output_data.data_dict[rotate].orientation if rotate else 0
            print('theta', theta)
            #theta -= 180
            rotated_data = output_data.rotate(theta)

            #Make cell object and add all the data
            #todo change cell initation and data adding interface
            c = Cell(rotated_data)

            c.name = 'img{}c{}'.format(str(i).zfill(3), str(l).zfill(3))
            cell_list.append(c)

    return cell_list