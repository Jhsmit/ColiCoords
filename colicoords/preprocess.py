import mahotas as mh
import numpy as np
from colicoords.cell import Cell, CellList


def data_to_cells(input_data, pad_width=3, cell_frac=0.5, rotate='binary', verbose=True):
    assert 'binary' in input_data.dclasses

    vprint = print if verbose else lambda *a, **k: None
    cell_list = []
    i_fill = int(np.ceil(np.log10(len(input_data))))
    for i, data in enumerate(input_data):
        binary = data.binary_img
        if (binary > 0).mean() > cell_frac or binary.mean() == 0.:
            vprint('Image {} {}: Too many or no cells'.format(binary.name, i))
            continue

        # Iterate over all cells in the image
        l_fill = int(np.ceil(np.log10(len(np.unique(binary)))))
        for l in np.unique(binary)[1:]:
            selected_binary = (binary == l).astype('int')
            min1, max1, min2, max2 = mh.bbox(selected_binary)
            min1p, max1p, min2p, max2p = min1 - pad_width, max1 + pad_width, min2 - pad_width, max2 + pad_width

            try:
                assert min1p > 0 and min2p > 0 and max1p < data.shape[0] and max2p < data.shape[1]
            except AssertionError:
                vprint('Cell {} on image {} {}: on the edge of the image'.format(l, binary.name, i))
                continue
            try:
                assert len(np.unique(binary[min1p:max1p, min2p:max2p])) == 2
            except AssertionError:
                vprint('Cell {} on image {} {}: multiple cells per selection'.format(l, data.binary_img.name, i))
                continue

            output_data = data[min1p:max1p, min2p:max2p].copy()
            output_data.binary_img //= output_data.binary_img.max()

            # Calculate rotation angle and rotate selections
            theta = output_data.data_dict[rotate].orientation if rotate else 0
            rotated_data = output_data.rotate(theta)


            #Make cell object and add all the data
            #todo change cell initation and data adding interface
            c = Cell(rotated_data)

            c.name = 'img{}c{}'.format(str(i).zfill(i_fill), str(l).zfill(l_fill))
            cell_list.append(c)

    return CellList(cell_list)