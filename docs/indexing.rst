Indexing
========

(section under construction)

.. code-block:: python

  data_slice = data[5:10, 0:100, 0:100]
  print(data.shape)
  print(data_slice.shape)
  >>> (20, 512, 512)
  >>> (20, 100, 100)

This particular slicing operation selects images 5 through 10 and takes the upper left 100x100 square. STORM data is
automatically sliced accordingly if its present in the data class. This slicing functionality is used by the
``data_to_cells`` method to obtain single-cell objects.

