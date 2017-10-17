import h5py
import numpy as np


with h5py.File('testfile', 'w') as f:

    g = f.create_group('testname')
    d = np.random.rand(100, 100)
    ds = g.create_dataset('data', data=d)
    g.attrs.create('dclass', np.string_('Fluorescence'))

    g1 = f.create_group('hoidoei')

#numpy.string_("Hello")
fr = h5py.File('testfile', 'r')

print(fr)

print(fr.keys())

k1 = list(fr.keys())
print('hoi')

print(k1)
i = fr.items()
print(i)

gout = fr['testname']
b = gout.attrs.get('dclass')

print(str(b))
print(b.astype('S'))
print(type(b))
print(b.decode('UTF-8'))

for k in k1:
    print(k)
    print('wut')