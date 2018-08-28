from setuptools import setup

setup(
    setup_requires=['pbr>=1.9', 'setuptools>=17.1'],
    pbr=False,
)







# from distutils.core import setup
#
#
# setup(
#     name='colicoords',
#     version='0.0.1',
#     description='Create a coordinate system in rod-shaped cells.',
#     author='Jochem Smit',
#     author_email='j.h.smit@rug.nl',
#     packages=['colicoords', 'colicoords.gui'],
#     keywords='cell microscopy ecoli coordinates',
#     requires=['numpy', 'matplotlib', 'seaborn', 'mahotas', 'scipy', 'h5py'],
#     # extras_require={'GUI': ['pyqt>=5'],
#     #                 'CNN': ['tensorflow', 'keras']}
# )

