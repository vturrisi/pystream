from setuptools import setup, find_packages
from os.path import basename, splitext
from glob import glob
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


extensions = [
    Extension("*", ['pystream/algorithms/base/*.pyx'],
              include_dirs=[numpy.get_include(), '.']),

    Extension("*", ['pystream/algorithms/base/*/*.pyx'],
              include_dirs=[numpy.get_include(), '.']),

    Extension("*", ['pystream/algorithms/change_detectors/*.pyx'],
              include_dirs=[numpy.get_include(), '.']),

    Extension("*", ['pystream/algorithms/ensembles/*.pyx'],
              include_dirs=[numpy.get_include(), '.']),

    Extension("*", ['pystream/evaluation/*.pyx'],
              include_dirs=[numpy.get_include(), '.']),

    Extension("*", ['pystream/utils/*.pyx'],
              include_dirs=[numpy.get_include(), '.']),
]

ext_modules = cythonize(extensions)

setup(name='pystream',
      version='1.0',
      description='Pystream',
      url='http://github.com/vturrisi/pystream',
      author='Victor Turrisi',
      license='MIT',
      packages=find_packages(exclude=('tests', 'evaluate')),
      ext_modules=ext_modules,
      setup_requires=['cython>=0.x'],
      include_package_data=True,
      zip_safe=False)
