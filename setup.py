from setuptools import setup, find_packages
from os.path import basename, splitext
from glob import glob


setup(name='pystream',
      version='0.1',
      description='Pystream',
      url='http://github.com/vturrisi/pystream',
      author='Victor Turrisi',
      license='MIT',
      package_dir={'': 'pystream'},
      include_package_data=True,
      zip_safe=False)
