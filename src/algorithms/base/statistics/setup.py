from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np


ext_modules = cythonize([
                        (Extension("tree_stats",
                         sources=["tree_stats.pyx"],
                         include_dirs=[np.get_include()], )),

                        (Extension("value_stats",
                         sources=["value_stats.pyx"],
                         include_dirs=[np.get_include()], )),
                        ])

if __name__ == "__main__":
    setup(ext_modules=ext_modules)
