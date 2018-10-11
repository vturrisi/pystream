import os

import numpy
from numpy.distutils.misc_util import Configuration
from Cython.Distutils import build_ext


def configuration(parent_package="", top_path=None):
    config = Configuration("ensembles", parent_package, top_path)
    libraries = []
    if os.name == 'posix':
        libraries.append('m')
    config.add_extension("arf",
                         sources=["arf.pyx"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args=["-O3"])
    config.add_extension("leveraging_bagging",
                         sources=["leveraging_bagging.pyx"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args=["-O3"])
    config.add_extension("oaue",
                         sources=["oaue.pyx"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args=["-O3"])
    config.add_extension("ozabagging",
                         sources=["ozabagging.pyx"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args=["-O3"])
    config.add_extension("ozaboosting",
                         sources=["ozaboosting.pyx"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args=["-O3"])
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration().todict(),
          cmdclass={'build_ext': build_ext},
          script_args=['build_ext'],
          options={'build_ext': {'inplace': True, 'force': True}})
