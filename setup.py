from numpy.distutils.core import setup


def configuration(parent_package='',top_path=None):
    from Cython.Build import cythonize
    from numpy.distutils.misc_util import Configuration


    config = Configuration(None, parent_package, top_path)

    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('pystream')

    config.ext_modules = cythonize(config.ext_modules, nthreads=4)
    return config


setup(name='pystream',
      version='1.1',
      description='Pystream',
      url='http://github.com/vturrisi/pystream',
      author='Victor Turrisi',
      license='MIT',
      **configuration().todict(),
      setup_requires=['cython>=0.x'],
      install_requires=['numpy>=1.14.1', 'pandas>=0.20.0', 'cython>=0.x'],
      include_package_data=True,
      zip_safe=False)