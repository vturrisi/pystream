def configuration(parent_package='',top_path=None):
    import os
    from os.path import join

    import numpy
    from numpy.distutils.misc_util import Configuration


    libs = []
    if os.name == 'posix':
        libs.append('m')

    config = Configuration('numeric_estimators', parent_package, top_path)

    dirs = [numpy.get_include(), '.']

    config.add_extension(name='gaussian_estimator',
                         sources=['gaussian_estimator.pyx'],
                         libraries=libs,
                         include_dirs=dirs)

    config.make_config_py() # installs __config__.py
    return config


if __name__ == '__main__':
    print('This is the wrong setup.py file to run')
