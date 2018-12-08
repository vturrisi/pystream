def configuration(parent_package='',top_path=None):
    import os

    import numpy
    from numpy.distutils.misc_util import Configuration


    libs = []
    if os.name == 'posix':
        libs.append('m')

    config = Configuration('ensembles', parent_package, top_path)

    dirs = [numpy.get_include(), '.']

    depends = ['../change_detectors/*.pyx',
               '../change_detectors/*.pxd']

    config.add_extension(name='arf',
                         sources=['arf.pyx'],
                         libraries=libs,
                         include_dirs=dirs,
                         depends=depends)

    config.add_extension(name='leveraging_bagging',
                         sources=['leveraging_bagging.pyx'],
                         libraries=libs,
                         include_dirs=dirs,
                         depends=depends)

    config.add_extension(name='oaue',
                         sources=['oaue.pyx'],
                         libraries=libs,
                         include_dirs=dirs)

    config.add_extension(name='ozabagging',
                         sources=['ozabagging.pyx'],
                         libraries=libs,
                         include_dirs=dirs)

    config.add_extension(name='ozaboosting',
                         sources=['ozaboosting.pyx'],
                         libraries=libs,
                         include_dirs=dirs)

    config.make_config_py() # installs __config__.py
    return config


if __name__ == '__main__':
    print('This is the wrong setup.py file to run')
