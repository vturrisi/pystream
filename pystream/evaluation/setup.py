def configuration(parent_package='',top_path=None):
    import os

    import numpy
    from numpy.distutils.misc_util import Configuration

    libs = []
    if os.name == 'posix':
        libs.append('m')

    config = Configuration('evaluation', parent_package, top_path)

    dirs = [numpy.get_include(), '.']

    config.add_extension(name='evaluate_prequential',
                         sources=['evaluate_prequential.pyx'],
                         libraries=libs,
                         include_dirs=dirs)

    config.add_extension(name='performance_statistics',
                         sources=['performance_statistics.pyx'],
                         libraries=libs,
                         include_dirs=dirs)

    config.make_config_py() # installs __config__.py
    return config


if __name__ == '__main__':
    print('This is the wrong setup.py file to run')
