def configuration(parent_package='',top_path=None):
    import os
    from os.path import join

    import numpy
    from numpy.distutils.misc_util import Configuration


    libs = []
    if os.name == 'posix':
        libs.append('m')

    config = Configuration('base', parent_package, top_path)

    config.add_subpackage('nominal_counters')
    config.add_subpackage('numeric_estimators')
    config.add_subpackage('statistics')

    dirs = [numpy.get_include(), '.']

    dependencies = ['nominal_counters', 'numeric_estimators']
    depends = [join(d, '*.pyx') for d in dependencies] +\
        [join(d, '*.pxd') for d in dependencies]

    config.add_extension(name='vfdt',
                         sources=['vfdt.pyx'],
                         libraries=libs,
                         include_dirs=dirs,
                         depends=depends)

    config.add_extension(name='svfdt',
                         sources=['svfdt.pyx'],
                         libraries=libs,
                         include_dirs=dirs,
                         depends=depends + ['vfdt.pyx'])

    config.add_extension(name='svfdt_ii',
                         sources=['svfdt_ii.pyx'],
                         libraries=libs,
                         include_dirs=dirs,
                         depends=depends + ['vfdt.pyx',
                                            'svfdt.pyx'])

    config.make_config_py() # installs __config__.py
    return config


if __name__ == '__main__':
    print('This is the wrong setup.py file to run')
