def configuration(parent_package='',top_path=None):
    import os
    from os.path import join

    import numpy
    from numpy.distutils.misc_util import Configuration


    libs = []
    if os.name == 'posix':
        libs.append('m')

    config = Configuration('statistics', parent_package, top_path)


    dirs = [numpy.get_include(), '.']

    config.add_extension(name='tree_stats',
                         sources=['tree_stats.pyx'],
                         libraries=libs,
                         include_dirs=dirs)

    config.add_extension(name='value_stats',
                         sources=['value_stats.pyx'],
                         libraries=libs,
                         include_dirs=dirs)

    config.make_config_py() # installs __config__.py
    return config


if __name__ == '__main__':
    print('This is the wrong setup.py file to run')





# import os

# import numpy
# from numpy.distutils.misc_util import Configuration
# from Cython.Distutils import build_ext


# def configuration(parent_package="", top_path=None):
#     config = Configuration("statistics", parent_package, top_path)
#     libraries = []
#     if os.name == 'posix':
#         libraries.append('m')
#     config.add_extension("tree_stats",
#                          sources=["tree_stats.pyx"],
#                          include_dirs=[numpy.get_include()],
#                          libraries=libraries,
#                          extra_compile_args=["-O3"])
#     config.add_extension("value_stats",
#                          sources=["value_stats.pyx"],
#                          include_dirs=[numpy.get_include()],
#                          libraries=libraries,
#                          extra_compile_args=["-O3"])
#     return config

# if __name__ == "__main__":
#     from numpy.distutils.core import setup
#     setup(**configuration().todict(),
#           cmdclass={'build_ext': build_ext},
#           script_args=['build_ext'],
#           options={'build_ext': {'inplace': True, 'force': True}})
