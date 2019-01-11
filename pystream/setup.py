def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration


    config = Configuration('pystream', parent_package, top_path)

    config.add_subpackage('algorithms')
    config.add_subpackage('evaluation')
    config.add_subpackage('utils')

    config.make_config_py() # installs __config__.py
    return config


if __name__ == '__main__':
    print('This is the wrong setup.py file to run')