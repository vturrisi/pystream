def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('algorithms', parent_package, top_path)

    config.add_subpackage('base')
    config.add_subpackage('change_detectors')
    config.add_subpackage('ensembles')

    config.make_config_py() # installs __config__.py
    return config

if __name__ == '__main__':
    print('This is the wrong setup.py file to run')