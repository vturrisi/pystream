try:
    from . import algorithms, evaluation, utils
except ImportError:
    msg = 'Error importing pystream: do not import from its source directory'
    raise ImportError(msg)

__all__ = ['algorithms', 'evaluation', 'utils']