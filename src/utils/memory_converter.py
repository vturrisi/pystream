def convert_size(size, unit='B', target_unit='MB'):
    """
    Basic memory size converter to multiple units

    Args:
        size: total size of the object
        unit: original format
        target_unit: target_unit

    Returns:
        the size (float) in the target_unit format
    """

    possible_units = ('B', 'KB', 'MB', 'GB')
    assert unit in possible_units
    assert target_unit in possible_units

    factor = possible_units.index(target_unit) - possible_units.index(unit)
    return size / (1024 ** factor)
