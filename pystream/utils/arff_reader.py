from io import StringIO
from itertools import takewhile
from .stream_gen import instance_gen


def read_arff(arff_file):
    """
    Auxiliary function to read an arff file

    Args:
        arff_file: file path

    Returns:
        generator using the instance_gen util function
        dtype: tuple of dtypes for numpy/pandas
        types: tuple of types in a more useful format
            (float for numeric attributes and
             a tuple of all possible values for discrete attributes)
        classes: a tuple of the possible classes (only supports numbers)
    """

    with open(arff_file) as f:
        meta_lines = list(takewhile(lambda line: '@data' not in line,
                          (line.strip() for line in f)))
        data = StringIO(f.read())
    meta_lines = list(filter(None, meta_lines))
    dtype, types = {}, []
    for line in meta_lines[:-1]:
        if '@attribute' not in line:
            continue
        _, name, type_ = line.split()
        if type_ == 'numeric':
            dtype[name] = float
            types.append(float)
        else:
            values = tuple(type_.split('{')[-1].split('}')[0].split(','))
            dtype[name] = object
            types.append(values)
    classes = tuple(int(i) for i in
                    meta_lines[-1].split('{')[-1].split('}')[0].split(','))
    return instance_gen(data, dtype), dtype, types, classes


def read_arff_meta(meta_file):
    """
    Auxiliary function to read a meta_file (the same as an arff header)

    Args:
        meta_file: file path

    Returns:
        dtype: tuple of dtypes for numpy/pandas
        types: tuple of types in a more useful format
            (float for numeric attributes and
             a tuple of all possible values for discrete attributes)
        classes: a tuple of the possible classes (only supports numbers)
    """

    with open(meta_file) as f:
        meta_lines = list(line.strip() for line in f)
    meta_lines = list(filter(None, meta_lines))
    dtype, types = {}, []
    for line in meta_lines[:-1]:
        if '@attribute' not in line:
            continue
        _, name, type_ = line.split()
        if type_ == 'numeric':
            dtype[name] = float
            types.append(float)
        else:
            values = tuple(type_.split('{')[-1].split('}')[0].split(','))
            dtype[name] = object
            types.append(values)
    classes = tuple(int(i) for i in
                    meta_lines[-1].split('{')[-1].split('}')[0].split(','))
    return dtype, types, classes
