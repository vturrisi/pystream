import pandas as pd
cimport numpy as np
import time


def instance_gen(data_or_fname, dict dtype, chunksize=None):
    """
    Auxiliary generator to read from a file.

    Args:
        data_or_fname: file object or file name
        dtype: dictionary containing the data types
            of all attributes in the dataset
        chunksize: optinal parameter if the given file is too large
            to hold on memory (pandas DataFrames use a lot of memory)

    Returns:
        generator where each value corresponds to an instance
            in the tuple format index, X, y tuple
    """
    cdef int i = 0
    cdef np.ndarray row

    if chunksize is None:
        chunksize = 500_000
    stream = pd.read_csv(data_or_fname, dtype=dtype,
                         iterator=True, chunksize=chunksize)
    for chunk in stream:
        for row in chunk.values:
            yield i, (row[:-1], int(row[-1]))
            i += 1


def instance_gen_log(data_or_fname, dict dtype, chunksize=None):
    """
    Auxiliary generator to read from a file.

    Args:
        data_or_fname: file object or file name
        dtype: dictionary containing the data types
            of all attributes in the dataset
        chunksize: optinal parameter if the given file is too large
            to hold on memory (pandas DataFrames use a lot of memory)

    Returns:
        generator where each value corresponds to an instance
            in the tuple format index, X, y tuple
    """
    cdef int i = 0
    cdef np.ndarray row

    if chunksize is None:
        chunksize = 500_000
    print(f'{str(time.time())[:14].replace(".", "")},started load_dataset')
    stream = pd.read_csv(data_or_fname, dtype=dtype,
                         iterator=True, chunksize=chunksize)
    for chunk in stream:
        print(f'{str(time.time())[:14].replace(".", "")},ended load_dataset')
        for row in chunk.values:
            yield i, (row[:-1], int(row[-1]))
            i += 1
