import numpy as np
import pandas as pd
from itertools import islice
import multiprocessing
from multiprocessing.pool import ThreadPool, Pool

N_CPUS = multiprocessing.cpu_count()


def batch_generator(iterable, n=1):
    if hasattr(iterable, '__len__'):
        # https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    elif hasattr(iterable, '__next__'):
        # https://stackoverflow.com/questions/1915170/split-a-generator-iterable-every-n-items-in-python-splitevery
        i = iter(iterable)
        piece = list(islice(i, n))
        while piece:
            yield piece
            piece = list(islice(i, n))
    else:
        raise ValueError('Iterable is not iterable?')


def map_batches_multiproc(func, iterable, chunksize, multiproc_mode='threads',
                          n_threads=None, threads_per_cpu=1.0):
    if n_threads is None:
        n_threads = int(threads_per_cpu * N_CPUS)

    if hasattr(iterable, '__len__') and len(iterable) <= chunksize:
        return [func(iterable)]

    with pool_type(multiproc_mode)(n_threads) as pool:
        batches = batch_generator(iterable, n=chunksize)
        return list(pool.imap(func, batches))


def pool_type(parallelism_type):
    if 'process' in parallelism_type.lower():
        return Pool
    elif 'thread' in parallelism_type.lower():
        return ThreadPool
    else:
        raise ValueError('Unsupported value for "parallelism_type"')


def parallelize_dataframe(df, func, n_partitions=N_CPUS, parallelism_type='process'):
    # with Pool(n_partitions) as pool:
    #     return pd.concat(pool.map(func, np.array_split(df, n_partitions)))
    df_split = np.array_split(df, n_partitions)
    with pool_type(parallelism_type)(n_partitions) as pool:
        res = pool.map(func, df_split)
    df = pd.concat(res)
    return df


def parallelize_array(arr, func, n_partitions=N_CPUS, parallelism_type='process'):
    # with Pool(n_partitions) as pool:
    #     return np.concatenate(pool.map(func, np.array_split(arr, n_partitions)))
    arr_split = np.array_split(arr, n_partitions)
    with pool_type(parallelism_type)(n_partitions) as pool:
        res = pool.map(func, arr_split)
    arr = np.concatenate(res)
    return arr
