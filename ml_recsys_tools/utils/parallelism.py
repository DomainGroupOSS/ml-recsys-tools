import numpy as np
import pandas as pd
from tqdm import tqdm

import multiprocessing
from multiprocessing.pool import ThreadPool, Pool

N_CPUS = multiprocessing.cpu_count()

def batch_generator(iterable, n=1):
    # https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def map_batches_multiproc(func, iterable, chunksize,
                          pbar=None, threads_per_cpu=1.0, multiproc_mode='threads'):

    if len(iterable) <= chunksize:
        return [func(iterable)]

    else:
        n_threads = int(threads_per_cpu*N_CPUS)
        pool_type = ThreadPool if multiproc_mode=='threads' else Pool
        with pool_type(n_threads) as pool:
            if pbar is None:
                return pool.map(func, batch_generator(iterable, n=chunksize))
            else:
                return list(tqdm(
                    pool.imap(func, batch_generator(iterable, n=chunksize)),
                    total=int(len(iterable) / chunksize),
                    desc=pbar,
                    mininterval=10.0,
                    maxinterval=100.0,
                    ascii=True))


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