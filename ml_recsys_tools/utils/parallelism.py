from multiprocessing import Process, Queue

import numpy as np
import pandas as pd
from itertools import islice
import multiprocessing
from multiprocessing.pool import ThreadPool, Pool

from ml_recsys_tools.utils.logger import simple_logger as logger


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
    df = pd.concat(res, sort=False)
    return df


def parallelize_array(arr, func, n_partitions=N_CPUS, parallelism_type='process'):
    # with Pool(n_partitions) as pool:
    #     return np.concatenate(pool.map(func, np.array_split(arr, n_partitions)))
    arr_split = np.array_split(arr, n_partitions)
    with pool_type(parallelism_type)(n_partitions) as pool:
        res = pool.map(func, arr_split)
    arr = np.concatenate(res)
    return arr


def non_daemonic_process_pool_map(func, jobs, n_workers, timeout_per_job=None):
    """
    function for calculating in parallel a function that may not be run
    a in a regular pool (due to forking processes for example)

    :param func: a function that accepts one input argument
    :param jobs: a list of input arguments to func
    :param n_workers: number of parallel workers
    :param timeout_per_job: timeout for processing a single job
    :return: list of results in the order of the "jobs" list
    """

    END_TOKEN = 'END'
    q_in = Queue()
    q_out = Queue()

    def queue_worker(q_in, q_out):
        arg_in = q_in.get()
        while arg_in != END_TOKEN:
            try:
                result = func(arg_in)
            except Exception as e:
                logger.exception(e)
                logger.error(f'Queue worker failed on input: {arg_in}, with {str(e)}')
                result = None
            q_out.put((arg_in, result))
            arg_in = q_in.get()
        q_out.put(END_TOKEN)

    # put jobs
    [q_in.put(c) for c in jobs + n_workers * [END_TOKEN]]

    # start workers
    workers = [Process(target=queue_worker, args=(q_in, q_out))
               for _ in range(n_workers)]
    [w.start() for w in workers]

    # wait for results
    n_finished = 0
    outputs = []
    while n_finished < n_workers:
        output = q_out.get(timeout=timeout_per_job)
        logger.info(f'queue out, got: {output}')
        if output == END_TOKEN:
            n_finished += 1
            logger.info(f'{n_finished}/{n_workers} queue workers done')
        else:
            outputs.append(output)

    # wait for workers to join
    logger.info('Joining queue workers')
    [w.join() for w in workers]
    logger.info('Joined all queue workers')

    # sort in original order
    results = [output[1] for output in
               sorted(outputs, key=lambda output: jobs.index(output[0]))]
    return results