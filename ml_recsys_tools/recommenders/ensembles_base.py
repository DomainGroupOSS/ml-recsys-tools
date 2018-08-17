import warnings
from abc import abstractmethod, ABC
from functools import partial
from itertools import repeat
from multiprocessing import Process, Queue
from multiprocessing.pool import ThreadPool, Pool
from queue import PriorityQueue
from threading import Thread

import numpy as np
import pandas as pd
import scipy.stats

from ml_recsys_tools.recommenders.recommender_base import BaseDFSparseRecommender
from ml_recsys_tools.utils.parallelism import N_CPUS


RANK_COMBINATION_FUNCS = {
    'mean': np.mean,
    'max': np.max,
    'min': np.min,
    'gmean': scipy.stats.gmean,
    'hmean': scipy.stats.hmean
}


def calc_dfs_and_combine_scores(calc_funcs, groupby_col, item_col, scores_col,
                                fill_val, combine_func='hmean', n_threads=1,
                                parallelism='process'):
    """
    combine multiple dataframes by voting on prediction rank

    :param calc_funcs: functions that return the dataframes to be combined
    :param combine_func: defaults 'hmean', the functions that is used to combine the predictions
        (can be callable line np.mean or a string that is assumed to be
        a key in rank_combination_functions mapping
    :param fill_val: rank to be assigned to NaN prediction values
        (items appearing in some dataframes but not in others)
    :param groupby_col: the column of the entities for which the ranking is calculated (e.g. users)
    :param item_col: the column of the entities to be ranked (items)
    :param scores_col: the column of the scores to be ranked (predictions)
    :param n_threads: number of calculation threads
    :param parallelism: type of parallelism (processes or threads)
    :return: a combined dataframe of the same format as the dataframes created by the calc_funcs
    """
    # set up
    multiproc = 'process' in parallelism
    _END = 'END'
    q_in = Queue()
    q_out = Queue() if multiproc else PriorityQueue()
    rank_cols = ['rank_' + str(i) for i in range(len(calc_funcs))]
    n_jobs = len(calc_funcs)
    n_workers = min(n_threads, n_jobs)
    if not callable(combine_func):
        combine_func = RANK_COMBINATION_FUNCS[combine_func]

    jitter = lambda: np.random.rand()

    def _calc_df_and_add_rank_score(i):
        df = calc_funcs[i]()
        df = df.drop_duplicates()

        # another pandas bug workaround
        df[groupby_col] = df[groupby_col].astype(str, copy=False)
        df[item_col] = df[item_col].astype(str, copy=False)
        df[scores_col] = df[scores_col].astype(float, copy=False)
        df = df.reset_index(drop=True) # resetting index due to pandas bug

        df[rank_cols[i]] = df. \
            groupby(groupby_col)[scores_col].\
            rank(ascending=False)

        df = df.drop(scores_col, axis=1).set_index([groupby_col, item_col])

        q_out.put((len(df) + jitter(), df))

    def _joiner():
        while True:
            _, df1 = q_out.get()
            if isinstance(df1, str) and df1 == _END:
                break
            _, df2 = q_out.get()
            if isinstance(df2, str) and df2 == _END:
                q_out.put((len(df1) + jitter(), df1))  # put it back
                break
            df_joined = df2.join(df1, how='outer')
            q_out.put((len(df_joined) + jitter(), df_joined))

    def _worker():
        i = q_in.get()
        while i != _END:
            _calc_df_and_add_rank_score(i)
            i = q_in.get()

    if multiproc:
        workers = [Process(target=_worker) for _ in range(n_workers)]
    else:
        workers = [Thread(target=_worker) for _ in range(n_workers)]

    joiner = Thread(target=_joiner)

    # submit and start jobs
    [q_in.put(i) for i in range(n_jobs)] + [q_in.put(_END) for _ in range(n_workers)]
    [j.start() for j in workers + [joiner]]
    [j.join() for j in workers]

    # stop joiner after workers are done by putting END token
    q_out.put((0, _END))
    joiner.join()

    # final reduce (faster to join in couples rather one by one)
    while q_out.qsize() > 1:
        _, df1 = q_out.get()
        _, df2 = q_out.get()
        df_joined = df2.join(df1, how='outer')
        q_out.put((len(df_joined), df_joined))
    # get final result
    _, merged_df = q_out.get()
    merged_df.fillna(fill_val, inplace=True)
    # combine ranks
    merged_df[scores_col] = combine_func(1 / merged_df[rank_cols].values, axis=1)

    # drop temp cols
    merged_df.drop(rank_cols, axis=1, inplace=True)

    return merged_df.reset_index()


class EnsembleBase(BaseDFSparseRecommender):

    def __init__(self,
                 combination_mode='hmean',
                 na_rank_fill=None,
                 **kwargs):
        self.combination_mode = combination_mode
        self.na_rank_fill = na_rank_fill
        self.recommenders = []
        super().__init__(**kwargs)

    def n_concurrent(self):
        return N_CPUS

    def set_params(self, **params):
        params = self._pop_set_params(params, ['combination_mode', 'na_rank_fill'])
        # set on self
        super().set_params(**params.copy())

    def _get_recommendations_flat(self, user_ids, item_ids, n_rec=100, **kwargs):

        calc_funcs = [
            partial(
                rec.get_recommendations,
                user_ids=user_ids, item_ids=item_ids,
                n_rec=n_rec, results_format='flat', **kwargs)
            for rec in self.recommenders]

        recos_flat = calc_dfs_and_combine_scores(
            calc_funcs=calc_funcs,
            combine_func=self.combination_mode,
            fill_val=self.na_rank_fill if self.na_rank_fill else (n_rec + 1),
            groupby_col=self._user_col,
            item_col=self._item_col,
            scores_col=self._prediction_col,
            n_threads=self.n_concurrent()
        )

        return recos_flat

    def get_similar_items(self, item_ids=None, target_item_ids=None, n_simil=10,
                          n_unfilt=100, results_format='lists', **kwargs):

        calc_funcs = [partial(rec.get_similar_items,
                              item_ids=item_ids, target_item_ids=target_item_ids,
                              n_simil=n_unfilt, results_format='flat', **kwargs)
                      for rec in self.recommenders]

        combined_simil_df = calc_dfs_and_combine_scores(
            calc_funcs=calc_funcs,
            combine_func=self.combination_mode,
            fill_val=self.na_rank_fill if self.na_rank_fill else (n_unfilt + 1),
            groupby_col=self._item_col_simil,
            item_col=self._item_col,
            scores_col=self._prediction_col)

        return combined_simil_df if results_format == 'flat' \
            else self._simil_flat_to_lists(combined_simil_df, n_cutoff=n_simil)

    def _predict_on_inds_dense(self, user_inds, item_inds):
        raise NotImplementedError()

    def predict_for_user(self, user_id, item_ids, rank_training_last=True,
                         sort=True, combine_original_order=False):

        calc_funcs = [
            partial(
                rec.predict_for_user,
                user_id=user_id,
                item_ids=item_ids,
                rank_training_last=rank_training_last,
                combine_original_order=combine_original_order,
            )
            for rec in self.recommenders]

        df = calc_dfs_and_combine_scores(
            calc_funcs=calc_funcs,
            combine_func=self.combination_mode,
            fill_val=len(item_ids),
            groupby_col=self._user_col,
            item_col=self._item_col,
            scores_col=self._prediction_col,
            n_threads=N_CPUS,
            parallelism='thread'
        )

        if sort:
            df.sort_values(self._prediction_col, ascending=False, inplace=True)

        return df


class SubdivisionEnsembleBase(EnsembleBase):

    def __init__(self,
                 n_recommenders=1,
                 concurrence_ratio=0.3,
                 concurrency_backend='threads',
                 **kwargs):
        self.n_recommenders = n_recommenders
        self.concurrence_ratio = concurrence_ratio
        self.concurrency_backend = concurrency_backend
        super().__init__(**kwargs)
        self.sub_class_type = None
        self._init_recommenders()

    def get_workers_pool(self, concurrency_backend=None):
        if concurrency_backend is None:
            concurrency_backend = self.concurrency_backend
        if 'thread' in concurrency_backend:
            return ThreadPool(self.n_concurrent())
        elif 'proc' in concurrency_backend:
            return Pool(self.n_concurrent(), maxtasksperchild=3)

    def _init_recommenders(self, **params):
        self.recommenders = [self.sub_class_type(**params.copy())
                             for _ in range(self.n_recommenders)]

    def n_concurrent(self):
        return int(min(np.ceil(len(self.recommenders) * self.concurrence_ratio), N_CPUS))

    def set_params(self, **params):
        params = self._pop_set_params(
            params, ['n_recommenders', 'concurrence_ratio'])
        # set on self
        super().set_params(**params.copy())
        # init sub models to make sure they're the right object already
        self._init_recommenders(**self.model_params)
        # # set for each sub_model
        # for model in self.recommenders:
        #     # model.set_params(**params.copy())
        #     model.set_params()

    @abstractmethod
    def _generate_sub_model_train_data(self, train_obs):
        pass

    @abstractmethod
    def _fit_sub_model(self, args):
        pass

    def fit(self, train_obs, **fit_params):
        self._set_data(train_obs)

        sub_model_train_data_generator = self._generate_sub_model_train_data(train_obs)

        n_recommenders = self.n_recommenders

        with self.get_workers_pool() as pool:
            self.recommenders = list(
                pool.imap(self._fit_sub_model,
                          zip(range(n_recommenders),
                              sub_model_train_data_generator,
                              repeat(fit_params, n_recommenders))))
        return self

    # def sub_model_evaluations(self, test_dfs, test_names, include_train=True):
    #     stats = []
    #     reports = []
    #     for m in self.recommenders:
    #         users = m.train_df[self.train_obs.uid_col].unique()
    #         items = m.train_df[self.train_obs.iid_col].unique()
    #         sub_test_dfs = [df[df[self.train_obs.uid_col].isin(users) &
    #                            df[self.train_obs.iid_col].isin(items)] for df in test_dfs]
    #         lfm_report = m.eval_on_test_by_ranking(
    #             include_train=include_train,
    #             test_dfs=sub_test_dfs,
    #             prefix='lfm sub model',
    #             test_names=test_names
    #         )
    #         stats.append('train: %d, test: %s' %
    #                      (len(m.train_df), [len(df) for df in sub_test_dfs]))
    #         reports.append(lfm_report)
    #     return stats, reports


class CombinationEnsembleBase(EnsembleBase):

    def __init__(self, recommenders, **kwargs):
        super().__init__(**kwargs)
        self.recommenders = recommenders
        self._reuse_data(self.recommenders[0])

    def fit(self, *args, **kwargs):
        warnings.warn('Fit is not supported, recommenders should already be fitted.')

