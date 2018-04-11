import warnings

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import scipy.sparse as sp

from multiprocessing.pool import Pool

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from ml_recsys_tools.utils.parallelism import batch_generator, parallelize_dataframe, N_CPUS
from ml_recsys_tools.utils.instrumentation import log_time_and_shape
from ml_recsys_tools.utils.logger import simple_logger as logger
from ml_recsys_tools.utils.pandas_utils import console_settings

console_settings()

RANDOM_STATE = 42


class ObservationsDF:

    def __init__(self, df_obs=None, uid_col='userid', iid_col='itemid', rating_col='rating', **kwargs):
        self.df_obs = df_obs
        self.uid_col = uid_col
        self.iid_col = iid_col
        self.rating_col = rating_col

        if self.df_obs is not None:
            self.df_obs[self.uid_col] = self.df_obs[self.uid_col].astype(str)
            self.df_obs[self.iid_col] = self.df_obs[self.iid_col].astype(str)
            self.df_obs[self.rating_col] = self.df_obs[self.rating_col].astype(float)

    def user_and_item_counts(self, plot=False):
        if len(self.df_obs):
            self.items_per_user = self.df_obs.groupby(self.uid_col).apply(len). \
                sort_values(ascending=False).reset_index(name='items_per_user')
            self.users_per_items = self.df_obs.groupby(self.iid_col).apply(len). \
                sort_values(ascending=False).reset_index(name='users_per_items')
        else:
            raise ValueError('Observations dataframe is empty')

        if plot:
            self.items_per_user.value_counts().reset_index(drop=True). \
                plot(logx=True, logy=True, grid=True, label='items_per_user')
            self.users_per_items.value_counts().reset_index(drop=True). \
                plot(logx=True, logy=True, grid=True, label='users_per_items')
            plt.ylabel('count of ..')
            plt.xlabel('with that many')
            plt.legend()

    @log_time_and_shape
    def sample_observations(self,
                            n_users=None,
                            n_items=None,
                            method='random',
                            min_user_hist=0,
                            min_item_hist=0,
                            random_state=None):
        """
        :param n_users: number of users to sample
        :param n_items: number of listings to sample
        :param method: either 'random' or 'top' (sample the top users and top items by views)
        :param min_user_hist: minimal number of unique items viewed by a user
        :param min_item_hist: minimal number of unique users who viewed a item
        :return: dataframe
        """
        self.user_and_item_counts()

        if min_item_hist:
            item_filt = self.users_per_items[
                self.users_per_items['users_per_items'] >= min_item_hist]
        else:
            item_filt = self.users_per_items

        if min_user_hist:
            users_filt = self.items_per_user[
                self.items_per_user['items_per_user'] >= min_user_hist]
        else:
            users_filt = self.items_per_user

        if n_users is None:
            users_sample = users_filt[self.uid_col]
        elif method == 'random':
            users_sample = users_filt.sample(n_users, random_state=random_state)[self.uid_col]
        elif method == 'top':
            users_sample = users_filt.iloc[:n_users][self.uid_col]
        else:
            raise ValueError('Uknown sampling method')

        if n_items is None:
            item_sample = item_filt[self.iid_col]
        elif method == 'random':
            item_sample = item_filt.sample(n_items, random_state=random_state)[self.iid_col]
        elif method == 'top':
            item_sample = item_filt.iloc[:n_items][self.iid_col]
        else:
            raise ValueError('Uknown sampling method')

        sample_df = self.df_obs[(self.df_obs[self.iid_col].isin(item_sample)) &
                                (self.df_obs[self.uid_col].isin(users_sample))]

        other = copy.deepcopy(self)
        other.df_obs = sample_df.copy()

        return other

    def filter_by_df(self, other_df_obs):
        other = copy.deepcopy(self)
        other.df_obs = self.df_obs[
            (self.df_obs[self.iid_col].isin(other_df_obs[self.iid_col].unique())) &
            (self.df_obs[self.uid_col].isin(other_df_obs[self.uid_col].unique()))]
        return other

    def user_filtered_df(self, user):
        return self.df_obs[self.df_obs[self.uid_col] == user]

    def items_filtered_df(self, item):
        return self.df_obs[self.df_obs[self.iid_col] == item]

    def get_sparse_matrix_helper(self):
        mat_builder = InteractionMatrixBuilder(
            self.df_obs, users_col=self.uid_col,
            items_col=self.iid_col, rating_col=self.rating_col)
        return mat_builder

    def split_train_test_to_dfs(self, ratio=0.2, users_ratio=1.0, random_state=None):
        if users_ratio < 1.0:
            return train_test_split_by_col(
                self.df_obs, col_ratio=users_ratio, test_ratio=ratio,
                col_name=self.uid_col, random_state=random_state)
        else:
            return train_test_split(self.df_obs, test_size=ratio, random_state=random_state)

    def split_train_test(self, ratio=0.2, users_ratio=1.0, random_state=None):
        obs_train, obs_test = self.split_train_test_to_dfs(
            ratio=ratio, users_ratio=users_ratio, random_state=random_state)

        train_other = copy.deepcopy(self)
        train_other.df_obs = obs_train.copy()

        test_other = copy.deepcopy(self)
        test_other.df_obs = obs_test.copy()

        return train_other, test_other


def train_test_split_by_col(df, col_ratio=0.2, test_ratio=0.2, col_name='userid', random_state=None):
    # split field unique values
    vals_train, vals_test = train_test_split(
        df[col_name].unique(), test_size=col_ratio, random_state=random_state)

    df_train_col = df[df[col_name].isin(vals_train)]
    df_test_col = df[df[col_name].isin(vals_test)]

    # split within selected field values
    df_non_test_items, df_test = train_test_split(
        df_test_col, test_size=test_ratio, random_state=random_state)

    # concat train dfs
    df_train = pd.concat([df_train_col, df_non_test_items], axis=0)
    # shuffle
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    return df_train, df_test


class InteractionMatrixBuilder:

    def __init__(self, source_df, users_col='userid', items_col='adid', rating_col='rating'):

        self.uid_source_col = users_col
        self.iid_source_col = items_col
        self.rating_source_col = rating_col
        self.uid_col = 'uuid_coord'
        self.iid_col = 'iid_coord'

        all_uids = source_df[self.uid_source_col].unique()
        all_iids = source_df[self.iid_source_col].unique()

        self.n_rows = len(all_uids)
        self.n_cols = len(all_iids)

        self.uid_encoder = LabelEncoder().fit(all_uids)
        self.iid_encoder = LabelEncoder().fit(all_iids)

        # this filter is due to this issue, can be removed with next version of sklearn (should be fixed)
        # https://stackoverflow.com/questions/49545947/sklearn-deprecationwarning-truth-value-of-an-array
        warnings.filterwarnings(message="The truth value of an empty array is ambiguous. "
                                        "Returning False, but in future this will result in an error. "
                                        "Use `array.size > 0` to check that an array is not empty.",
                                action='ignore', category=DeprecationWarning)

    def _add_encoded_cols(self, df):
        df = df.assign(
            **{self.uid_col: self.uid_encoder.transform(df[self.uid_source_col].values.astype(str)),
               self.iid_col: self.iid_encoder.transform(df[self.iid_source_col].values.astype(str))})
        return df

    def add_encoded_cols(self, df):
        return parallelize_dataframe(df, self._add_encoded_cols)

    @log_time_and_shape
    def build_sparse_interaction_matrix(self, df, job_size=150000):
        """
        note:
            all this complexity if to to prevent spikes of memory usage and still keep the time close to optimal
            the DF is split into chunks, and each chunk is processed in parallel
        :param df:
        :param job_size: the length of data that a single core should process at a time,
            this parameter allows to control the tradeoff between memory and time,
            large values will be faster but will spike the memory
        :return: the sparse matrix populated with interactions, of shape (n_users, n_items)
            of the source DF (which which this builder with initialized
        """

        df = self.remove_unseen_labels(df)

        n_jobs = len(df) / job_size
        n_parallel = int(max(1, min(N_CPUS, n_jobs, 10)))
        n_chunks = max(1, int(n_jobs / n_parallel))

        u_arrays = np.array_split(df[self.uid_source_col].values.astype(str), n_chunks)
        i_arrays = np.array_split(df[self.iid_source_col].values.astype(str), n_chunks)

        r, c = [], []
        with Pool(n_parallel, maxtasksperchild=5) as pool:
            for sub_u, sub_i in zip(u_arrays, i_arrays):
                r.append(np.concatenate(
                    pool.map(self.uid_encoder.transform, np.array_split(sub_u, n_parallel))))
                c.append(np.concatenate(
                    pool.map(self.iid_encoder.transform, np.array_split(sub_i, n_parallel))))

        mat = sp.coo_matrix(
            (df[self.rating_source_col].values,
             (np.concatenate(r), np.concatenate(c))),
            shape=(self.n_rows, self.n_cols),
            dtype=np.float32)

        ## faster but more memory hungry version, why it comsumes so much memory - I couldn't solve
        # mat = sp.coo_matrix(
        #     (df[self.rating_source_col].values,
        #      (parallelize_array(
        #          df[self.uid_source_col].values, self.uid_encoder.transform, n_partitions=n_parallel),
        #       parallelize_array(
        #           df[self.iid_source_col].values, self.iid_encoder.transform, n_partitions=n_parallel))),
        #     shape=(self.n_rows, self.n_cols),
        #     dtype=np.float32)

        return mat.tocsr()

    @log_time_and_shape
    def remove_unseen_labels(self, df):
        new_u = ~df[self.uid_source_col].isin(self.uid_encoder.classes_)
        new_i = ~df[self.iid_source_col].isin(self.iid_encoder.classes_)
        percent_new_u = np.mean(new_u)
        percent_new_i = np.mean(new_i)
        if percent_new_u > 0.0 or percent_new_i > 0.0:
            logger.info(
                'Discarding %.1f%% samples with unseen '
                'users(%.1f%%) / unseen items (%.1f%%) from DF(len: %s).' % \
                (100 * np.mean(new_u | new_i), 100 * percent_new_u, 100 * percent_new_i, len(df)))
            return df[~new_u & ~new_i].copy()
        else:
            return df

    @log_time_and_shape
    def predictions_df_to_sparse_ranks(self, preds_df):
        preds_all = self.build_sparse_interaction_matrix(preds_df)
        return self.predictions_to_ranks(preds_all)

    @staticmethod
    @log_time_and_shape
    def predictions_to_ranks(sp_preds):
        # convert prediction matrix to ranks matrix
        ranks_mat = sp_preds.tocsr().copy()
        ranks_mat.sort_indices()
        for i in range(ranks_mat.shape[0]):
            ranks_mat.data[ranks_mat.indptr[i]: ranks_mat.indptr[i + 1]] = \
                np.argsort(
                    np.argsort(
                        -ranks_mat.data[ranks_mat.indptr[i]: ranks_mat.indptr[i + 1]])). \
                    astype(np.float32)
        return ranks_mat

    @staticmethod
    def _filt_ranks_mat_by_filt_mat(inds, ranks_mat, filter_mat):
        # apparently a faster implementation of this would be by using lil matrix format

        ranks_mat.sort_indices()
        filter_mat.sort_indices()
        filt_ranks = filter_mat.copy()
        filt_ranks.data *= 0  # remove actual data
        filt_ranks.data += int(filt_ranks.shape[1]/2)  # add number of columns / 2 - meaning chance rank
        for i in inds:
            f_s = filt_ranks.indptr[i]
            f_e = filt_ranks.indptr[i + 1]
            r_s = ranks_mat.indptr[i]
            r_e = ranks_mat.indptr[i + 1]

            f_cols = filt_ranks.indices[f_s: f_e]
            r_cols = ranks_mat.indices[r_s: r_e]

            mask_filt = np.isin(f_cols, r_cols)
            mask_ranks = np.isin(r_cols, f_cols)

            # adding 1 to differentiate from 0 as first rank
            filt_ranks.data[f_s: f_e][mask_filt] = ranks_mat.data[r_s: r_e][mask_ranks] + 1

        filt_ranks.eliminate_zeros()
        filt_ranks.data -= 1
        filt_ranks = filt_ranks.tocoo()
        return filt_ranks

    @classmethod
    @log_time_and_shape
    def filter_all_ranks_by_sparse_selection(cls, sparse_filter_mat, all_recos_ranks_mat):
        """
        generates rankings for a an evaluation of a dataset (test set), relative to all valid predictions

        :param sparse_filter_mat: sparse matrix of test observations (ground truth)
        :param all_recos_ranks_mat: sparse matrix of all ranked predictions
        :return: sparse matrix of ranks of the predictions for GT observations in the full prediction matrix
        """

        def crop_rows(mat, ind_start, ind_end):
            mat = mat.tocsr().copy()
            mat.sort_indices()
            mat.data += 1
            mat.data[:mat.indptr[ind_start]] *= 0
            mat.data[(mat.indptr[ind_end + 1] + 1):] *= 0
            mat.eliminate_zeros()
            mat.data -= 1
            return mat

        filter_mat = sparse_filter_mat.tocsr()
        ranks_mat = all_recos_ranks_mat.tocsr()
        filter_mat.sort_indices()
        ranks_mat.sort_indices()

        assert ranks_mat.shape == filter_mat.shape

        n_rows = ranks_mat.shape[0]
        res = []
        with Pool(N_CPUS) as pool:
            for ind_batch in batch_generator(np.arange(n_rows),
                                             n=int(n_rows / N_CPUS)):
                res.append(pool.apply_async(
                    cls._filt_ranks_mat_by_filt_mat,
                    args=(
                        ind_batch,
                        crop_rows(ranks_mat, ind_batch[0], ind_batch[-1]),
                        crop_rows(filter_mat, ind_batch[0], ind_batch[-1]))))
            ret = [r.get(timeout=3600) for r in res]

        data = np.concatenate([r.data for r in ret])
        row = np.concatenate([r.row for r in ret])
        col = np.concatenate([r.col for r in ret])
        filt_ranks = sp.coo_matrix((data, (row, col)), shape=ranks_mat.shape)

        # ranks are 0 based, and float32 in LFM
        filt_ranks = filt_ranks.astype(np.float32).tocsr()

        return filt_ranks

