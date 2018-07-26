import warnings

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import scipy.sparse as sp

from multiprocessing.pool import Pool

from sklearn.model_selection import train_test_split

from ml_recsys_tools.utils.parallelism import batch_generator, N_CPUS
from ml_recsys_tools.utils.instrumentation import LogCallsTimeAndOutput
from ml_recsys_tools.utils.logger import simple_logger as logger
from ml_recsys_tools.utils.pandas_utils import console_settings
from ml_recsys_tools.utils.sklearn_extenstions import PDLabelEncoder

console_settings()

RANDOM_STATE = 42


class ObservationsDF(LogCallsTimeAndOutput):

    def __init__(self, df_obs=None, uid_col='userid', iid_col='itemid', timestamp_col=None,
                 rating_col='rating', verbose=True, **kwargs):
        super().__init__(verbose, **kwargs)
        self.df_obs = df_obs
        self.uid_col = uid_col
        self.iid_col = iid_col
        self.rating_col = rating_col
        self.timestamp_col = timestamp_col

        if self.df_obs is not None:
            self.df_obs[self.uid_col] = self.df_obs[self.uid_col].astype(str)
            self.df_obs[self.iid_col] = self.df_obs[self.iid_col].astype(str)
            self.df_obs[self.rating_col] = self.df_obs[self.rating_col].astype(float)

        self._check_duplicated_interactions()

    def __len__(self):
        return len(self.df_obs)

    def __repr__(self):
        return super().__repr__() + ', %d Observations' % len(self)

    def __add__(self, other):
        self.df_obs = pd.concat([self.df_obs, other.df_obs])
        self._check_duplicated_interactions()
        return self

    @property
    def user_ids(self):
        return self.df_obs[self.uid_col].values

    @property
    def item_ids(self):
        return self.df_obs[self.iid_col].values

    @property
    def ratings(self):
        return self.df_obs[self.rating_col].values

    @property
    def timestamps(self):
        return self.df_obs[self.timestamp_col].values

    def _check_duplicated_interactions(self):
        dups = self.df_obs.duplicated([self.uid_col, self.iid_col])
        if dups.sum():
            logger.warn('ObservationsDF: Dropping %s duplicate interactions.'
                        % str(dups.sum()))
            self.df_obs = self.df_obs[~dups]

    def _user_and_item_counts(self, plot=False):
        if len(self.df_obs):
            items_per_user = self.df_obs[[self.uid_col]].groupby(self.uid_col).apply(len). \
                sort_values(ascending=False).reset_index(name='items_per_user')
            users_per_items = self.df_obs[[self.iid_col]].groupby(self.iid_col).apply(len). \
                sort_values(ascending=False).reset_index(name='users_per_items')
        else:
            raise ValueError('Observations dataframe is empty')

        if plot:
            items_per_user.value_counts().reset_index(drop=True). \
                plot(logx=True, logy=True, grid=True, label='items_per_user')
            users_per_items.value_counts().reset_index(drop=True). \
                plot(logx=True, logy=True, grid=True, label='users_per_items')
            plt.ylabel('count of ..')
            plt.xlabel('with that many')
            plt.legend()
        return items_per_user, users_per_items

    def sample_observations(self,
                            n_users=None,
                            n_items=None,
                            method='random',
                            min_user_hist=0,
                            min_item_hist=0,
                            users_to_keep=(),
                            items_to_keep=(),
                            random_state=None):
        """
        :param n_users: number of users to sample
        :param n_items: number of listings to sample
        :param method: either 'random' or 'top' (sample the top users and top items by views)
        :param min_user_hist: minimal number of unique items viewed by a user
        :param users_to_keep: specific users that have to be kept
        :param items_to_keep: specific items that have to be kept
        :return: dataframe
        """
        if method == 'top' or min_user_hist or min_item_hist:
            items_per_user, users_per_items = self._user_and_item_counts()

            if min_user_hist:
                items_per_user = items_per_user[
                    items_per_user['items_per_user'] >= min_user_hist]
            users_filt = items_per_user[self.uid_col].values

            if min_item_hist:
                users_per_items = users_per_items[
                    users_per_items['users_per_items'] >= min_item_hist]
            item_filt = users_per_items[self.iid_col].values

        else:
            users_filt = self.df_obs[self.uid_col].unique().astype(str)
            item_filt = self.df_obs[self.iid_col].unique().astype(str)

        if n_users is None:
            users_sample = users_filt
        elif method == 'random':
            # users_sample = users_filt.sample(n_users, random_state=random_state)
            np.random.seed(random_state)
            users_sample = np.random.choice(users_filt, n_users, replace=False)
        elif method == 'top':
            users_sample = users_filt[:n_users]
        else:
            raise ValueError('Uknown sampling method')

        if n_items is None:
            item_sample = item_filt
        elif method == 'random':
            # item_sample = item_filt.sample(n_items, random_state=random_state)
            np.random.seed(random_state)
            item_sample = np.random.choice(item_filt, n_items, replace=False)
        elif method == 'top':
            item_sample = item_filt[:n_items]
        else:
            raise ValueError('Uknown sampling method')

        if len(users_to_keep):
            users_sample = np.concatenate([users_sample, np.array(users_to_keep)])

        if len(items_to_keep):
            item_sample = np.concatenate([item_sample, np.array(items_to_keep)])

        sample_df = self.df_obs[(self.df_obs[self.iid_col].isin(item_sample)) &
                                (self.df_obs[self.uid_col].isin(users_sample))]

        other = copy.deepcopy(self)
        other.df_obs = sample_df.copy()

        return other

    def filter_columns_by_df(self, other_df_obs):
        """
        removes users or items that are not in the other user dataframe
        :param other_df_obs: other dataframe, that has the same structure (column names)
        :return: new observation handler
        """
        other = copy.deepcopy(self)
        other.df_obs = self.df_obs[
            (self.df_obs[self.iid_col].isin(other_df_obs[self.iid_col].unique())) &
            (self.df_obs[self.uid_col].isin(other_df_obs[self.uid_col].unique()))]
        return other

    def remove_interactions_by_df(self, other_df_obs):
        """
        removes all interactions that are present in the other dataframe.
        e.g. remove training examples from the full dataframe
        :param other_df_obs: other dataframe
        :return: new observation handler
        """
        df_filtered = pd.merge(
            self.df_obs, other_df_obs, on=[self.iid_col, self.uid_col], how='left')
        df_filtered = df_filtered[df_filtered[self.rating_col + '_y'].isnull()]. \
            rename({self.rating_col + '_x': self.rating_col}, axis=1). \
            drop(self.rating_col + '_y', axis=1)
        other = copy.deepcopy(self)
        other.df_obs = df_filtered
        return other

    def user_filtered_df(self, user):
        return self.df_obs[self.df_obs[self.uid_col] == user]

    def items_filtered_df(self, item):
        return self.df_obs[self.df_obs[self.iid_col] == item]

    @staticmethod
    def time_filter_on_df(df, time_col, days_delta_tuple):
        time_max = pd.Timestamp(df[time_col].max()) - pd.Timedelta(days=min(days_delta_tuple))
        time_delta = pd.Timedelta(days=abs(days_delta_tuple[0] - days_delta_tuple[1]))
        time_filt = (str(time_max - time_delta) < df[time_col].astype(str)) & \
                    (df[time_col].astype(str) <= str(time_max))
        return time_filt

    def get_sparse_matrix_helper(self):
        mat_builder = InteractionMatrixBuilder(
            self.df_obs, users_col=self.uid_col,
            items_col=self.iid_col, rating_col=self.rating_col)
        return mat_builder

    def split_train_test_to_dfs(self, ratio=0.2, users_ratio=1.0,
                                time_split_column=None, random_state=None):
        """
        splits the underlying dataframe  into train and test using the arguments to choose the method.
        If time_split_column is provided: the split is done using that column and provided ratio.
        Otherwise a random split either within a subset of users or not is done.

        Raises an error if both users_ratio and time_split_column are not their defaults,
        because only one method can be used.

        :param ratio: the fraction of data that should be in the resulting test segment.
        :param users_ratio: the fraction of users for which a split should be done. Using this arguments
            speeds up subsequent evaluation, because prediction will only be calculated for a subset
            of users. The ratio is only applied to those users.
        :param time_split_column: an optional argument that changes the split into a time split.
        :param random_state: a random state that is only used if the split if a shuffle split (not time).

        :return: two ObservationsDF objects, one with training data and the other with test data.
        """

        if users_ratio < 1.0 and time_split_column is not None:
            raise ValueError('Can either split by time, or for subset of users, not both.')

        if users_ratio < 1.0:
            return train_test_split_by_col(
                self.df_obs, col_ratio=users_ratio, test_ratio=ratio,
                col_name=self.uid_col, random_state=random_state)

        elif time_split_column:
            self.df_obs.sort_values(time_split_column, inplace=True)
            split_ind = int((len(self.df_obs)-1) * ratio)
            return self.df_obs.iloc[:-split_ind].copy(), self.df_obs.iloc[-split_ind:].copy()

        else:
            return train_test_split(self.df_obs, test_size=ratio, random_state=random_state)

    def split_by_time_col(self, time_col, days_delta_tuple):

        time_filt = self.time_filter_on_df(
            self.df_obs, time_col=time_col, days_delta_tuple=days_delta_tuple)

        df_train = self.df_obs[~time_filt].copy()
        df_test = self.df_obs[time_filt].copy()

        train_other = copy.deepcopy(self)
        train_other.df_obs = df_train

        test_other = copy.deepcopy(self)
        test_other.df_obs = df_test

        return train_other, test_other

    def split_train_test(self, ratio=0.2, users_ratio=1.0, time_split_column=None, random_state=None):
        """
        splits the object into train and test objects using the arguments to choose the method.
        If time_split_column is provided: the split is done using that column and provided ratio.
        Otherwise it's a random split either within a subset of users or not is done.

        :param ratio: the fraction of data that should be in the resulting test segment
        :param users_ratio: the fraction of users for which a split should be done. Using this arguments
            speeds up subsequent evaluation, because prediction will only be calculated for a subset
            of users. The ratio is only applied to those users.
        :param time_split_column: an optional argument that changes the split into a time split.
        :param random_state: a random state that is only used if the split if a shuffle split (not time).

        :return: two ObservationsDF objects, one with training data and the other with test data.
        """
        df_train, df_test = self.split_train_test_to_dfs(
            ratio=ratio,
            users_ratio=users_ratio,
            random_state=random_state,
            time_split_column=time_split_column)

        train_other = copy.deepcopy(self)
        train_other.df_obs = df_train

        test_other = copy.deepcopy(self)
        test_other.df_obs = df_test

        return train_other, test_other

    def generate_time_batches(self, n_batches, time_col, days_delta=1):
        remaining = copy.deepcopy(self)
        inf_days = 10000  # just lots of days
        for i in reversed(range(n_batches)):
            # this is done so that the earliest batch contains all
            # the data that's not in later batches
            delta = days_delta if i < (n_batches - 1) else inf_days

            remaining, batch_obs = remaining.split_by_time_col(
                time_col, (i * days_delta, i * days_delta + delta))
            yield batch_obs.df_obs


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
    df_train = pd.concat([df_train_col, df_non_test_items])
    # shuffle
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    return df_train, df_test


class InteractionMatrixBuilder(LogCallsTimeAndOutput):

    # this filter is due to this issue, can be removed with next version of sklearn (should be fixed)
    # https://stackoverflow.com/questions/49545947/sklearn-deprecationwarning-truth-value-of-an-array
    warnings.filterwarnings(message="The truth value of an empty array is ambiguous. "
                                    "Returning False, but in future this will result in an error. "
                                    "Use `array.size > 0` to check that an array is not empty.",
                            action='ignore', category=DeprecationWarning)

    def __init__(self, source_df, users_col='userid', items_col='adid', rating_col='rating', verbose=True):
        super().__init__(verbose=verbose)

        self.uid_source_col = users_col
        self.iid_source_col = items_col
        self.rating_source_col = rating_col
        self.uid_col = 'uuid_coord'
        self.iid_col = 'iid_coord'

        all_uids = source_df[self.uid_source_col].unique()
        all_iids = source_df[self.iid_source_col].unique()

        # shuffling because np.unique() returns elements in almost sorted order by counts,
        # and it's probably not a good thing: it changes regional sparsity,
        # and at a later stage might be sampled / iterated in order
        np.random.shuffle(all_uids)
        np.random.shuffle(all_iids)

        self.n_rows = len(all_uids)
        self.n_cols = len(all_iids)

        # self.uid_encoder = LabelEncoder().fit(all_uids)
        # self.iid_encoder = LabelEncoder().fit(all_iids)
        self.uid_encoder = PDLabelEncoder().fit(all_uids)
        self.iid_encoder = PDLabelEncoder().fit(all_iids)

    def add_encoded_cols(self, df):
        df = df.assign(
            **{self.uid_col: self.uid_encoder.transform(df[self.uid_source_col].values.astype(str)),
               self.iid_col: self.iid_encoder.transform(df[self.iid_source_col].values.astype(str))})
        return df

    def build_sparse_interaction_matrix(self, df):
        """
        :param df:
        :return: the sparse matrix populated with interactions, of shape (n_users, n_items)
            of the source DF (which which this builder with initialized
        """

        df = self.remove_unseen_labels(df)

        mat = sp.coo_matrix(
            (df[self.rating_source_col].values,
             (self.uid_encoder.transform(df[self.uid_source_col].values),
              self.iid_encoder.transform(df[self.iid_source_col].values))),
            shape=(self.n_rows, self.n_cols),
            dtype=np.float32).tocsr()

        return mat

    def remove_unseen_labels(self, df):
        # new_u = ~df[self.uid_source_col].isin(self.uid_encoder.classes_)
        new_u = self.uid_encoder.find_new_labels(df[self.uid_source_col])
        # new_i = ~df[self.iid_source_col].isin(self.iid_encoder.classes_)
        new_i = self.iid_encoder.find_new_labels(df[self.iid_source_col])
        percent_new_u = np.mean(new_u)
        percent_new_i = np.mean(new_i)
        if percent_new_u > 0.0 or percent_new_i > 0.0:
            logger.info(
                'Discarding %.1f%% samples with unseen '
                'users(%d) / unseen items(%d) from DF(len: %s).' % \
                (100 * np.mean(new_u | new_i), np.sum(new_u), np.sum(new_i), len(df)))
            return df[~new_u & ~new_i].copy()
        else:
            return df

    def predictions_df_to_sparse_ranks(self, preds_df):
        preds_all = self.build_sparse_interaction_matrix(preds_df)
        return self.predictions_to_ranks(preds_all)

    @staticmethod
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
        filt_ranks.data += int(filt_ranks.shape[1] / 2)  # add number of columns / 2 - meaning chance rank
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

    @staticmethod
    def crop_rows(mat, inds_stay):
        mat = mat.tocoo()
        min_data = np.min(mat.data)
        mat.data += min_data
        mat.data[~np.in1d(mat.row, inds_stay)] *= 0
        mat.eliminate_zeros()
        mat.data -= min_data
        return mat.tocsr()

    @staticmethod
    def crop_rows_continuous(mat, ind_start, ind_end):
        mat = mat.tocsr().copy()
        mat.sort_indices()
        mat.data += 1
        mat.data[:mat.indptr[ind_start]] *= 0
        mat.data[(mat.indptr[ind_end + 1] + 1):] *= 0
        mat.eliminate_zeros()
        mat.data -= 1
        return mat

    @classmethod
    def filter_all_ranks_by_sparse_selection(cls, sparse_filter_mat, all_recos_ranks_mat):
        """
        generates rankings for a an evaluation of a dataset (test set), relative to all valid predictions

        :param sparse_filter_mat: sparse matrix of test observations (ground truth)
        :param all_recos_ranks_mat: sparse matrix of all ranked predictions
        :return: sparse matrix of ranks of the predictions for GT observations in the full prediction matrix
        """
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
                        cls.crop_rows_continuous(ranks_mat, ind_batch[0], ind_batch[-1]),
                        cls.crop_rows_continuous(filter_mat, ind_batch[0], ind_batch[-1]))))
            ret = [r.get(timeout=3600) for r in res]

        data = np.concatenate([r.data for r in ret])
        row = np.concatenate([r.row for r in ret])
        col = np.concatenate([r.col for r in ret])
        filt_ranks = sp.coo_matrix((data, (row, col)), shape=ranks_mat.shape)

        # ranks are 0 based, and float32 in LFM
        filt_ranks = filt_ranks.astype(np.float32).tocsr()

        return filt_ranks
