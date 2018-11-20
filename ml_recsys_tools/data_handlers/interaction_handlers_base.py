import warnings
from types import SimpleNamespace

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
        super().__init__(verbose=verbose, **kwargs)
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
        self.df_obs = pd.concat([self.df_obs, other.df_obs], sort=False)
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
            logger.warning('ObservationsDF: Dropping %s duplicate interactions.'
                        % str(dups.sum()))
            self.df_obs = self.df_obs[~dups]

    def users_history_counts(self):
        if len(self.df_obs):
            return self.df_obs[self.uid_col].value_counts()
        else:
            raise ValueError('Observations dataframe is empty')

    def items_history_counts(self):
        if len(self.df_obs):
            return self.df_obs[self.iid_col].value_counts()
        else:
            raise ValueError('Observations dataframe is empty')

    def data_info(self):
        rating_pctl = np.percentile(self.ratings, [20, 80])
        return {
            'len': len(self.df_obs),
            'n_unique_users': len(np.unique(self.user_ids)),
            'n_unique_items': len(np.unique(self.item_ids)),
            'ratings_20_pctl': rating_pctl[0],
            'ratings_80_pctl': rating_pctl[1],
        }

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
            items_per_user = self.users_history_counts()
            users_per_items= self.items_history_counts()

            if min_user_hist:
                items_per_user = items_per_user[items_per_user.values >= min_user_hist]
            users_filt = items_per_user.index.astype(str).values

            if min_item_hist:
                users_per_items = users_per_items[users_per_items.values >= min_item_hist]
            item_filt = users_per_items.index.astype(str).values

        else:
            users_filt = self.df_obs[self.uid_col].unique().astype(str)
            item_filt = self.df_obs[self.iid_col].unique().astype(str)

        if n_users is None or n_users >= len(users_filt):
            users_sample = users_filt
        elif method == 'random':
            # users_sample = users_filt.sample(n_users, random_state=random_state)
            np.random.seed(random_state)
            users_sample = np.random.choice(users_filt, n_users, replace=False)
        elif method == 'top':
            users_sample = users_filt[:n_users]
        else:
            raise ValueError('Uknown sampling method')

        if n_items is None or n_items >= len(item_filt):
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

    def filter_interactions_by_df(self, other_df_obs, mode):
        """
        removes / keeps all interactions that are present in the other dataframe.
        e.g. remove training examples from a full observations dataframe following
        :param other_df_obs: other dataframe
        :param mode: whether to remove to keep the observations in the dataframe,
            that are also in the other dataframe
        :return: new observation handler
        """
        df_filtered = pd.merge(
            self.df_obs, other_df_obs, on=[self.iid_col, self.uid_col], how='left')

        if mode=='remove':
            filt_mask = df_filtered[self.rating_col + '_y'].isnull()
        elif mode=='keep':
            filt_mask = df_filtered[self.rating_col + '_y'].notnull()
        else:
            raise ValueError(f'unknown value for mode:{mode}')

        df_filtered = df_filtered[filt_mask]. \
            rename({self.rating_col + '_x': self.rating_col}, axis=1). \
            drop(self.rating_col + '_y', axis=1)
        other = copy.deepcopy(self)
        other.df_obs = df_filtered
        return other

    def users_filtered_df_obs(self, users):
        if isinstance(users, str):
            users = [users]
        return self.df_obs[self.df_obs[self.uid_col].isin(users)]

    def items_filtered_df_obs(self, items):
        if isinstance(items, str):
            items = [items]
        return self.df_obs[self.df_obs[self.iid_col].isin(items)]

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
    df_train = pd.concat([df_train_col, df_non_test_items], sort=False)
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
    def crop_rows(mat, inds_stay):
        mat = mat.tocoo()
        min_data = np.min(mat.data)
        mat.data += min_data
        mat.data[~np.in1d(mat.row, inds_stay)] *= 0
        mat.eliminate_zeros()
        mat.data -= min_data
        return mat.tocsr()

    @classmethod
    def filter_all_ranks_by_sparse_selection(cls, sparse_filter_mat, all_recos_ranks_mat):
        """
        generates rankings for a an evaluation of a dataset (test set), relative to all valid predictions

        :param sparse_filter_mat: sparse matrix of test observations (ground truth)
        :param all_recos_ranks_mat: sparse matrix of all ranked predictions
        :return: sparse matrix of ranks of the predictions for GT observations in the full prediction matrix
        """
        filter_mat = sparse_filter_mat.copy().tocsr()
        ranks_mat = all_recos_ranks_mat.copy().tocsr()
        ranks_mat.data += 1
        filt_ranks = filter_mat.astype(bool).multiply(ranks_mat)
        filt_ranks.eliminate_zeros()
        filt_ranks.data -= 1

        return filt_ranks
