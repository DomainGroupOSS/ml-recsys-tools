import os
from abc import ABC, abstractmethod
from copy import deepcopy
from time import sleep
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pickle

import time

from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder

from ml_recsys_tools.utils.logger import simple_logger as logger
from ml_recsys_tools.utils.automl import BayesSearchHoldOut, SearchSpaceGuess
from ml_recsys_tools.evaluation.ranks_scoring import mean_scores_report_on_ranks
from ml_recsys_tools.data_handlers.interaction_handlers_base import InteractionMatrixBuilder, RANDOM_STATE, \
    ObservationsDF
from ml_recsys_tools.utils.instrumentation import LogCallsTimeAndOutput


class BaseDFRecommender(ABC, LogCallsTimeAndOutput):
    default_model_params = {}
    default_fit_params = {}

    def __init__(self, user_col='userid', item_col='itemid',
                 rating_col='rating', prediction_col='prediction',
                 model_params=None, fit_params=None, verbose=True):
        super().__init__(verbose)
        self._user_col = user_col
        self._item_col = item_col
        self._item_col_simil = item_col + '_source'
        self._rating_col = rating_col
        self._prediction_col = prediction_col
        self.model_params = self.default_model_params.copy()
        self.fit_params = self.default_fit_params.copy()
        self._set_model_params(model_params)
        self._set_fit_params(fit_params)
        # self.train_df = None
        self.model = None

    @classmethod
    def guess_search_space(cls):
        return SearchSpaceGuess(cls).\
            set_from_dict(cls.default_model_params).\
            set_from_dict(cls.default_fit_params)

    @staticmethod
    def _dict_update(d, u):
        d = d.copy()
        if u: d.update(u)
        return d

    @staticmethod
    def _pop_set_dict(d, params, pop_params):
        for p in pop_params:
            if p in params:
                d[p] = params.pop(p)
        return params

    def _pop_set_params(self, params, pop_params):
        return self._pop_set_dict(self.__dict__, params, pop_params)

    def _set_model_params(self, params):
        self.model_params = self._dict_update(self.model_params, params)

    def _set_fit_params(self, params):
        self.fit_params = self._dict_update(self.fit_params, params)

    def set_params(self, **params):
        """
        this is for skopt / sklearn compatibility
        """
        # pop-set fit_params if provided in bulk
        self._set_fit_params(params.pop('fit_params', {}))
        # pop-set model_params if provided in bulk
        self._set_model_params(params.pop('model_params', {}))
        # pop-set fit_params by keys from default_fit_params if provided flat
        params = self._pop_set_dict(self.fit_params, params, self.default_fit_params.keys())
        # the rest are assumed to be model_params provided flat
        self._set_model_params(params)

    @abstractmethod
    def fit(self, train_obs: ObservationsDF, *args, **kwargs):
        return self

    @abstractmethod
    def get_recommendations(
            self, user_ids=None, item_ids=None, n_rec=10,
            exclude_training=True,
            results_format='lists',
            **kwargs):
        return pd.DataFrame()

    @abstractmethod
    def eval_on_test_by_ranking(
            self, test_dfs, test_names=('',), prefix='rec ',
            include_train=True, items_filter=None, k=10,
            **kwargs):
        return pd.DataFrame()

    def _recos_lists_to_flat(self, recos_lists_df):

        # n_users = len(recos_lists_df)
        n_recos = len(recos_lists_df[self._item_col].iloc[0])

        recos_df_flat = pd.DataFrame({
            self._user_col: recos_lists_df[self._user_col].values.repeat(n_recos),
            self._item_col: np.concatenate(recos_lists_df[self._item_col].values),
            self._prediction_col: np.concatenate(recos_lists_df[self._prediction_col].values),
        })

        return recos_df_flat

    @staticmethod
    def _flat_df_to_lists(df, sort_col, group_col, n_cutoff, target_columns):
        order = [group_col] + target_columns
        return df[order]. \
            sort_values(sort_col, ascending=False). \
            groupby(group_col). \
            aggregate(lambda x: list(x)[:n_cutoff]). \
            reset_index()

    def _recos_flat_to_lists(self, df, n_cutoff=None):
        return self._flat_df_to_lists(
            df,
            n_cutoff=n_cutoff,
            sort_col=self._prediction_col,
            group_col=self._user_col,
            target_columns=[self._item_col, self._prediction_col])

    def _simil_flat_to_lists(self, df, n_cutoff=None):
        return self._flat_df_to_lists(
            df,
            n_cutoff=n_cutoff,
            sort_col=self._prediction_col,
            group_col=self._item_col_simil,
            target_columns=[self._item_col, self._prediction_col])

    def _format_results_df(self, source_vec, target_ids_mat, scores_mat, results_format):
        if 'recommendations' in results_format:
            source_col = self._user_col
            target_col = self._item_col
            scores_col = self._prediction_col
        elif 'similarities' in results_format:
            source_col = self._item_col_simil
            target_col = self._item_col
            scores_col = self._prediction_col
        else:
            raise NotImplementedError('results_format: ' + results_format)

        order = [source_col, target_col, scores_col]

        if 'lists' in results_format:
            return pd.DataFrame({
                source_col: source_vec,
                target_col: list(target_ids_mat),
                scores_col: list(scores_mat)
            })[order]
        elif 'flat' in results_format:
            n_rec = target_ids_mat.shape[1]
            return pd.DataFrame({
                source_col: np.array(source_vec).repeat(n_rec),
                target_col: np.concatenate(list(target_ids_mat)),
                scores_col: np.concatenate(list(scores_mat)),
            })[order]

        else:
            raise NotImplementedError('results_format: ' + results_format)

    def hyper_param_search_on_split_data(
            self, train_data, validation_data, hp_space,
            n_iters=100, random_search=0.9, **kwargs):

        bo = RecoBayesSearchHoldOut(search_space=hp_space,
                                    pipeline=self,
                                    loss=None,
                                    random_ratio=random_search,
                                    **kwargs)

        res_bo, best_params, best_model = \
            bo.optimize(data_dict={'training': train_data, 'validation': validation_data},
                        n_calls=n_iters)

        return SimpleNamespace(**{
            'optimizer': bo,
            'report': bo.best_results_summary(res_bo, percentile=0),
            'mutual_info': bo.params_mutual_info(),
            'result': res_bo,
            'best_params': best_params,
            'best_model': best_model})

    def pickle_to_file(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    def hyper_param_search(self, train_obs, hp_space, n_iters=100,
                           valid_ratio=0.04, random_search=0.9,
                           valid_split_time_col=None, **kwargs):

        train_obs_bo, valid_obs_bo = train_obs.split_train_test(
            ratio=valid_ratio ** 0.5 if valid_split_time_col is None else valid_ratio,
            users_ratio=valid_ratio ** 0.5 if valid_split_time_col is None else 1,
            time_split_column=valid_split_time_col,
            random_state=RANDOM_STATE)

        return self.hyper_param_search_on_split_data(
            train_data=train_obs_bo, validation_data=valid_obs_bo.df_obs,
            hp_space=hp_space, n_iters=n_iters, random_search=random_search, **kwargs)


class BaseDFSparseRecommender(BaseDFRecommender):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sparse_mat_builder = None
        self.train_mat = None
        self.external_features_mat = None

    def _set_data(self, train_obs, calc_train_mat=True):
        train_df = train_obs.df_obs
        self.sparse_mat_builder = train_obs.get_sparse_matrix_helper()
        self.all_users = train_df[self.sparse_mat_builder.uid_source_col].unique().astype(str)
        self.all_items = train_df[self.sparse_mat_builder.iid_source_col].unique().astype(str)
        # shuffling because np.unique() returns elements in almost sorted order by counts,
        # and it's probably not a good thing: it changes regional sparsity,
        # and at a later stage might be sampled / iterated in order
        np.random.shuffle(self.all_users)
        np.random.shuffle(self.all_items)
        if calc_train_mat:
            self.train_mat = self.sparse_mat_builder.build_sparse_interaction_matrix(train_df)

    def _reuse_data(self, other):
        self.all_users = other.all_users
        self.all_items = other.all_items
        self.sparse_mat_builder = other.sparse_mat_builder
        self.train_mat = other.train_mat

    def remove_unseen_users(self, user_ids, message_prefix=''):
        return self._filter_array(
            user_ids,
            filter_array=self.all_users,
            message_prefix=message_prefix,
            message_suffix='users that were not in training set.')

    def remove_unseen_items(self, item_ids, message_prefix=''):
        return self._filter_array(
            item_ids,
            filter_array=self.all_items,
            message_prefix=message_prefix,
            message_suffix='items that were not in training set.')

    @staticmethod
    def _filter_array(array, filter_array, message_prefix='', message_suffix=''):
        array = np.array(array).astype(str)
        relevance_mask = np.isin(array, np.array(filter_array).astype(str))
        n_discard = np.sum(~relevance_mask)
        if n_discard > 0:
            logger.info(
                '%s Discarding %d (out of %d) %s' %
                (message_prefix, int(n_discard), len(array), message_suffix))
        return array[relevance_mask]

    # def _remove_training_from_df(self, flat_df):
    #     flat_df = pd.merge(
    #         flat_df, self.train_df,
    #         left_on=[self._user_col, self._item_col],
    #         right_on=[self._user_col, self.sparse_mat_builder.iid_source_col],
    #         how='left')
    #     # keep only data that was present in left (recommendations) but no in right (training)
    #     flat_df = flat_df[flat_df[self._rating_col].isnull()] \
    #         [[self._user_col, self._item_col, self._prediction_col]]
    #     return flat_df

    @staticmethod
    def _remove_self_similarities(flat_df, col1, col2):
        return flat_df[flat_df[col1].values != flat_df[col2].values].copy()

    @staticmethod
    def _eval_on_test_by_ranking_LFM(train_ranks_func, test_tanks_func,
                                     test_dfs, test_names=('',), prefix='',
                                     include_train=True, k=10):
        """
        this is just to avoid the same flow twice (or more)
        :param train_ranks_func: function that return the ranks and sparse mat of training set
        :param test_tanks_func: function that return the ranks and sparse mat of a test set
        :param test_dfs: test dataframes
        :param test_names: test dataframes names
        :param prefix: prefix for this report
        :param include_train: whether to evaluate training or not
        :return: a report dataframe
        """

        # test
        if isinstance(test_dfs, pd.DataFrame):
            test_dfs = [test_dfs]

        report_dfs = []

        # train
        if include_train:
            ranks_mat, sp_mat = train_ranks_func()
            report_dfs.append(mean_scores_report_on_ranks(
                ranks_list=[ranks_mat], datasets=[sp_mat],
                dataset_names=[prefix + 'train'], k=k))

        for test_df, test_name in zip(test_dfs, test_names):
            ranks_mat, sp_mat = test_tanks_func(test_df)
            report_dfs.append(mean_scores_report_on_ranks(
                ranks_list=[ranks_mat], datasets=[sp_mat],
                dataset_names=[prefix + test_name + 'test'], k=k))

        report_df = pd.concat(report_dfs)
        return report_df

    def get_prediction_mat_builder_adapter(self, mat_builder: InteractionMatrixBuilder):
        mat_builder = deepcopy(mat_builder)
        mat_builder.uid_source_col = self._user_col
        mat_builder.iid_source_col = self._item_col
        mat_builder.rating_source_col = self._prediction_col
        return mat_builder

    @abstractmethod
    def _get_recommendations_flat(self, user_ids, n_rec, item_ids=None,
                                  exclude_training=True, pbar=None, **kwargs):
        return pd.DataFrame()

    def get_recommendations(
            self, user_ids=None, item_ids=None, n_rec=10,
            exclude_training=True, pbar=None,
            results_format='lists'):

        if user_ids is not None:
            user_ids = self.remove_unseen_users(user_ids, message_prefix='get_recommendations: ')
        else:
            user_ids = self.all_users

        if item_ids is not None:
            item_ids = self.remove_unseen_items(item_ids, message_prefix='get_recommendations: ')

        recos_flat = self._get_recommendations_flat(
            user_ids=user_ids, item_ids=item_ids, n_rec=n_rec,
            exclude_training=exclude_training, pbar=pbar)

        if results_format == 'flat':
            return recos_flat
        else:
            return self._recos_flat_to_lists(recos_flat, n_cutoff=n_rec)

    def _check_item_ids_args(self, item_ids, target_item_ids):
        item_ids = self.remove_unseen_items(item_ids) \
            if item_ids is not None else self.all_items
        target_item_ids = self.remove_unseen_items(target_item_ids) \
            if target_item_ids is not None else None
        return item_ids, target_item_ids

    def _check_user_ids_args(self, user_ids, target_user_ids):
        user_ids = self.remove_unseen_users(user_ids) \
            if user_ids is not None else self.all_users
        target_user_ids = self.remove_unseen_users(target_user_ids) \
            if target_user_ids is not None else None
        return user_ids, target_user_ids

    def eval_on_test_by_ranking(self, test_dfs, test_names=('',), prefix='rec ',
                                include_train=True, items_filter=None,
                                n_rec=100, k=10):
        @self.logging_decorator
        def relevant_users():
            # get only those users that are present in the evaluation dataframes
            all_test_users = []
            [all_test_users.extend(df[self._user_col].unique().tolist()) for df in test_dfs]
            all_test_users = np.array(all_test_users)
            return all_test_users

        if isinstance(test_dfs, pd.DataFrame):
            test_dfs = [test_dfs]
        elif isinstance(test_dfs, ObservationsDF):
            test_dfs = [test_dfs.df_obs]

        mat_builder = self.sparse_mat_builder
        pred_mat_builder = self.get_prediction_mat_builder_adapter(mat_builder)

        users = self.remove_unseen_users(relevant_users())

        if include_train:
            recos_flat_train = self.get_recommendations(
                user_ids=users,
                item_ids=None,
                n_rec=min(n_rec, mat_builder.n_cols),
                exclude_training=False,
                results_format='flat')

        recos_flat_test = self.get_recommendations(
            user_ids=users,
            item_ids=items_filter,
            n_rec=min(n_rec, mat_builder.n_cols),
            exclude_training=True,
            results_format='flat')

        ranks_all_test = pred_mat_builder.predictions_df_to_sparse_ranks(
            recos_flat_test)

        @self.logging_decorator
        def _get_training_ranks():
            users_inds = mat_builder.uid_encoder.transform(users)
            users_inds.sort()
            # train_df = self.train_df[self.train_df[self._user_col].isin(users)].copy()
            # sp_train = self.sparse_mat_builder. \
            #     build_sparse_interaction_matrix(train_df).tocsr()
            sp_train = mat_builder.crop_rows(self.train_mat, inds_stay=users_inds)
            sp_train_ranks = pred_mat_builder. \
                filter_all_ranks_by_sparse_selection(
                sp_train, pred_mat_builder.predictions_df_to_sparse_ranks(recos_flat_train))
            return sp_train_ranks, sp_train

        @self.logging_decorator
        def _get_test_ranks(test_df):
            sp_test = self.sparse_mat_builder. \
                build_sparse_interaction_matrix(test_df).tocsr()
            sp_test_ranks = pred_mat_builder. \
                filter_all_ranks_by_sparse_selection(sp_test, ranks_all_test)
            return sp_test_ranks, sp_test

        report_df = self._eval_on_test_by_ranking_LFM(
            train_ranks_func=_get_training_ranks,
            test_tanks_func=_get_test_ranks,
            test_dfs=test_dfs,
            test_names=test_names,
            prefix=prefix,
            include_train=include_train,
            k=k)

        return report_df


class RecoBayesSearchHoldOut(BayesSearchHoldOut, LogCallsTimeAndOutput):

    def __init__(self, metric='AUC', k=10, interrupt_message_file=None, verbose=True, **kwargs):
        self.metric = metric
        self.k = k
        self.interrupt_message_file = interrupt_message_file
        super().__init__(verbose=verbose, **kwargs)
        self.all_metrics = pd.DataFrame()

    def _check_interrupt(self):
        if self.interrupt_message_file is not None \
                and os.path.exists(self.interrupt_message_file):

            with open(self.interrupt_message_file) as f:
                message = f.readline()

            if 'stop' in message:
                raise InterruptedError('interrupted by "stop" message in %s'
                                       % self.interrupt_message_file)
            elif 'pause' in message:
                logger.warn('Paused by "pause" message in %s'
                            % self.interrupt_message_file)
                while 'pause' in message:
                    sleep(1)
                    with open(self.interrupt_message_file) as f:
                        message = f.readline()
                self._check_interrupt()

            elif 'update' in message:
                logger.warn('Updating HP space due to "update" message in %s'
                            % self.interrupt_message_file)
                raise NotImplementedError('not yet implemented')

    def _record_all_metrics(self, report_df, values, time_taken, target_loss):
        # records the time and the other metrics
        params_dict = self.values_to_dict(values)
        report_df['target_loss'] = target_loss
        report_df['time_taken'] = time_taken
        report_df = report_df.assign(**params_dict)
        self.all_metrics = self.all_metrics.append(report_df)

    def best_results_summary(self, res_bo, percentile=95):
        return self.all_metrics. \
            reset_index(). \
            drop('index', axis=1). \
            sort_values('target_loss')

    def params_mutual_info(self):
        mutual_info = {}
        loss = self.all_metrics['target_loss'].values.reshape(-1, 1)
        for feat in self.search_space.keys():
            vec = self.all_metrics[feat].values.reshape(-1, 1)
            try:
                mutual_info[feat] = mutual_info_regression(vec, loss)[0]
            except ValueError:  # categorical feature (string)
                mutual_info[feat] = mutual_info_regression(
                    LabelEncoder().fit_transform(vec).reshape(-1, 1), loss)[0]
        return pd.DataFrame([mutual_info])

    def objective_func(self, values):
        try:
            self._check_interrupt()
            pipeline = self.init_pipeline(values)
            start = time.time()
            pipeline.fit(self.data_dict['training'])
            report_df = pipeline.eval_on_test_by_ranking(
                self.data_dict['validation'], include_train=False, prefix='', k=self.k)
            loss = 1 - report_df.loc['test', self.metric]
            self._record_all_metrics(
                report_df=report_df,
                values=values,
                time_taken=time.time() - start,
                target_loss=loss)
            return loss

        except Exception as e:
            logger.exception(e)
            logger.error(values)
            # raise e
            return 1.0
