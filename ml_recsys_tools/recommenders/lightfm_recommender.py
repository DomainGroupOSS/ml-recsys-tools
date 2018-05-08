from copy import deepcopy

import numpy as np
import pandas as pd
from functools import partial

from lightfm import LightFM
import lightfm.lightfm

from ml_recsys_tools.data_handlers.interaction_handlers_base import RANDOM_STATE
from ml_recsys_tools.recommenders.factorization_base import FactorizationRecommender
from ml_recsys_tools.utils.automl import early_stopping_runner
from ml_recsys_tools.utils.instrumentation import simple_logger
from ml_recsys_tools.utils.parallelism import map_batches_multiproc, N_CPUS
from ml_recsys_tools.utils.similarity import top_N_sorted


# monkey patch print function
def _epoch_logger(s, print_each_n=20):
    try:
        if not int(s.replace('Epoch ', '')) % print_each_n:
            simple_logger.info(s)
    except Exception as e:
        simple_logger.error('Failed in _epoch_logger: %s' % str(e))
        simple_logger.info(s)

lightfm.lightfm.print = _epoch_logger


class LightFMRecommender(FactorizationRecommender):

    default_fit_params = {
        'epochs': 100,
        'item_features': None,
        'num_threads': N_CPUS,
        'verbose': True,
    }

    default_model_params = {
        'loss': 'warp',
        'learning_schedule': 'adadelta',
        'no_components': 10,
        'max_sampled': 10,
        'item_alpha': 0,
        'user_alpha': 0,
    }

    def __init__(self,
                 use_sample_weight=False,
                 external_features=None,
                 external_features_params=None, **kwargs):
        self.use_sample_weight = use_sample_weight
        self.external_features = external_features
        self.external_features_params = external_features_params or {}
        super().__init__(**kwargs)

    def _prep_for_fit(self, train_obs, **fit_params):
        # assign all observation data
        self._set_data(train_obs)
        fit_params['sample_weight'] = self.train_mat.tocoo() \
            if self.use_sample_weight else None
        self._set_fit_params(fit_params)
        self._add_external_features()
        # init model and set params
        self.model = LightFM(**self.model_params)

    def _add_external_features(self):
        if self.external_features is not None:
            self.external_features_mat = self.external_features.\
                create_items_features_matrix(
                    items_encoder=self.sparse_mat_builder.iid_encoder,
                    **self.external_features_params)
            simple_logger.info('External item features matrix: %s' %
                            str(self.external_features_mat.shape))

        # add external features if specified
        self.fit_params['item_features'] = self.external_features_mat
        if self.external_features_mat is not None:
            simple_logger.info('Fitting using external features mat: %s'
                               % str(self.external_features_mat.shape))

    def fit(self, train_obs, **fit_params):
        self._prep_for_fit(train_obs, **fit_params)
        self.model.fit(self.train_mat, **self.fit_params)
        return self

    def fit_partial(self, train_obs, epochs=1):
        self._set_epochs(epochs)
        if self.model is None:
            self.fit(train_obs)
        else:
            self.model.fit_partial(self.train_mat)
        return self

    def fit_batches(self, train_obs, train_dfs, epochs_per_batch=None, **fit_params):
        self._prep_for_fit(train_obs)
        for i, df in enumerate(train_dfs):
            batch_train_mat = self.sparse_mat_builder.build_sparse_interaction_matrix(df)

            if epochs_per_batch is not None:
                fit_params['epochs'] = epochs_per_batch

            fit_params['sample_weight'] = batch_train_mat.tocoo() \
                if self.use_sample_weight else None

            self._set_fit_params(fit_params)

            simple_logger.info('Fitting batch %d (%d interactions)' % (i, len(df)))
            self.model.fit_partial(batch_train_mat, **self.fit_params)

    def _set_epochs(self, epochs):
        self.set_params(epochs=epochs)

    def set_params(self, **params):
        """
        this is for skopt / sklearn compatibility
        """
        if 'epochs' in params:
            self._set_fit_params({'epochs': params.pop('epochs')})
        params = self._pop_set_params(
            params, ['use_sample_weight', 'external_features', 'external_features_params'])
        super().set_params(**params)

    def _get_item_factors(self, mode=None):

        n_items = len(self.sparse_mat_builder.iid_encoder.classes_)

        biases, representations = self.model.get_item_representations(self.fit_params['item_features'])

        if mode is None:
            pass  # default mode

        elif mode == 'external_features':
            external_features_mat = self.external_features_mat

            assert external_features_mat is not None, \
                'Must define and add a feature matrix for "external_features" similarity.'

            representations = external_features_mat

        elif (mode == 'no_features') and (self.fit_params['item_features'] is not None):

            simple_logger.info('LightFM recommender: get_similar_items: "no_features" mode '
                               'assumes ID mat was added and is the last part of the feature matrix.')

            assert self.model.item_embeddings.shape[0] > n_items, \
                'Either no ID matrix was added, or no features added'

            representations = self.model.item_embeddings[-n_items:, :]

        else:
            raise ValueError('Uknown representation mode: %s' % mode)

        return biases, representations

    def _get_user_factors(self, mode=None):
        return self.model.get_user_representations()

    def predict_on_df(self, df):
        mat_builder = self.get_prediction_mat_builder_adapter(self.sparse_mat_builder)
        df = mat_builder.remove_unseen_labels(df)
        df = mat_builder.add_encoded_cols(df)
        df[self._prediction_col] = self.model.predict(
            df[mat_builder.uid_col].values,
            df[mat_builder.iid_col].values,
            item_features=self.fit_params['item_features'],
            num_threads=self.fit_params['num_threads'])
        df.drop([mat_builder.uid_col, mat_builder.iid_col], axis=1, inplace=True)
        return df

    def eval_on_test_by_ranking_exact(self, test_dfs, test_names=('',),
                                      prefix='lfm ', include_train=True, k=10):

        @self.logging_decorator
        def _get_training_ranks():
            ranks_mat = self.model.predict_rank(
                self.train_mat,
                item_features=self.fit_params['item_features'],
                num_threads=self.fit_params['num_threads'])
            return ranks_mat, self.train_mat

        @self.logging_decorator
        def _get_test_ranks(test_df):
            test_sparse = self.sparse_mat_builder.build_sparse_interaction_matrix(test_df)
            ranks_mat = self.model.predict_rank(
                test_sparse, train_interactions=self.train_mat,
                item_features=self.fit_params['item_features'],
                num_threads=self.fit_params['num_threads'])
            return ranks_mat, test_sparse

        return self._eval_on_test_by_ranking_LFM(
            train_ranks_func=_get_training_ranks,
            test_tanks_func=_get_test_ranks,
            test_dfs=test_dfs,
            test_names=test_names,
            prefix=prefix,
            include_train=include_train)

    def get_recommendations_exact(
            self, user_ids, n_rec=10, exclude_training=True, chunksize=200, results_format='lists'):

        calc_func = partial(
            self._get_recommendations_exact,
            n_rec=n_rec,
            exclude_training=exclude_training,
            results_format=results_format)

        chunksize = int(35000 * chunksize / self.sparse_mat_builder.n_cols)

        ret = map_batches_multiproc(
            calc_func, user_ids, chunksize=chunksize, pbar='get_recommendations_exact_and_slow')
        return pd.concat(ret, axis=0)

    def _predict_for_users_dense(self, user_ids, exclude_training):

        mat_builder = self.sparse_mat_builder
        n_items = mat_builder.n_cols

        user_inds = mat_builder.uid_encoder.transform(user_ids)

        n_users = len(user_inds)
        user_inds_mat = user_inds.repeat(n_items)
        item_inds_mat = np.tile(np.arange(n_items), n_users)

        full_pred_mat = self.model.predict(
            user_inds_mat,
            item_inds_mat,
            item_features=self.fit_params['item_features'],
            num_threads=self.fit_params['num_threads']). \
            reshape((n_users, n_items))

        train_mat = self.train_mat.tocsr()

        if exclude_training:
            train_mat.sort_indices()
            for pred_ind, user_ind in enumerate(user_inds):
                train_inds = train_mat.indices[
                             train_mat.indptr[user_ind]: train_mat.indptr[user_ind + 1]]
                full_pred_mat[pred_ind, train_inds] = -np.inf

        return full_pred_mat

    def _get_recommendations_exact(self, user_ids, n_rec=10, exclude_training=True,
                                   results_format='lists'):

        full_pred_mat = self._predict_for_users_dense(user_ids, exclude_training=exclude_training)

        top_inds, top_scores = top_N_sorted(full_pred_mat, n=n_rec)

        item_ids = self.sparse_mat_builder.iid_encoder.inverse_transform(top_inds)

        return self._format_results_df(
            source_vec=user_ids, target_ids_mat=item_ids,
            scores_mat=top_scores, results_format='recommendations_' + results_format)
