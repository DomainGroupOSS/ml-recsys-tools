from lightfm import LightFM
import lightfm.lightfm

from ml_recsys_tools.recommenders.factorization_base import BaseFactorizationRecommender
from ml_recsys_tools.utils.instrumentation import simple_logger
from ml_recsys_tools.utils.parallelism import N_CPUS


# monkey patch print function
def _epoch_logger(s, print_each_n=20):
    try:
        if not int(s.replace('Epoch ', '')) % print_each_n:
            simple_logger.info(s)
    except Exception as e:
        simple_logger.error('Failed in _epoch_logger: %s' % str(e))
        simple_logger.info(s)

lightfm.lightfm.print = _epoch_logger


class LightFMRecommender(BaseFactorizationRecommender):

    default_model_params = {
        'loss': 'warp',
        'learning_schedule': 'adadelta',
        'no_components': 10,
        'max_sampled': 10,
        'item_alpha': 0,
        'user_alpha': 0,
    }

    default_fit_params = {
        'epochs': 100,
        'item_features': None,
        'num_threads': N_CPUS,
        'verbose': True,
    }

    default_external_features_params = dict(add_identity_mat=True)

    def __init__(self,
                 use_sample_weight=False,
                 external_features=None,
                 external_features_params=None, **kwargs):
        self.use_sample_weight = use_sample_weight
        self.external_features = external_features
        self.external_features_params = external_features_params or \
                                        self.default_external_features_params.copy()

        super().__init__(**kwargs)

    def _prep_for_fit(self, train_obs, **fit_params):
        # self.toggle_mkl_blas_1_thread(True)
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
                fit_transform_ids_df_to_mat(
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

    def _predict(self, user_ids, item_ids):
        return self.model.predict(user_ids, item_ids,
                                  item_features=self.fit_params['item_features'],
                                  num_threads=self.fit_params['num_threads'])


    def _predict_rank(self, test_mat, train_mat=None):
        return self.model.predict_rank(
            test_interactions=test_mat,
            train_interactions=train_mat,
            item_features=self.fit_params['item_features'],
            num_threads=self.fit_params['num_threads'])

