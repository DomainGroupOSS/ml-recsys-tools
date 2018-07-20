from implicit.als import AlternatingLeastSquares
# from implicit.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import bm25_weight

from ml_recsys_tools.recommenders.factorization_base import BaseFactorizationRecommender
from ml_recsys_tools.utils.instrumentation import simple_logger


class ALSRecommender(BaseFactorizationRecommender):

    default_model_params = dict(
        factors=200,
        iterations=2,
        calculate_training_loss=True,
        num_threads=0)

    default_fit_params = dict(
        use_bm25=True,
        bm25_k1=100,
        bm25_b=0.15,
        cg_steps=4,
        regularization=0,
    )

    def _get_item_factors(self, mode=None):
        return None, self.model.item_factors

    def _get_user_factors(self, mode=None):
        return None, self.model.user_factors

    def _prep_for_fit(self, train_obs, **fit_params):
        # self.toggle_mkl_blas_1_thread(True)
        self._set_data(train_obs)
        self.set_params(**fit_params)
        self.model = AlternatingLeastSquares(**self.model_params)
        self.model.cg_steps = self.fit_params['cg_steps']  # not passable to __init__()
        self._set_implib_train_mat(self.train_mat)

    def _set_implib_train_mat(self, train_mat):
        # implib ALS expects matrix in items x users format
        self.implib_train_mat = train_mat.T
        if self.fit_params['use_bm25']:
            self.implib_train_mat = bm25_weight(
                self.implib_train_mat,
                K1=self.fit_params['bm25_k1'],
                B=self.fit_params['bm25_b'])
        self.model.regularization = \
            self.fit_params['regularization'] * self.implib_train_mat.nnz

    def fit(self, train_obs, **fit_params):
        self._prep_for_fit(train_obs, **fit_params)
        self.model.fit(self.implib_train_mat)

    def fit_partial(self, train_obs, epochs=1):
        self._set_epochs(epochs)
        if self.model is None:
            self.fit(train_obs)
        else:
            self.model.fit(self.implib_train_mat)
        return self

    def fit_batches(self, train_obs, train_dfs, epochs_per_batch=None, **fit_params):
        self._prep_for_fit(train_obs)
        for i, df in enumerate(train_dfs):
            batch_train_mat = self.sparse_mat_builder.build_sparse_interaction_matrix(df)

            if epochs_per_batch is not None:
                fit_params['iterations'] = epochs_per_batch
            else:
                fit_params['iterations'] = 1
                self.model.cg_steps = 1
            self._set_fit_params(fit_params)

            self._set_implib_train_mat(batch_train_mat)

            simple_logger.info('Fitting batch %d (%d interactions)' % (i, len(df)))
            self.model.fit(self.implib_train_mat)

    def _set_epochs(self, epochs):
        self.set_params(iterations=epochs)
        if self.model is not None:
            self.model.iterations = epochs

    def _predict_on_inds(self, user_inds, item_inds):
        raise NotImplementedError()

    def _predict_rank(self, test_mat, train_mat=None):
        raise NotImplementedError()


