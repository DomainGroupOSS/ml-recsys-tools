from implicit.als import AlternatingLeastSquares
# from implicit.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import bm25_weight

from ml_recsys_tools.recommenders.factorization_base import FactorizationRecommender
from ml_recsys_tools.utils.instrumentation import simple_logger


class ALSRecommender(FactorizationRecommender):

    default_fit_params = dict(
        use_bm25=False,
        bm25_k1=100,
        bm25_b=0.8,
        cg_steps=3,
        regularization=0.01,
    )

    default_model_params = dict(
        factors=100,
        iterations=15,
        calculate_training_loss=True,
        num_threads=0)

    def _get_item_factors(self, mode=None):
        return None, self.model.item_factors

    def _get_user_factors(self, mode=None):
        return None, self.model.user_factors

    def _prep_for_fit(self, train_obs, **fit_params):
        self._set_data(train_obs)
        self.set_params(**fit_params)
        self.model = AlternatingLeastSquares(**self.model_params)
        self._set_implib_train_mat()

    def _set_implib_train_mat(self):
        # implib ALS expects matrix in items x users format
        self.implib_train_mat = self.train_mat.T
        if self.fit_params['use_bm25']:
            self.implib_train_mat = bm25_weight(
                self.implib_train_mat,
                K1=self.fit_params['bm25_k1'],
                B=self.fit_params['bm25_b'])
        self.model.regularization = \
            self.fit_params['regularization'] * self.implib_train_mat.nnz
        self.model.cg_steps = self.fit_params['cg_steps']

    def set_params(self, **params):
        params = self._pop_set_dict(
            self.fit_params, params, self.default_fit_params.keys())
        super().set_params(**params)

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

    def _set_epochs(self, epochs):
        self.set_params(iterations=epochs)
        if self.model is not None:
            self.model.iterations = epochs


