from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight

from ml_recsys_tools.recommenders.factorization_base import FactorizationRecommender
from ml_recsys_tools.utils.instrumentation import simple_logger


class ALSRecommender(FactorizationRecommender):

    default_fit_params = dict(
        use_bm25=False,
        bm25_k1=100,
        bm25_b=0.8,
    )

    default_model_params = dict(
        factors=100,
         regularization=0.01,
         iterations=15,
         calculate_training_loss=True,
         num_threads=0)

    def _get_item_factors(self, mode=None):
        return None, self.model.item_factors

    def _get_user_factors(self, mode=None):
        return None, self.model.user_factors

    def _prep_for_fit(self, train_obs, **fit_params):
        self._set_data(train_obs)
        self._set_fit_params(fit_params)
        self.model = AlternatingLeastSquares(**self.model_params)

    def set_params(self, **params):
        params = self._pop_set_dict(self.fit_params, params, self.default_fit_params.keys())
        super().set_params(**params)

    def fit(self, train_obs, **fit_params):
        self._prep_for_fit(train_obs, **fit_params)
        # implib ALS expects matrix in items x users format
        als_train_mat = self.train_mat.T
        if self.fit_params['use_bm25']:
            als_train_mat = bm25_weight(
                als_train_mat,
                K1=self.fit_params['bm25_k1'],
                B=self.fit_params['bm25_b'])
        self.model.fit(als_train_mat)