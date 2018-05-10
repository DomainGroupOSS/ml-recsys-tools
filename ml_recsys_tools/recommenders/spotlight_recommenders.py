from spotlight.interactions import Interactions
from spotlight.factorization.implicit import ImplicitFactorizationModel

from ml_recsys_tools.recommenders.factorization_base import BaseFactorizationRecommender


def spotlight_interactions_from_sparse(sp_mat):
    sp_mat = sp_mat.tocoo()
    return Interactions(user_ids=sp_mat.row, item_ids=sp_mat.col, ratings=sp_mat.data)


class SpotlightImplicitRecommender(BaseFactorizationRecommender):

    default_model_params = dict(
        loss='pointwise',  # 'bpr', 'hinge', 'adaptive hinge'
        embedding_dim=128,
        n_iter=10,
        batch_size=256,
        l2=0.0,
        learning_rate=1e-2,
        sparse=False,
        num_negative_samples=5)

    default_fit_params = dict(
        verbose=True)

    def _prep_for_fit(self, train_obs, **fit_params):
        self._set_data(train_obs)
        self.set_params(**fit_params)
        self.model = ImplicitFactorizationModel(**self.model_params)
        self._set_spotlight_train_data(self.train_mat)

    def _set_spotlight_train_data(self, train_mat):
        self.spotlight_dataset = spotlight_interactions_from_sparse(train_mat)

    # def set_params(self, **params):
    #     params = self._pop_set_dict(
    #         self.fit_params, params, self.default_fit_params.keys())
    #     super().set_params(**params)

    def fit(self, train_obs, **fit_params):
        self._prep_for_fit(train_obs, **fit_params)
        self.model.fit(self.spotlight_dataset, **self.fit_params)

    def fit_partial(self, train_obs, epochs=1):
        raise NotImplementedError()
        # self._set_epochs(epochs)
        # if self.model is None:
        #     self.fit(train_obs)
        # else:
        #     self.model.fit(self.spotlight_dataset)
        # return self

    def _set_epochs(self, epochs):
        self.set_params(n_iter=epochs)

    def _predict(self, user_ids, item_ids):
        return self.model.predict(user_ids, item_ids)

    def _get_recommendations_flat(
            self, user_ids, n_rec, item_ids=None, exclude_training=True,
            pbar=None, item_features_mode=None, use_biases=True):
        return self.get_recommendations_exact(
            user_ids=user_ids, item_ids=item_ids, n_rec=n_rec,
            exclude_training=exclude_training, results_format='flat')

    def _get_item_factors(self, mode=None):
        raise NotImplementedError()

    def _get_user_factors(self, mode=None):
        raise NotImplementedError()

    def _predict_rank(self, test_mat, train_mat=None):
        raise NotImplementedError()
