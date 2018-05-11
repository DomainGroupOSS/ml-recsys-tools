import pandas as pd
import numpy as np
from spotlight.evaluation import FLOAT_MAX

from spotlight.interactions import Interactions
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.sequence.implicit import ImplicitSequenceModel

from ml_recsys_tools.recommenders.factorization_base import BaseFactorizationRecommender
from ml_recsys_tools.recommenders.recommender_base import BaseDFSparseRecommender
from ml_recsys_tools.utils.similarity import top_N_sorted


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

    def fit(self, train_obs, **fit_params):
        self._prep_for_fit(train_obs, **fit_params)
        self.model.fit(self.spotlight_dataset, **self.fit_params)

    def fit_partial(self, train_obs, epochs=1):
        self._set_epochs(epochs)
        if self.model is None:
            self.fit(train_obs)
        else:
            self.model.fit(self.spotlight_dataset)
        return self

    def _set_epochs(self, epochs):
        self.set_params(n_iter=epochs)

    def _predict(self, user_ids, item_ids):
        return self.model.predict(user_ids, item_ids)

    def _get_item_factors(self, mode=None):
        return self.model._net.item_biases.weight.data.numpy().ravel(), \
               self.model._net.item_embeddings.weight.data.numpy()

    def _get_user_factors(self, mode=None):
        return self.model._net.user_biases.weight.data.numpy().ravel(), \
               self.model._net.user_embeddings.weight.data.numpy()

    def _predict_rank(self, test_mat, train_mat=None):
        raise NotImplementedError()


class SequenceRecommender(BaseDFSparseRecommender):

    default_model_params = dict(
        n_iter=3,
        representation='cnn',
        loss='bpr')

    def _interactions_sequence_from_obs(self,
            obs, timestamp_col='first_timestamp',
            max_sequence_length=10, min_sequence_length=None, step_size=None):
        obs.timestamp_col = timestamp_col
        return Interactions(
            user_ids=self.sparse_mat_builder.uid_encoder.
                transform(obs.user_ids.astype(str)).astype('int32'),
            item_ids=self.sparse_mat_builder.iid_encoder.
                transform(obs.item_ids.astype(str)).astype('int32') + 1,
            ratings=obs.ratings,
            timestamps=obs.timestamps
        ). \
            to_sequence(
            max_sequence_length=max_sequence_length,
            min_sequence_length=min_sequence_length,
            step_size=step_size
        )

    def fit(self, train_obs, **kwargs):
        self._set_data(train_obs)
        self.sequence_interactions = self._interactions_sequence_from_obs(train_obs)
        self.model = ImplicitSequenceModel(**self.model_params)
        self.model.fit(self.sequence_interactions)

    def _predict(self, item_ids=None, exclude_training=True):

        if item_ids is None:
            item_ids = self.sparse_mat_builder.iid_encoder.classes_

        item_inds = self.sparse_mat_builder.iid_encoder.transform(item_ids) + 1

        sequences = self.sequence_interactions.sequences

        for i in range(len(sequences)):

            predictions = self.model.predict(sequences[i], item_ids=item_inds)

            if exclude_training:
                predictions[sequences[i]] = FLOAT_MAX

        return NotImplementedError()



    def _get_recommendations_flat(self, user_ids, n_rec, item_ids=None,
                                  exclude_training=True, pbar=None, **kwargs):

        full_pred_mat = self._predict(item_ids=item_ids, exclude_training=exclude_training)

        top_inds, top_scores = top_N_sorted(full_pred_mat, n=n_rec)

        best_ids = self.sparse_mat_builder.iid_encoder.inverse_transform(top_inds - 1)

        return self._format_results_df(
            source_vec=user_ids, target_ids_mat=best_ids,
            scores_mat=top_scores, results_format='recommendations_flat')