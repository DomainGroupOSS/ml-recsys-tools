import pandas as pd
import numpy as np

from spotlight.interactions import Interactions
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.sequence.representations import CNNNet

from ml_recsys_tools.recommenders.factorization_base import BaseFactorizationRecommender
from ml_recsys_tools.recommenders.recommender_base import BaseDFSparseRecommender
from ml_recsys_tools.utils.instrumentation import collect_named_init_params
from ml_recsys_tools.utils.similarity import top_N_sorted


def spotlight_interactions_from_sparse(sp_mat):
    sp_mat = sp_mat.tocoo()
    return Interactions(user_ids=sp_mat.row, item_ids=sp_mat.col, ratings=sp_mat.data)


class EmbeddingFactorsRecommender(BaseFactorizationRecommender):

    default_model_params = dict(
        loss='adaptive_hinge',  # 'bpr', 'hinge', 'adaptive hinge'
        embedding_dim=32,
        n_iter=10,
        batch_size=256,
        l2=0.0,
        learning_rate=1e-2,
        num_negative_samples=5)

    default_fit_params = dict(
        verbose=True)

    def _prep_for_fit(self, train_obs, **fit_params):
        # self.toggle_mkl_blas_1_thread(False)
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

    def _predict_on_inds(self, user_inds, item_inds):
        return self.model.predict(user_inds, item_inds)

    def _get_item_factors(self, mode=None):
        return self.model._net.item_biases.weight.data.numpy().ravel(), \
               self.model._net.item_embeddings.weight.data.numpy()

    def _get_user_factors(self, mode=None):
        return self.model._net.user_biases.weight.data.numpy().ravel(), \
               self.model._net.user_embeddings.weight.data.numpy()

    def _predict_rank(self, test_mat, train_mat=None):
        raise NotImplementedError()


class SequenceEmbeddingRecommender(BaseDFSparseRecommender):

    default_model_params = dict(
        loss='adaptive_hinge',  # 'pointwise', 'bpr', 'hinge', 'adaptive_hinge'
        representation='lstm',  # 'pooling', 'cnn', 'lstm', 'mixture'
        embedding_dim=32,
        n_iter=4,
        batch_size=64,
        l2=0.0,
        learning_rate=2e-3,
        num_negative_samples=25
    )

    default_fit_params = dict(
        max_sequence_length=200,
        timestamp_col='first_timestamp'
    )

    def _interactions_sequence_from_obs(
            self,
            obs,
            timestamp_col='first_timestamp',
            max_sequence_length=10,
            min_sequence_length=None,
            step_size=None,
            **kwargs):

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

    def _prep_for_fit(self, train_obs, **fit_params):
        # self.toggle_mkl_blas_1_thread(False)
        self._set_data(train_obs)
        self._set_fit_params(fit_params)
        self.sequence_interactions = \
            self._interactions_sequence_from_obs(train_obs, **self.fit_params)

    def fit(self, train_obs, **fit_params):
        self._prep_for_fit(train_obs, **fit_params)
        self.model = ImplicitSequenceModel(**self.model_params)
        self.model.fit(self.sequence_interactions)

    def _get_recommendations_flat(self, user_ids, n_rec, item_ids=None,
                                  exclude_training=True, **kwargs):

        return self._get_recommendations_exact(
            user_ids=user_ids, item_ids=item_ids, n_rec=n_rec,
            exclude_training=exclude_training, results_format='flat')

    def _predict_on_inds_dense(self, user_inds, item_inds):
        sequences = self.sequence_interactions.sequences

        pred_mat = np.zeros((len(user_inds), len(item_inds)))

        # TODO: very SLOW, try multiproc (batched from caller)

        item_inds_spot = item_inds.reshape(-1, 1) + 1

        for i_row, user_ind in enumerate(user_inds):
            pred_mat[i_row, :] = self.model.predict(sequences[user_ind], item_ids=item_inds_spot)

        return pred_mat


class CNNEmbeddingRecommender(SequenceEmbeddingRecommender):

    cnn_default_params = dict(
        kernel_width=9,
        dilation=1,
        num_layers=3,
        nonlinearity='relu',
        residual_connections=True)

    default_fit_params = dict(
        **SequenceEmbeddingRecommender.default_fit_params,
        **cnn_default_params
    )

    def _cnn_net(self, interactions):
        cnn_params = dict(
            num_items=interactions.num_items,
            embedding_dim=self.model_params['embedding_dim'],
            **{k:v for k, v in self.fit_params.items()
               if k in collect_named_init_params(CNNNet)['CNNNet']})

        # updating dilation parameter to be the right tuple
        cnn_params['dilation'] = [cnn_params['dilation']*(2**i)
                                      for i in range(cnn_params['num_layers'])]

        return CNNNet(**cnn_params)

    def _prep_for_fit(self, train_obs, **fit_params):
        super()._prep_for_fit(train_obs, **fit_params)
        self.model_params['representation'] = self._cnn_net(self.sequence_interactions)

