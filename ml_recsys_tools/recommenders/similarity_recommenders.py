from copy import deepcopy

import numpy as np

from ml_recsys_tools.data_handlers.interactions_with_features import ItemsHandler
from ml_recsys_tools.recommenders.recommender_base import BaseDFSparseRecommender
from ml_recsys_tools.utils.parallelism import map_batches_multiproc
from ml_recsys_tools.utils.similarity import top_N_sorted_on_sparse, top_N_sorted, most_similar


class BaseSimilarityRecommeder(BaseDFSparseRecommender):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.similarity_mat = None

    def _prep_for_fit(self, train_obs, **fit_params):
        self._set_fit_params(fit_params)
        self._set_data(train_obs)

    def _check_no_negatives(self):
        # prevents negative scores from being smaller than sparse zeros (e.g. for euclidean similarity)
        if len(self.similarity_mat.data) and np.min(self.similarity_mat.data) < 0.01:
            self.similarity_mat.data += np.abs(np.min(self.similarity_mat.data) - 0.01)

    def _predict_on_inds_dense(self, user_inds, item_inds):
        user_pred_mat = self.train_mat[user_inds, :] * self.similarity_mat
        return user_pred_mat[:, item_inds].toarray()

    def _get_recommendations_flat(
            self, user_ids, item_ids=None, exclusions=True,
            n_rec=100, **kwargs):

        self._check_no_negatives()

        user_inds = self.user_inds(user_ids)

        if item_ids is not None:
            item_inds = self.item_inds(item_ids)
            item_inds.sort()
        else:
            item_inds = None

        def best_for_batch(u_inds):
            user_pred_mat = self.train_mat[u_inds, :] * self.similarity_mat
            if exclusions:
                user_pred_mat -= self.exclude_mat[u_inds, :] * np.inf
            if item_inds is not None:
                user_pred_mat = user_pred_mat[:, item_inds]
            return top_N_sorted(user_pred_mat.toarray(), n_rec)

        ret = map_batches_multiproc(best_for_batch, user_inds, chunksize=200)

        best_inds = np.concatenate([r[0] for r in ret], axis=0)
        best_scores = np.concatenate([r[1] for r in ret], axis=0)

        # back to ids
        if item_inds is not None:
            best_inds = item_inds[best_inds.astype(int)]

        best_ids = self.sparse_mat_builder.iid_encoder.\
            inverse_transform(best_inds.astype(int))

        return self._format_results_df(
            source_vec=user_ids, target_ids_mat=best_ids, scores_mat=best_scores,
            results_format='recommendations_flat')

    def get_similar_items(self, item_ids=None, target_item_ids=None,
                          n_simil=10, results_format='lists', **kwargs):

        item_ids, target_item_ids = \
            self._check_item_ids_args(item_ids, target_item_ids)

        self._check_no_negatives()

        best_ids, best_scores = top_N_sorted_on_sparse(
            source_ids=item_ids,
            target_ids=target_item_ids,
            encoder=self.sparse_mat_builder.iid_encoder,
            sparse_mat=self.similarity_mat,
            n_top=n_simil,
        )

        simil_df = self._format_results_df(
            item_ids, target_ids_mat=best_ids,
            scores_mat=best_scores, results_format='similarities_' + results_format)
        return simil_df


class SimilarityDFRecommender(BaseSimilarityRecommeder):

    def get_similarity_builder(self):
        # this is hacky, but I want to make sure this recommender is even useful first
        # should be some other class' method
        simil_mat_builder = deepcopy(self.sparse_mat_builder)
        simil_mat_builder.uid_source_col = self._item_col_simil
        simil_mat_builder.iid_source_col = self._item_col
        simil_mat_builder.rating_source_col = self._prediction_col
        simil_mat_builder.n_rows = simil_mat_builder.n_cols
        simil_mat_builder.uid_encoder = simil_mat_builder.iid_encoder
        return simil_mat_builder

    def _prep_for_fit(self, train_obs, **fit_params):
        super()._prep_for_fit(train_obs, **fit_params)
        self.similarity_mat_builder = self.get_similarity_builder()

    def fit(self, train_obs, simil_df_flat, **fit_params):
        self._prep_for_fit(train_obs, **fit_params)
        self.similarity_mat = self.similarity_mat_builder. \
            build_sparse_interaction_matrix(simil_df_flat)
        self.similarity_mat += self.similarity_mat.T

    def continue_fit(self, simil_df_flat):
        partial_similarity_mat = self.similarity_mat_builder. \
            build_sparse_interaction_matrix(simil_df_flat)
        self.similarity_mat += partial_similarity_mat + partial_similarity_mat.T


class FeaturesSimilRecommender(SimilarityDFRecommender):

    def __init__(self,
                 numeric_n_bins=32,
                 simil_mode='cosine',
                 n_simil=100,
                 selection_filter=None,
                 numeric_cols=None,
                 categirical_cols=None,
                 binary_cols=None,
                 **kwargs):
        self.numeric_n_bins = numeric_n_bins
        self.simil_mode = simil_mode
        self.n_simil = n_simil
        self.selection_filter = selection_filter
        self.numeric_cols = numeric_cols
        self.categirical_cols = categirical_cols
        self.binary_cols = binary_cols
        super().__init__(**kwargs)

    def set_params(self, **params):
        params = self._pop_set_params(
            params, ['numeric_n_bins', 'simil_mode', 'n_simil'])
        super().set_params(**params)

    def fit(self, train_obs: ItemsHandler, **fit_params):
        self._prep_for_fit(train_obs, **fit_params)
        self.item_featuriser = train_obs.get_item_features(
            selection_filter=self.selection_filter,
            num_cols=self.numeric_cols,
            cat_cols=self.categirical_cols,
            bin_cols=self.binary_cols)

        self.item_feat_mat = self.item_featuriser.create_sparse_features_mat(
            items_encoder=self.sparse_mat_builder.iid_encoder,
            numeric_n_bins=self.numeric_n_bins,
            add_identity_mat=False,
            **fit_params)

        simil_df = self.get_similar_items(
            self.all_items,
            n_simil=self.n_simil,
            results_format='flat',
            simil_mode=self.simil_mode)

        super().fit(train_obs, simil_df, **fit_params)
        return self

    def get_similar_items(self, item_ids=None, target_item_ids=None, n_simil=10,
                          remove_self=True, embeddings_mode=None,
                          simil_mode='cosine', results_format='lists'):

        item_ids, target_item_ids = self._check_item_ids_args(item_ids, target_item_ids)

        simil_mat = self.item_feat_mat

        best_ids, best_scores = most_similar(
            source_ids=item_ids,
            target_ids=target_item_ids,
            source_encoder=self.sparse_mat_builder.iid_encoder,
            source_mat=simil_mat,
            n=n_simil+1 if remove_self else n_simil,
            simil_mode=simil_mode,
        )

        simil_df = self._format_results_df(
            item_ids, target_ids_mat=best_ids,
            scores_mat=best_scores, results_format='similarities_flat')

        if remove_self:
            simil_df = self._remove_self_similarities(
                simil_df, col1=self._item_col_simil, col2=self._item_col)

        if 'lists' in results_format:
            simil_df = self._simil_flat_to_lists(simil_df, n_cutoff=n_simil)

        return simil_df
