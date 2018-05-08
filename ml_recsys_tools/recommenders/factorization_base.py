from abc import abstractmethod

import numpy as np

from ml_recsys_tools.recommenders.recommender_base import BaseDFSparseRecommender
from ml_recsys_tools.utils.similarity import most_similar


class FactorizationRecommender(BaseDFSparseRecommender):

    @abstractmethod
    def _get_item_factors(self, mode=None):
        return  np.array([0]), np.array([0])

    @abstractmethod
    def _get_user_factors(self, mode=None):
        return np.array([0]), np.array([0])

    @abstractmethod
    def _prep_for_fit(self, train_obs, **fit_params):
        pass

    def get_similar_items(self, item_ids=None, target_item_ids=None, n_simil=10,
                          remove_self=True, embeddings_mode=None,
                          simil_mode='cosine', results_format='lists', pbar=None):
        """
        uses learned embeddings to get N most similar items

        :param item_ids: vector of item IDs
        :param n_simil: number of most similar items to retrieve
        :param remove_self: whether to remove the the query items from the lists (similarity to self should be maximal)
        :param embeddings_mode: the item representations to use for calculation:
             None (default) - means full representations
             'external_features' - calculation based only external features (assumes those exist)
             'no_features' - calculation based only on internal features (assumed identity mat was part of the features)
        :param simil_mode: mode of similairyt calculation:
            'cosine' (default) - cosine similarity bewtween representations (normalized dot product with no biases)
            'dot' - unnormalized dot product with addition of biases
            'euclidean' - inverse of euclidean distance
            'cooccurance' - no usage of learned features - just cooccurence of items matrix
                (number of 2nd degree connections in user-item graph)
        :param results_format:
            'flat' for dataframe of triplets (source_item, similar_item, similarity)
            'lists' for dataframe of lists (source_item, list of similar items, list of similarity scores)
        :param pbar: name of tqdm progress bar (None means no tqdm)

        :return: a matrix of most similar IDs [n_ids, N], a matrix of score of those similarities [n_ids, N]
        """

        item_ids, target_item_ids = self._check_item_ids_args(item_ids, target_item_ids)

        biases, representations = self._get_item_factors(mode=embeddings_mode)

        best_ids, best_scores = most_similar(
            source_ids=item_ids,
            target_ids=target_item_ids,
            source_encoder=self.sparse_mat_builder.iid_encoder,
            source_mat=representations,
            source_biases=biases,
            n=n_simil+1 if remove_self else n_simil,
            simil_mode=simil_mode,
            pbar=pbar
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

    def get_similar_users(self, user_ids=None, target_user_ids=None, n_simil=10, remove_self=True,
                          simil_mode='cosine', pbar=None):
        """
        same as get_similar_items but for users
        """
        user_ids, target_user_ids = self._check_user_ids_args(user_ids, target_user_ids)

        user_biases, user_representations = self._get_user_factors()

        best_ids, best_scores = most_similar(
            source_ids=user_ids,
            target_ids=target_user_ids,
            source_encoder=self.sparse_mat_builder.uid_encoder,
            source_mat=user_representations,
            source_biases=user_biases,
            n=n_simil,
            simil_mode=simil_mode,
            pbar=pbar
        )

        simil_df = self._format_results_df(
            user_ids, target_ids_mat=best_ids,
            scores_mat=best_scores, results_format='similarities_flat'). \
            rename({self._item_col_simil: self._user_col}, axis=1)
        # this is UGLY, if this function is ever useful, fix this please (the renaming shortcut)

        if remove_self:
            simil_df = self._remove_self_similarities(
                simil_df, col1=self._user_col, col2=self._item_col)

        simil_df = self._recos_flat_to_lists(simil_df, n_cutoff=n_simil)

        return simil_df

    def _get_recommendations_flat(
            self, user_ids, n_rec, item_ids=None, exclude_training=True,
            pbar=None, item_features_mode=None, use_biases=True):

        user_biases, user_representations = self._get_user_factors()
        item_biases, item_representations = self._get_item_factors(mode=item_features_mode)

        if not use_biases:
            user_biases, item_biases = None, None

        best_ids, best_scores = most_similar(
            source_ids=user_ids,
            target_ids=item_ids,
            source_encoder=self.sparse_mat_builder.uid_encoder,
            target_encoder=self.sparse_mat_builder.iid_encoder,
            source_mat=user_representations,
            target_mat=item_representations,
            source_biases=user_biases,
            target_biases=item_biases,
            exclude_mat_sp=self.train_mat if exclude_training else None,
            n=n_rec,
            simil_mode='dot',
            pbar=pbar
        )

        return self._format_results_df(
            source_vec=user_ids, target_ids_mat=best_ids, scores_mat=best_scores,
            results_format='recommendations_flat')