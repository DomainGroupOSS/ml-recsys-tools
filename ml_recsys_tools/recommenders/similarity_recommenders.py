from copy import deepcopy
from functools import partial

import numpy as np
from sklearn.preprocessing import normalize

from ml_recsys_tools.recommenders.recommender_base import BaseDFSparseRecommender
from ml_recsys_tools.utils.similarity import top_N_sorted_on_sparse, custom_row_func_on_sparse
from ml_recsys_tools.utils.instrumentation import log_time_and_shape


@log_time_and_shape
def interactions_mat_to_cooccurrence_mat(
        obs_mat, normalize_items=True, degree=1, base_min_cooccurrence=1,
        prune_ratio=0.5, decay=0.3, min_cooccurrence=3, trans_func='ones'):
    def prune_mat(m, ratio=0.0, cutoff=0):
        if (ratio == 0.0 and cutoff == 0) or (not len(m.data)):
            return m
        else:
            if ratio > 0.0:
                if len(m.data) > 50000:
                    data_sample = np.random.choice(
                        m.data, min(len(m.data), 10000), replace=False)
                else:
                    data_sample = m.data
                cutoff = max(cutoff, np.percentile(data_sample, int(100 * ratio)))

            m.data[m.data < cutoff] *= 0
            m.eliminate_zeros()
        return m

    def first_degree_cooccurrence(mat, min_cooc=1):

        if trans_func == 'ones':
            # binarize interaction
            mat.data = np.ones(mat.data.shape)

        elif trans_func == 'log':
            mat.data = np.log10(mat.data + 1)

        elif trans_func == 'none':
            pass

        else:
            raise ValueError('Unknown trans_func: %s' % trans_func)

        # 1st degree interaction matrix
        cooc_mat = mat.T * mat
        # remove self similarities
        cooc_mat.setdiag(0)
        # maybe less memory
        cooc_mat = cooc_mat.astype(np.float32)

        if min_cooc > 1:
            # threshold interactions
            cooc_mat = prune_mat(cooc_mat, 0.0, min_cooc)

        return cooc_mat

    cooc_mat_base = first_degree_cooccurrence(obs_mat, base_min_cooccurrence)

    if degree > 1:

        # keep weight constant
        total_weight = np.sum(cooc_mat_base.data)

        higher_deg_cooc = prune_mat(cooc_mat_base.copy(), prune_ratio, min_cooccurrence)

        for i in range(degree - 1):
            higher_deg_cooc += \
                decay ** (i + 1) * \
                higher_deg_cooc * \
                higher_deg_cooc

            # remove self similarities
            higher_deg_cooc.setdiag(0)

        higher_deg_cooc.data *= total_weight / (np.sum(higher_deg_cooc.data) + 1)  # avoid divide by 0

        cooc_mat_base += higher_deg_cooc

    # mormalization
    if normalize_items:
        cooc_mat_base = normalize(cooc_mat_base, norm='l1', axis=1)

    return cooc_mat_base


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

    @BaseDFSparseRecommender.do_not_decorate
    def _recommend_for_item_inds(self, source_item_inds, *ignored_args, target_item_inds=None,
                                 n_rec=100, exclude_training=True):

        # if target_item_inds is None:
        sub_mat = self.similarity_mat[source_item_inds, :]
        # else:
        #     sub_mat = self.similarity_mat[item_inds, :][:, target_item_inds]

        sum_simils = np.array(np.sum(sub_mat, axis=0)).ravel()

        if exclude_training:
            # if target_item_inds is not None:
            #     sum_simils[np.isin(target_item_inds, item_inds)] *= 0
            # else:
            sum_simils[source_item_inds] *= 0

        if target_item_inds is not None:
            sum_simils = sum_simils[target_item_inds]

        n_rec_min = min(n_rec, len(sum_simils))

        i_part = np.argpartition(sum_simils, -n_rec_min)[-n_rec_min:]
        i_sort = i_part[np.argsort(-sum_simils[i_part])[:n_rec_min]]

        scores = sum_simils[i_sort]
        inds = target_item_inds[i_sort] if target_item_inds is not None else i_sort

        return inds, scores

    # def recommend_for_interaction_history(self, interactions_ids, n_rec):
    #     interactions_inds = self.sparse_mat_builder.iid_encoder.transform(
    #         np.array(interactions_ids, dtype=str))
    #     rec_ids, rec_scores = self._recommend_for_item_inds(interactions_inds, n_rec_unfilt=n_rec)
    #     return self.sparse_mat_builder.iid_encoder.inverse_transform(rec_ids), rec_scores

    def _get_recommendations_flat(
            self, user_ids, item_ids, exclude_training=True,
            n_rec=100, pbar=None, **kwargs):

        self._check_no_negatives()

        item_inds = self.sparse_mat_builder.iid_encoder.transform(item_ids)
        item_inds.sort()

        top_simil_for_users = partial(
            self._recommend_for_item_inds,
            target_item_inds=item_inds,
            n_rec=n_rec,
            exclude_training=exclude_training)

        best_ids, best_scores = custom_row_func_on_sparse(
            row_func=top_simil_for_users,
            source_ids=user_ids,
            source_encoder=self.sparse_mat_builder.uid_encoder,
            target_encoder=self.sparse_mat_builder.iid_encoder,
            sparse_mat=self.train_mat,
            pbar=pbar,
            chunksize=500,
        )

        return self._format_results_df(
            source_vec=user_ids, target_ids_mat=best_ids, scores_mat=best_scores,
            results_format='recommendations_flat')

    def get_similar_items(self, item_ids=None, target_item_ids=None,
                          n_simil=10, results_format='lists', pbar=None, **kwargs):

        item_ids, target_item_ids = \
            self._check_item_ids_args(item_ids, target_item_ids)

        self._check_no_negatives()

        best_ids, best_scores = top_N_sorted_on_sparse(
            source_ids=item_ids,
            target_ids=target_item_ids,
            encoder=self.sparse_mat_builder.iid_encoder,
            sparse_mat=self.similarity_mat,
            n_top=n_simil,
            pbar=pbar
        )

        simil_df = self._format_results_df(
            item_ids, target_ids_mat=best_ids,
            scores_mat=best_scores, results_format='similarities_' + results_format)
        return simil_df


class ItemCoocRecommender(BaseSimilarityRecommeder):

    def __init__(self, degree=1, normalize_items=True,
                 prune_ratio=0.0, decay=0.5, min_cooccurrence=1,
                 base_min_cooccurrence=1, trans_func='ones', **kwargs):
        super().__init__(
            fit_params=dict(
                normalize_items=normalize_items,
                degree=degree,
                prune_ratio=prune_ratio,
                decay=decay,
                min_cooccurrence=min_cooccurrence,
                base_min_cooccurrence=base_min_cooccurrence,
                trans_func=trans_func,
            ),
            **kwargs)

    def fit(self, train_obs, **fit_params):
        self._prep_for_fit(train_obs, **fit_params)
        self.similarity_mat = interactions_mat_to_cooccurrence_mat(
            self.train_mat, **self.fit_params)
        self.similarity_mat += self.similarity_mat.T

    def set_params(self, **params):
        """
        this is for skopt / sklearn compatibility
        """
        params = self._pop_set_dict(
            self.fit_params,
            params,
            ['degree', 'normalize_items', 'prune_ratio',
             'decay', 'min_cooccurrence', 'base_min_cooccurrence',
             'trans_func'])

        super().set_params(**params)


class UserCoocRecommender(ItemCoocRecommender):

    def fit(self, train_obs, **fit_params):
        self._prep_for_fit(train_obs, **fit_params)
        self.similarity_mat = interactions_mat_to_cooccurrence_mat(
            self.train_mat.T, **self.fit_params)

    def recommend_for_interaction_history(self, interactions_ids, n_rec):
        raise NotImplementedError

    def _get_recommendations_flat(
            self, user_ids, item_ids, n_rec=100,
            exclude_training=True, pbar=None, **kwargs):

        item_inds = self.sparse_mat_builder.iid_encoder.transform(item_ids)
        item_inds.sort()

        def row_func(user_inds, row_data, exclude_inds):
            sub_mat = self.train_mat[user_inds, :]
            sub_mat.sort_indices()

            if exclude_inds is not None:
                sub_mat[:, exclude_inds].data *= 0

            for i, r in enumerate(row_data):
                sub_mat.data[sub_mat.indptr[i]:sub_mat.indptr[i + 1]] *= r

            sum_weight_occurs = np.array(np.sum(sub_mat.tocsr(), axis=0)).ravel()
            sum_weight_occurs = sum_weight_occurs[item_inds]

            n_rec_min = min(n_rec, len(sum_weight_occurs))

            i_part = np.argpartition(sum_weight_occurs, -n_rec_min)[-n_rec_min:]
            i_sort = i_part[np.argsort(-sum_weight_occurs[i_part])[:n_rec_min]]

            return item_inds[i_sort], sum_weight_occurs[i_sort]

        best_ids, best_scores = custom_row_func_on_sparse(
            row_func=row_func,
            source_ids=user_ids,
            source_encoder=self.sparse_mat_builder.uid_encoder,
            target_encoder=self.sparse_mat_builder.iid_encoder,
            sparse_mat=self.similarity_mat,
            exclude_mat_sp=self.train_mat if exclude_training else None,
            pbar=pbar
        )

        return self._format_results_df(
            source_vec=user_ids, target_ids_mat=best_ids, scores_mat=best_scores,
            results_format='recommendations_flat')

    def get_similar_items(self, item_ids=None, target_item_ids=None, n_simil=10,
                          results_format='lists', pbar=None, **kwargs):
        raise NotImplementedError


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
