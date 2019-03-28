import numpy as np
from sklearn.preprocessing import normalize

from ml_recsys_tools.recommenders.similarity_recommenders import BaseSimilarityRecommeder
from ml_recsys_tools.utils.instrumentation import log_time_and_shape
from ml_recsys_tools.utils.parallelism import map_batches_multiproc
from ml_recsys_tools.utils.similarity import top_N_sorted


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


class ItemCoocRecommender(BaseSimilarityRecommeder):

    default_fit_params = dict(
        normalize_items=True,
        degree=1,
        prune_ratio=0.0,
        decay=0.5,
        min_cooccurrence=1,
        base_min_cooccurrence=1,
        trans_func='ones',
    )

    def fit(self, train_obs, **fit_params):
        self._prep_for_fit(train_obs, **fit_params)
        self.similarity_mat = interactions_mat_to_cooccurrence_mat(
            self.train_mat, **self.fit_params)
        self.similarity_mat += self.similarity_mat.T


class UserCoocRecommender(ItemCoocRecommender):

    def fit(self, train_obs, **fit_params):
        self._prep_for_fit(train_obs, **fit_params)
        self.similarity_mat = interactions_mat_to_cooccurrence_mat(
            self.train_mat.T, **self.fit_params)

    def _predict_on_inds_dense(self, user_inds, item_inds):
        user_pred_mat = self.similarity_mat[user_inds, :] * self.train_mat
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
            user_pred_mat = self.similarity_mat[u_inds, :] * self.train_mat
            if exclusions:
                user_pred_mat -= self.exclude_mat[u_inds, :] * np.inf
            if item_inds is not None:
                user_pred_mat = user_pred_mat[:, item_inds]
            return top_N_sorted(user_pred_mat.toarray(), n_rec)

        ret = map_batches_multiproc(best_for_batch, user_inds, chunksize=1000)

        best_inds = np.concatenate([r[0] for r in ret], axis=0)
        best_scores = np.concatenate([r[1] for r in ret], axis=0)

        # back to ids
        if item_inds is not None:
            best_inds = item_inds[best_inds.astype(int)]

        best_ids = self.sparse_mat_builder.iid_encoder. \
            inverse_transform(best_inds.astype(int))

        return self._format_results_df(
            source_vec=user_ids, target_ids_mat=best_ids, scores_mat=best_scores,
            results_format='recommendations_flat')

    def get_similar_items(self, item_ids=None, target_item_ids=None, n_simil=10,
                          results_format='lists', **kwargs):
        raise NotImplementedError