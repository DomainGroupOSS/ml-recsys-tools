from functools import partial
from itertools import repeat

from ml_recsys_tools.recommenders.similarity_recommenders import SimilarityDFRecommender
from ml_recsys_tools.utils.parallelism import batch_generator
from ml_recsys_tools.recommenders.ensembles_base import CombinationEnsembleBase, calc_dfs_and_combine_scores


class CombinedRankEnsemble(CombinationEnsembleBase):
    pass

class CombinedSimilRecoEns(SimilarityDFRecommender):

    def __init__(self,
                 recommenders,
                 similarity_func_params=None,
                 n_unfilt=100,
                 numeric_n_bins=32,
                 combination_mode='hmean',
                 **kwargs):
        self.recommenders = recommenders
        self.similarity_func_params = similarity_func_params
        self.n_unfilt = n_unfilt
        self.combination_mode = combination_mode
        self.numeric_n_bins = numeric_n_bins
        super().__init__(**kwargs)

    def set_params(self, **params):
        params = self._pop_set_params(
            params, ['n_unfilt', 'numeric_n_bins',
                     'combination_mode'])
        super().set_params(**params)

    def _get_similarity_func_params(self):
        if self.similarity_func_params is None:
            return repeat({}, len(self.recommenders))
        elif isinstance(self.similarity_func_params, dict):
            return repeat(self.similarity_func_params, len(self.recommenders))
        elif hasattr(self.similarity_func_params, '__len__'):
            return self.similarity_func_params
        else:
            raise ValueError('Unsupported format for similarity functions parameters: %s'
                             % str(self.similarity_func_params))

    def fit(self, train_obs, batch_size=10000,
            similarity_queue=None, similarity_queue_cutoff=10, **fit_params):

        itemids = self.recommenders[0].all_items

        for i, items in enumerate(batch_generator(itemids, batch_size)):

            calc_funcs = [
                partial(rec.get_similar_items,
                        item_ids=items, n_simil=self.n_unfilt, results_format='flat', **params)
                for rec, params in zip(self.recommenders, self._get_similarity_func_params())]

            simil_df = calc_dfs_and_combine_scores(
                calc_funcs=calc_funcs,
                combine_func=self.combination_mode,
                fill_val=self.n_unfilt + 1,
                groupby_col=self._item_col_simil,
                item_col=self._item_col,
                scores_col=self._prediction_col,
            )

            if similarity_queue:
                similarity_queue.put(
                    self._simil_flat_to_lists(
                        simil_df, n_cutoff=similarity_queue_cutoff))

            if i == 0:
                super().fit(train_obs, simil_df, **fit_params)
            else:
                super().continue_fit(simil_df)

        if similarity_queue:
            similarity_queue.put('END')

        return self


class CascadeEnsemble(CombinationEnsembleBase):

    def __init__(self, recommenders, **kwargs):
        super().__init__(recommenders, **kwargs)
        assert len(recommenders) == 2, \
            'only 2 recommenders supported'
        assert hasattr(self.recommenders[1], 'predict_on_df'), \
            'no "predict_on_df" for second recommender'

    def _get_recommendations_flat(self, user_ids, n_rec, item_ids=None,
                                  exclude_training=True, **kwargs):
        recos_df = self.recommenders[0].get_recommendations(
            user_ids=user_ids, item_ids=item_ids, n_rec=n_rec,
            exclude_training=exclude_training,
            results_format='flat', **kwargs)
        pred_mat_builder = self.get_prediction_mat_builder_adapter(self.sparse_mat_builder)
        return self.recommenders[1].predict_on_df(
            recos_df,
            user_col=pred_mat_builder.uid_source_col,
            item_col=pred_mat_builder.iid_source_col)

    def get_similar_items(self, item_ids=None, target_item_ids=None, n_simil=10,
                          n_unfilt=100, results_format='lists', **kwargs):
        raise NotImplementedError()

