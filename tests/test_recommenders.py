import pandas as pd

from examples.prep_movielense_data import get_and_prep_data
from ml_recsys_tools.utils.testiing import TestCaseWithState

rating_csv_path, users_csv_path, movies_csv_path = get_and_prep_data()


class TestRecommendersBasic(TestCaseWithState):

    def test_1_obs_handler(self):
        from ml_recsys_tools.data_handlers.interaction_handlers_base import ObservationsDF

        ratings_df = pd.read_csv(rating_csv_path)
        obs = ObservationsDF(ratings_df, uid_col='userid', iid_col='itemid')
        obs = obs.sample_observations(n_users=500)
        # TODO: assertions
        self.state.train_obs, self.state.test_obs = obs.split_train_test(ratio=0.2, users_ratio=1.0)


    def test_2_lightfm_recommender(self):
        from ml_recsys_tools.recommenders.lightfm_recommender import LightFMRecommender

        lfm_rec = LightFMRecommender()
        lfm_rec.fit(self.state.train_obs, epochs=10)

        # TODO: assertions
        rep_exact = lfm_rec.eval_on_test_by_ranking_exact(self.state.test_obs.df_obs, prefix='lfm regular exact ')

        rep_reg = lfm_rec.eval_on_test_by_ranking(self.state.test_obs.df_obs, prefix='lfm regular ', n_rec=100)

        recs = lfm_rec.get_recommendations(lfm_rec.all_users, n_rec=5)

        simils = lfm_rec.get_similar_items(lfm_rec.all_items, n_simil=5)

        lfm_rec.fit_with_early_stop(self.state.train_obs, epochs_max=20, epochs_step=2, stop_patience=1,
                                    valid_ratio=0.2, metric='n-MRR@10', refit_on_all=True)


        space = lfm_rec.guess_search_space()
        hp_results = lfm_rec.hyper_param_search(
            self.state.train_obs,
            metric='n-MRR@10',
            hp_space=dict(
                no_components=space.Integer(10, 40),
                epochs=space.Integer(5, 20),
                item_alpha=space.Real(1e-8, 1e-5, prior='log-uniform')
            ),
            n_iters=2,
            )

        hp_results.best_model.fit(self.state.train_obs)
        rep_hp = hp_results.best_model.eval_on_test_by_ranking(self.state.test_obs, prefix='lfm hp search ')

        self.state.lfm_rec = lfm_rec


    def test_cooc_recommender(self):
        from ml_recsys_tools.recommenders.similarity_recommenders import ItemCoocRecommender
        self.item_cooc_rec = ItemCoocRecommender()
        self.item_cooc_rec.fit(self.train_obs)
        print(self.item_cooc_rec.eval_on_test_by_ranking(self.test_obs, prefix='item cooccurrence '))


# combine LightFM and Cooccurrence recommenders
# using recommendation ranks, and evaluate
from ml_recsys_tools.recommenders.combination_ensembles import CombinedRankEnsemble
comb_ranks_rec = CombinedRankEnsemble(
    recommenders=[lfm_rec, item_cooc_rec])
print(comb_ranks_rec.eval_on_test_by_ranking(test_obs, prefix='combined ranks '))


# combine LightFM and Cooccurrence recommenders
# using similarities, and evaluate
from ml_recsys_tools.recommenders.combination_ensembles import CombinedSimilRecoEns
comb_simil_rec = CombinedSimilRecoEns(
    recommenders=[lfm_rec, item_cooc_rec])
comb_simil_rec.fit(train_obs)
print(comb_simil_rec.eval_on_test_by_ranking(test_obs, prefix='combined simils '))


# combine LightFM, Cooccurrence and Combined Simil recommender
# using recommendation ranks, and evaluate
from ml_recsys_tools.recommenders.combination_ensembles import CombinedRankEnsemble
comb_ranks_simil_rec = CombinedRankEnsemble(
    recommenders=[lfm_rec, item_cooc_rec, comb_simil_rec])
print(comb_ranks_simil_rec.eval_on_test_by_ranking(
    test_obs, prefix='combined ranks and simils '))


# get all recommendations and print a sample
recs_ens = comb_ranks_simil_rec.get_recommendations(
    comb_ranks_simil_rec.all_users, n_rec=5)
print(recs_ens.sample(5))

# get all similarities and print a sample
simils_ens = comb_ranks_simil_rec.get_similar_items(
    comb_ranks_simil_rec.all_items, n_simil=5)
print(simils_ens.sample(10))