import pandas as pd
import numpy as np
from copy import deepcopy

from ml_recsys_tools.data_handlers.interaction_handlers_base import ObservationsDF

from examples.prep_movielense_data import get_and_prep_data

from ml_recsys_tools.utils.testing import TestCaseWithState
from test_movielens_data import movielens_dir

rating_csv_path, users_csv_path, movies_csv_path = get_and_prep_data(movielens_dir)


class TestRecommendersBasic(TestCaseWithState):

    @classmethod
    def setUpClass(cls):
        cls.k = 10
        cls.n = 10
        cls.metric = 'n-MRR@%d' % cls.k

    def _setup_obs_handler(self):
        ratings_df = pd.read_csv(rating_csv_path)
        obs = ObservationsDF(ratings_df, uid_col='userid', iid_col='itemid')
        obs = obs.sample_observations(n_users=1000, n_items=1000)
        self.state.train_obs, self.state.test_obs = obs.split_train_test(ratio=0.2, users_ratio=1.0)

    def test_b_1_lfm_recommender(self):
        self._setup_obs_handler()

        from ml_recsys_tools.recommenders.lightfm_recommender import LightFMRecommender
        lfm_rec = LightFMRecommender()
        lfm_rec.fit(self.state.train_obs, epochs=10)
        self.assertEqual(lfm_rec.fit_params['epochs'], 10)
        self._check_recommendations_and_similarities(lfm_rec)
        self.state.lfm_rec = lfm_rec

    def _check_recommendations(self, recs, users, n):
        # check format
        self.assertEqual(len(recs), len(users))
        self.assertListEqual(list(recs.columns), ['userid', 'itemid', 'prediction'])
        self.assertTrue(all(recs['itemid'].apply(len).values == n))

        # check predictions sorted
        for i in np.random.choice(np.arange(len(recs)), min(100, len(recs))):
            self.assertListEqual(recs['prediction'][i], sorted(recs['prediction'][i], reverse=True))

    def _check_similarities(self, simils, items, n):
        # check format
        self.assertEqual(len(simils), len(items))
        self.assertListEqual(list(simils.columns), ['itemid_source', 'itemid', 'prediction'])
        self.assertTrue(all(simils['itemid'].apply(len).values == n))

        # sample check no self-similarities returned
        for i in np.random.choice(np.arange(len(simils)), min(100, len(simils))):
            self.assertTrue(simils['itemid_source'][i] not in simils['itemid'][i])

        # sample check predictions are sorted
        for i in np.random.choice(np.arange(len(simils)), min(100, len(simils))):
            self.assertListEqual(list(simils['prediction'][i]), sorted(simils['prediction'][i], reverse=True))

    def _check_recommendations_and_similarities(self, rec):
        self._check_recommendations(rec.get_recommendations(n_rec=self.n), rec.all_users, n=self.n)
        self._check_similarities(rec.get_similar_items(n_simil=self.n), rec.all_items, n=self.n)

    def test_b_2_lfm_rec_evaluation(self):
        k = self.k

        rep_exact = self.state.lfm_rec.eval_on_test_by_ranking_exact(
            self.state.test_obs.df_obs, prefix='lfm regular exact ', k=k)
        print(rep_exact)

        rep_reg = self.state.lfm_rec.eval_on_test_by_ranking(
            self.state.test_obs.df_obs, prefix='lfm regular ', n_rec=200, k=k)
        print(rep_reg)

        self.assertListEqual(list(rep_reg.columns), list(rep_exact.columns))

        # test that those fields are almost equal for the two test methods
        tolerance = 0.05
        for col in rep_reg.columns:
            self.assertTrue(all(abs(1 - (rep_exact[col].values / rep_reg[col].values)) < tolerance))

    def test_b_3_lfm_early_stop(self):
        lfm_rec = deepcopy(self.state.lfm_rec)
        lfm_rec.fit(self.state.train_obs, epochs=1)
        prev_epochs = lfm_rec.fit_params['epochs']

        lfm_rec.fit_with_early_stop(
            self.state.train_obs,
            epochs_max=5, epochs_step=5, stop_patience=1,
            valid_ratio=0.2, metric=self.metric, k=self.k,
            refit_on_all=False, plot_convergence=False)

        sut_epochs = lfm_rec.fit_params['epochs']

        # check that epochs parameter changed
        self.assertNotEqual(prev_epochs, sut_epochs)

        # check that in the report dataframe the maximum metric value is for our new epoch number
        self.assertEqual(lfm_rec.early_stop_metrics_df[self.metric].idxmax(), sut_epochs)

    def test_b_4_lfm_hp_search(self):
        lfm_rec = deepcopy(self.state.lfm_rec)
        space = lfm_rec.guess_search_space()
        n_iters = 4
        hp_space = dict(
            no_components=space.Integer(10, 40),
            epochs=space.Integer(5, 20),
            item_alpha=space.Real(1e-8, 1e-5, prior='log-uniform')
        )
        hp_results = lfm_rec.hyper_param_search(
            self.state.train_obs,
            metric=self.metric,
            k=self.k,
            plot_graph=False,
            hp_space=hp_space,
            n_iters=n_iters,
        )

        # check that best model works
        self._check_recommendations_and_similarities(hp_results.best_model)

        # check that hp space and params have same keys
        best_params = hp_results.best_params
        self.assertListEqual(sorted(best_params.keys()), sorted(hp_space.keys()))

        # check report format
        rep = hp_results.report
        self.assertEqual(len(rep), n_iters)

        # check that best values in report are best values for best params
        best_params_sut = rep.loc[rep['target_loss'].idxmin()][best_params.keys()].to_dict()
        self.assertDictEqual(best_params_sut, best_params)

    def test_c_cooc_recommender(self):
        from ml_recsys_tools.recommenders.similarity_recommenders import ItemCoocRecommender

        item_cooc_rec = ItemCoocRecommender()
        item_cooc_rec.fit(self.state.train_obs)
        item_cooc_rep = item_cooc_rec.eval_on_test_by_ranking(self.state.test_obs, prefix='item cooccurrence ')
        print(item_cooc_rep)
        self._check_recommendations_and_similarities(item_cooc_rec)
        self.state.item_cooc_rec = item_cooc_rec

    def test_c_als_recommender(self):
        from ml_recsys_tools.recommenders.implib_recommenders import ALSRecommender

        als_rec = ALSRecommender()
        als_rec.fit(self.state.train_obs)
        als_rep = als_rec.eval_on_test_by_ranking(self.state.test_obs, prefix='als ')
        print(als_rep)
        self._check_recommendations_and_similarities(als_rec)

    def test_c_spotlight_implicit_recommender(self):
        from ml_recsys_tools.recommenders.spotlight_recommenders import EmbeddingFactorsRecommender

        rec = EmbeddingFactorsRecommender()
        rec.fit(self.state.train_obs)
        report = rec.eval_on_test_by_ranking(self.state.test_obs, prefix='spot ')
        print(report)
        self._check_recommendations_and_similarities(rec)

    def test_d_comb_rank_ens(self):
        from ml_recsys_tools.recommenders.combination_ensembles import CombinedRankEnsemble

        comb_ranks_rec = CombinedRankEnsemble(
            recommenders=[self.state.lfm_rec, self.state.item_cooc_rec])
        comb_rank_rep = comb_ranks_rec.eval_on_test_by_ranking(self.state.test_obs, prefix='combined ranks ')
        print(comb_rank_rep)
        self._check_recommendations_and_similarities(comb_ranks_rec)

    def test_d_comb_simil_ens(self):
        from ml_recsys_tools.recommenders.combination_ensembles import CombinedSimilRecoEns

        comb_simil_rec = CombinedSimilRecoEns(
            recommenders=[self.state.lfm_rec, self.state.item_cooc_rec])
        comb_simil_rec.fit(self.state.train_obs)
        comb_simil_rep = comb_simil_rec.eval_on_test_by_ranking(self.state.test_obs, prefix='combined simils ')
        print(comb_simil_rep)
        self._check_recommendations_and_similarities(comb_simil_rec)


