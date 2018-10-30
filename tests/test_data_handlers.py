import pandas as pd
from ml_recsys_tools.datasets.prep_movielense_data import get_and_prep_data
from ml_recsys_tools.utils.testing import TestCaseWithState
from tests.test_movielens_data import movielens_dir

rating_csv_path, users_csv_path, movies_csv_path = get_and_prep_data(movielens_dir)


class TestRecommendersBasic(TestCaseWithState):

    def _obs_split_data_check(self, obs_full, obs1, obs2):
        # all the data is still there
        self.assertEqual(len(obs1.df_obs) + len(obs2.df_obs), len(obs_full.df_obs))
        # no intersections
        intersections = pd.merge(obs1.df_obs, obs2.df_obs, on=['userid', 'itemid'], how='inner')
        self.assertEqual(len(intersections), 0)

    def test_splits(self):
        from ml_recsys_tools.data_handlers.interaction_handlers_base import ObservationsDF

        ratings_df = pd.read_csv(rating_csv_path)
        obs = ObservationsDF(ratings_df, uid_col='userid', iid_col='itemid', timestamp_col='timestamp')
        obs = obs.sample_observations(n_users=1000, n_items=1000)

        ratio = 0.2

        # regular split
        train_obs, test_obs = obs.split_train_test(ratio=ratio)
        self._obs_split_data_check(obs, train_obs, test_obs)
        self.state.train_obs, self.state.test_obs = train_obs, test_obs

        # split for only some users
        user_ratio = 0.2
        train_obs, test_obs = obs.split_train_test(ratio=ratio, users_ratio=user_ratio)
        self._obs_split_data_check(obs, train_obs, test_obs)
        post_split_ratio = test_obs.df_obs['userid'].nunique() / train_obs.df_obs['userid'].nunique()
        self.assertAlmostEqual(user_ratio, post_split_ratio, places=1)

        # split by timestamp
        time_col = obs.timestamp_col
        train_obs, test_obs = obs.split_train_test(ratio=ratio, time_split_column=time_col)
        self._obs_split_data_check(obs, train_obs, test_obs)
        self.assertGreaterEqual(test_obs.df_obs[time_col].min(), train_obs.df_obs[time_col].max())