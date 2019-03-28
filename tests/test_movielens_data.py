import unittest
import os
import pandas as pd

movielens_dir = os.path.join(os.path.dirname(__file__), '../examples/out')

class TestMovieLens(unittest.TestCase):
    def test_data(self):
        from ml_recsys_tools.datasets.prep_movielense_data import get_and_prep_data
        rating_csv_path, users_csv_path, movies_csv_path = get_and_prep_data(movielens_dir)

        ratings_df = pd.read_csv(rating_csv_path)
        self.assertListEqual(list(ratings_df.columns), ['rating', 'timestamp', 'itemid', 'userid'])
        self.assertEqual(len(ratings_df), 1000209)

        users_df = pd.read_csv(users_csv_path)
        self.assertListEqual(list(users_df.columns), ['gender', 'age', 'occupation',
                                        'zipcode', 'index', 'occupation_name', 'userid'])
        self.assertEqual(len(users_df), 6040)

        movies_df = pd.read_csv(movies_csv_path)
        self.assertSetEqual(set(movies_df.columns),
                            {'itemid', 'Adventure', 'FilmNoir', 'Comedy', 'SciFi',
                               'Fantasy', 'Crime', 'Mystery', 'Action', 'Thriller', 'Horror',
                               'Musical', 'Drama', 'Western', 'War', 'Animation', 'Romance',
                               'Childrens', 'Documentary'})
        self.assertEqual(len(movies_df), 3883)

        from ml_recsys_tools.data_handlers.interaction_handlers_base import ObservationsDF
        obs = ObservationsDF(df_obs=ratings_df)
        info = obs.data_info()
        self.assertEqual(info['len'], 989539)
        self.assertEqual(info['n_unique_items'], 3706)
        self.assertEqual(info['n_unique_users'], 5796)
        self.assertEqual(info['ratings_20_pctl'], 3.0)
        self.assertEqual(info['ratings_80_pctl'], 5.0)