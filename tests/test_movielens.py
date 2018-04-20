import unittest
import os
import pandas as pd

data_dir = os.path.join(os.path.dirname(__file__), '../examples/out')

class TestMovieLens(unittest.TestCase):
    def test_data(self):
        from examples.prep_movielense_data import get_and_prep_data
        rating_csv_path, users_csv_path, movies_csv_path = get_and_prep_data(data_dir)

        ratings_df = pd.read_csv(rating_csv_path)
        self.assertListEqual(ratings_df.columns, ['rating', 'timestamp', 'itemid', 'userid'])
        self.assertEqual(len(ratings_df), 1000209)

        users_df = pd.read_csv(users_csv_path)
        self.assertListEqual(users_df.columns, ['user_ind', 'gender', 'age', 'occupation',
                                        'zipcode', 'index', 'occupation_name', 'userid'])
        self.assertEqual(len(users_df), 6040)

        movies_df = pd.read_csv(movies_csv_path)
        self.assertListEqual(movies_df.columns, ['item_ind', 'itemid', 'genres'])
        self.assertEqual(len(movies_df), 3883)
