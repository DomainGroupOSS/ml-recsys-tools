"""
This is a basic example on datasets-1M demonstrating:
    - basic data ingestion without any item/user features
    - LightFM recommender
    - basic evaluation
    - recommendations output
    - similar items output
"""

# dataset: download and prepare dataframes
from ml_recsys_tools.datasets.prep_movielense_data import get_and_prep_data

rating_csv_path, users_csv_path, movies_csv_path = get_and_prep_data()

# read the interactions dataframe and create a data handler object and  split to train and test
import pandas as pd

ratings_df = pd.read_csv(rating_csv_path)
from ml_recsys_tools.data_handlers.interaction_handlers_base import ObservationsDF

obs = ObservationsDF(ratings_df, uid_col='userid', iid_col='itemid')
train_obs, test_obs = obs.split_train_test(ratio=0.2)

# train and test LightFM recommender
from ml_recsys_tools.recommenders.lightfm_recommender import LightFMRecommender

lfm_rec = LightFMRecommender()
lfm_rec.fit(train_obs, epochs=10)

# print summary evaluation report:
print(lfm_rec.eval_on_test_by_ranking(test_obs.df_obs, prefix='lfm ', n_rec=100))

# get all recommendations and print a sample (training interactions are filtered out by default)
recs = lfm_rec.get_recommendations(lfm_rec.all_users, n_rec=5)
print(recs.sample(5))

# get all similarities and print a sample
simils = lfm_rec.get_similar_items(lfm_rec.all_items, n_simil=5)
print(simils.sample(10))
