"""
using multiple test sets
"""

# dataset: download and prepare dataframes
import pandas as pd
from ml_recsys_tools.datasets.prep_movielense_data import get_and_prep_data
from ml_recsys_tools.recommenders.lightfm_recommender import LightFMRecommender

rating_csv_path, users_csv_path, movies_csv_path = get_and_prep_data()

# read the interactions dataframe and create a data handler object and  split to train and test
ratings_df = pd.read_csv(rating_csv_path)
from ml_recsys_tools.data_handlers.interaction_handlers_base import ObservationsDF

obs = ObservationsDF(ratings_df)
train_obs, test_obs = obs.split_train_test(ratio=0.2, users_ratio=0.2)

def construct_multiple_test_sets(test_df, train_df):
    # by user history - active and inactive users
    user_hist_counts = train_df.userid.value_counts()
    user_hist_counts.hist(bins=100, alpha=0.5)
    active_users = user_hist_counts[user_hist_counts >= 300].index.tolist()
    test_df_act_us = test_df[test_df.userid.isin(active_users)]
    test_df_nonact_us = test_df[~test_df.userid.isin(active_users)]

    # by item popularity- popular and unpopular items
    item_hist_counts = train_df.itemid.value_counts()
    item_hist_counts.hist(bins=100, alpha=0.5)
    popular_items = item_hist_counts[item_hist_counts >= 1000].index.tolist()
    test_df_pop_movies = test_df[test_df.itemid.isin(popular_items)]
    test_df_nonpop_movies = test_df[~test_df.itemid.isin(popular_items)]

    test_dfs = [test_df, test_df_act_us, test_df_nonact_us, test_df_pop_movies, test_df_nonpop_movies]
    test_names = ['all ', 'active users ', 'inactive users ', 'popular movies ', 'unpopular movies ']
    df_lens = [len(t) for t in test_dfs]
    print('Test DFs counts: ' + str(list(zip(test_names, df_lens))))
    return test_dfs, test_names

test_dfs, test_names = construct_multiple_test_sets(test_df=test_obs.df_obs, train_df=train_obs.df_obs)

# evaluation
lfm_rec = LightFMRecommender()
lfm_rec.fit(train_obs, epochs=10)
print(lfm_rec.eval_on_test_by_ranking(
    test_dfs=test_dfs, test_names=test_names, prefix='lfm regular '))
