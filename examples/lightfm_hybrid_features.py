from ml_recsys_tools.datasets.prep_movielense_data import get_and_prep_data
rating_csv_path, users_csv_path, movies_csv_path = get_and_prep_data()

# read the interactions dataframe and create a data handler object and  split to train and test
import pandas as pd
ratings_df = pd.read_csv(rating_csv_path)
movies_df = pd.read_csv(movies_csv_path)

from ml_recsys_tools.data_handlers.interactions_with_features import ObsWithFeatures

obs = ObsWithFeatures(df_obs=ratings_df, df_items=movies_df,
                      uid_col='userid', iid_col='itemid', item_id_col='itemid')
train_obs, test_obs = obs.split_train_test(ratio=0.2)

# compare LightFM recommenders
from ml_recsys_tools.recommenders.lightfm_recommender import LightFMRecommender

# no features - just CF
cf_only = LightFMRecommender()
cf_only.fit(train_obs, epochs=20)
print(cf_only.eval_on_test_by_ranking(test_obs.df_obs, prefix='lfm ', n_rec=100))


# using movie genres and CF (hybrid mode) - slightly better
feature_columns = list(movies_df.columns.difference(['itemid']))
hybrid = LightFMRecommender(external_features=train_obs.get_item_features(bin_cols=feature_columns))
hybrid.fit(train_obs, epochs=20)
print(hybrid.eval_on_test_by_ranking(test_obs.df_obs, prefix='lfm hybrid ', n_rec=100))


# using only genres - much worse than both - but still better than chance
feature_columns = list(movies_df.columns.difference(['item_ind', 'itemid']))
only_feat = LightFMRecommender(external_features=train_obs.get_item_features(bin_cols=feature_columns),
                            external_features_params=dict(add_identity_mat=False))
only_feat.fit(train_obs, epochs=20)
print(only_feat.eval_on_test_by_ranking(test_obs.df_obs, prefix='lfm feat only ', n_rec=100))
