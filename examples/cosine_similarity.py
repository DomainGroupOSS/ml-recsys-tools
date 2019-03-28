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

from ml_recsys_tools.recommenders.similarity_recommenders import FeaturesSimilRecommender

# list of feature columns
feature_columns = list(movies_df.columns.difference(['itemid']))

# fit cosine similarity recommender
sim_rec = FeaturesSimilRecommender(simil_mode='cosine', n_simil=500)
sim_rec.fit(train_obs)

# training is excluded because accuracy on train 0 - because training is directly used to sample recommendations
print(sim_rec.eval_on_test_by_ranking(test_obs.df_obs, prefix='cosine ', n_rec=100, include_train=False))

# only retrieves movies with similar genres..
sim_rec.get_similar_items(item_ids=['Fight Club (1999)',
                                    'Godfather, The (1972)',
                                    'Ace Ventura: Pet Detective (1994)',
                                    'Aladdin (1992)'])

