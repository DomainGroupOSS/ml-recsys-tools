"""
This is an example on movielens-1M demonstrating some additional recommenders
"""

from examples.prep_movielense_data import get_and_prep_data
import pandas as pd
from ml_recsys_tools.data_handlers.interaction_handlers_base import ObservationsDF

rating_csv_path, users_csv_path, movies_csv_path = get_and_prep_data()
ratings_df = pd.read_csv(rating_csv_path)

obs = ObservationsDF(ratings_df, uid_col='userid', iid_col='itemid')
train_obs, test_obs = obs.split_train_test(ratio=0.2)

# train and evaluate a Cooccurrence recommender (fast and deterministic)
from ml_recsys_tools.recommenders.spotlight_recommenders import EmbeddingFactorsRecommender
emb_rec = EmbeddingFactorsRecommender()
emb_rec.fit(train_obs)
print(emb_rec.eval_on_test_by_ranking_exact(test_obs, prefix='implicit embeddings '))
