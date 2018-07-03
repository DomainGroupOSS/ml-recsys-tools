"""
Example explaining the peculiriaties of evaluation
"""

from ml_recsys_tools.datasets.prep_movielense_data import get_and_prep_data
import pandas as pd
from ml_recsys_tools.data_handlers.interaction_handlers_base import ObservationsDF
from ml_recsys_tools.recommenders.lightfm_recommender import LightFMRecommender

rating_csv_path, users_csv_path, movies_csv_path = get_and_prep_data()
ratings_df = pd.read_csv(rating_csv_path)

obs = ObservationsDF(ratings_df, uid_col='userid', iid_col='itemid')
train_obs, test_obs = obs.split_train_test(ratio=0.2)

# train and test LightFM recommender
lfm_rec = LightFMRecommender()
lfm_rec.fit(train_obs, epochs=10)

# print evaluation results:
# for LightFM there is an exact method that on large and sparse
# data might be too slow (for this data it's much faster though)
print(lfm_rec.eval_on_test_by_ranking_exact(test_obs.df_obs, prefix='lfm regular exact '))

# this ranking evaluation is done by sampling top n_rec recommendations
# rather than all ranks for all items (very slow and memory-wise expensive for large data).
# choosing higher values for n_rec makes
# the evaluation more accurate (less pessimmistic)
# this way the evaluation is mostly accurate for the top results,
# and is quite pessimmistic (especially for AUC, which scores for all ranks) and any non @k metric
print(lfm_rec.eval_on_test_by_ranking(test_obs.df_obs, prefix='lfm regular ', n_rec=100))
