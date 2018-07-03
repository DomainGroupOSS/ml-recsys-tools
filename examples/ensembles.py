"""
This is an example on datasets-1M demonstrating how to combine recommender using ensemble recommenders
"""

from ml_recsys_tools.datasets.prep_movielense_data import get_and_prep_data
import pandas as pd
from ml_recsys_tools.data_handlers.interaction_handlers_base import ObservationsDF

rating_csv_path, users_csv_path, movies_csv_path = get_and_prep_data()
ratings_df = pd.read_csv(rating_csv_path)

obs = ObservationsDF(ratings_df, uid_col='userid', iid_col='itemid')
train_obs, test_obs = obs.split_train_test(ratio=0.2)

# train and test LightFM recommender
from ml_recsys_tools.recommenders.lightfm_recommender import LightFMRecommender

lfm_rec = LightFMRecommender()
lfm_rec.fit(train_obs, epochs=10)

# train and evaluate a Cooccurrence recommender (fast and deterministic)
from ml_recsys_tools.recommenders.similarity_recommenders import ItemCoocRecommender

item_cooc_rec = ItemCoocRecommender()
item_cooc_rec.fit(train_obs)
print(item_cooc_rec.eval_on_test_by_ranking(test_obs, prefix='item cooccurrence '))

# combine LightFM and Cooccurrence recommenders
# using recommendation ranks, and evaluate
from ml_recsys_tools.recommenders.combination_ensembles import CombinedRankEnsemble

comb_ranks_rec = CombinedRankEnsemble(
    recommenders=[lfm_rec, item_cooc_rec])
print(comb_ranks_rec.eval_on_test_by_ranking(test_obs, prefix='combined ranks '))

# combine LightFM and Cooccurrence recommenders
# using similarities, and evaluate
from ml_recsys_tools.recommenders.combination_ensembles import CombinedSimilRecoEns

comb_simil_rec = CombinedSimilRecoEns(
    recommenders=[lfm_rec, item_cooc_rec])
comb_simil_rec.fit(train_obs)
print(comb_simil_rec.eval_on_test_by_ranking(test_obs, prefix='combined simils '))

# combine LightFM, Cooccurrence and Combined Simil recommender
# using recommendation ranks, and evaluate
from ml_recsys_tools.recommenders.combination_ensembles import CombinedRankEnsemble

comb_ranks_simil_rec = CombinedRankEnsemble(
    recommenders=[lfm_rec, item_cooc_rec, comb_simil_rec])
print(comb_ranks_simil_rec.eval_on_test_by_ranking(
    test_obs, prefix='combined ranks and simils '))

# get all recommendations and print a sample
recs_ens = comb_ranks_simil_rec.get_recommendations(
    comb_ranks_simil_rec.all_users, n_rec=5)
print(recs_ens.sample(5))

# get all similarities and print a sample
simils_ens = comb_ranks_simil_rec.get_similar_items(
    comb_ranks_simil_rec.all_items, n_simil=5)
print(simils_ens.sample(10))
