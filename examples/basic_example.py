"""
This is a basic example on movielens-1M demonstrating:
    - basic data ingestion without any item/user features
    - LightFM recommender:
        fit, evaluation, early stopping,
        hyper-param search, recommendations, similarities
    - Cooccurrence recommender
    - Two combination ensembles (Ranks and Simils)
"""

# dataset: download and prepare dataframes
from examples.prep_movielense_data import get_and_prep_data
rating_csv_path, users_csv_path, movies_csv_path = get_and_prep_data()


# read the interactions dataframe and create a data handler object and  split to train and test
import pandas as pd
ratings_df = pd.read_csv(rating_csv_path)
from ml_recsys_tools.data_handlers.interaction_handlers_base import ObservationsDF
obs = ObservationsDF(ratings_df)
train_obs, test_obs = obs.split_train_test(ratio=0.2, users_ratio=1.0)


# train and test LightFM recommender
from ml_recsys_tools.recommenders.lightfm_recommender import LightFMRecommender
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


# get all recommendations and print a sample (training interactions are filtered out by default)
recs = lfm_rec.get_recommendations(lfm_rec.all_users(), n_rec=5)
print(recs.sample(5))

# get all similarities and print a sample
simils = lfm_rec.get_similar_items(lfm_rec.all_items(), n_simil=5)
print(simils.sample(10))


# train LightFM with early stopping and print evaluation results
lfm_rec.fit_with_early_stop(train_obs, epochs_max=30, epochs_step=1, stop_patience=1,
                            valid_ratio=0.2, metric='n-MRR@10', refit_on_all=True)
print(lfm_rec.eval_on_test_by_ranking(test_obs.df_obs, prefix='lfm early stop '))


# perform a hyperparameter search on LightFM recommender
space = lfm_rec.guess_search_space()
hp_results = lfm_rec.hyper_param_search(
    train_obs,
    metric='n-MRR@10',
    hp_space=dict(
        no_components=space.Integer(20, 100),
        epochs=space.Integer(5, 50),
        item_alpha=space.Real(1e-8, 1e-5, prior='log-uniform')
    ),
    n_iters=5,
    )
print(hp_results.report)
# refit best model on full training data
# and print evaluation results
hp_results.best_model.fit(train_obs)
print(hp_results.best_model.eval_on_test_by_ranking(test_obs, prefix='lfm hp search '))


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
    comb_ranks_simil_rec.all_users(), n_rec=5)
print(recs_ens.sample(5))

# get all similarities and print a sample
simils_ens = comb_ranks_simil_rec.get_similar_items(
    comb_ranks_simil_rec.all_items(), n_simil=5)
print(simils_ens.sample(10))