"""
using multiple test sets
"""

# dataset: download and prepare dataframes
from examples.prep_movielense_data import get_and_prep_data
rating_csv_path, users_csv_path, movies_csv_path = get_and_prep_data()


# read the interactions dataframe and create a data handler object and  split to train and test
import pandas as pd
ratings_df = pd.read_csv(rating_csv_path)
from ml_recsys_tools.data_handlers.interaction_handlers_base import ObservationsDF
obs = ObservationsDF(ratings_df)
train_obs, test_obs = obs.split_train_test(ratio=0.2, users_ratio=0.2)


# train and test LightFM recommender
from ml_recsys_tools.recommenders.lightfm_recommender import LightFMRecommender
lfm_rec = LightFMRecommender()
lfm_rec.fit(train_obs, epochs=10)

# let's construct more interesting test sets
train_df = train_obs.df_obs
test_df = test_obs.df_obs

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

# evaluation
print(lfm_rec.eval_on_test_by_ranking(
    test_dfs=test_dfs, test_names=test_names, prefix='lfm regular '))

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