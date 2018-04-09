import pandas as pd

from ml_recsys_tools.examples.prep_movielense_data import get_and_prep_data
rating_csv_path, users_csv_path, movies_csv_path = get_and_prep_data()
ratings_df = pd.read_csv(rating_csv_path)


from ml_recsys_tools.data_handlers.interaction_handlers_base import ObservationsDF
obs = ObservationsDF(ratings_df)
train_obs, test_obs = obs.split_train_test()


from ml_recsys_tools.recommenders.lightfm_recommender import LightFMRecommender
model_params = LightFMRecommender.default_model_params
lfm_rec = LightFMRecommender(model_params=model_params)


lfm_rec.fit(train_obs, epochs=10)
print(lfm_rec.eval_on_test_by_ranking(test_obs.df_obs, prefix='lfm regular '))


lfm_rec.fit_with_early_stop(train_obs, epochs_max=30, epochs_step=1, stop_patience=1,
                            valid_ratio=0.2, metric='n-MRR@10', refit_on_all=True)
print(lfm_rec.eval_on_test_by_ranking(test_obs.df_obs, prefix='lfm early stop '))


space = lfm_rec.guess_search_space()
hp_results = lfm_rec.hyper_param_search(
    train_obs,
    metric='n-MRR@10',
    hp_space=dict(
        no_components=space.Integer(20, 100),
        epochs=space.Integer(5, 50),
        item_alpha=space.Real(1e-8, 1e-5, prior='log-uniform')
    ),
    n_iters=20,
    )
print(hp_results.report)

hp_results.best_model.fit(train_obs)
print(hp_results.best_model.eval_on_test_by_ranking(test_obs, prefix='lfm early stop'))


from ml_recsys_tools.recommenders.similarity_recommenders import ItemCoocRecommender
item_cooc_rec = ItemCoocRecommender()
item_cooc_rec.fit(train_obs)
print(item_cooc_rec.eval_on_test_by_ranking(test_obs, prefix='item cooccurrence '))


from ml_recsys_tools.recommenders.combination_ensembles import CombinedRankEnsemble
comb_ranks_rec = CombinedRankEnsemble(recommenders=[lfm_rec, item_cooc_rec])
print(comb_ranks_rec.eval_on_test_by_ranking(test_obs, prefix='combined ranks '))


from ml_recsys_tools.recommenders.combination_ensembles import CombinedSimilRecoEns
comb_ranks_and_simi_rec = CombinedSimilRecoEns(recommenders=[lfm_rec, item_cooc_rec])
comb_ranks_and_simi_rec.fit(train_obs)
print(comb_ranks_and_simi_rec.eval_on_test_by_ranking(test_obs, prefix='combined ranks and simils '))


recs = lfm_rec.get_recommendations(lfm_rec.all_training_users())
recs.sample(5)

simils = lfm_rec.get_similar_items(lfm_rec.all_training_items())
simils.sample(5)

simils = comb_ranks_and_simi_rec.get_similar_items(lfm_rec.all_training_items())
simils.sample(5)
