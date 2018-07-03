"""
This is an example on datasets-1M demonstrating:
    - More advanced fitting features: fit, evaluation, early stopping, hyper-param search
"""

from ml_recsys_tools.datasets.prep_movielense_data import get_and_prep_data
import pandas as pd
from ml_recsys_tools.data_handlers.interaction_handlers_base import ObservationsDF
from ml_recsys_tools.recommenders.lightfm_recommender import LightFMRecommender

rating_csv_path, users_csv_path, movies_csv_path = get_and_prep_data()
ratings_df = pd.read_csv(rating_csv_path)

obs = ObservationsDF(ratings_df, uid_col='userid', iid_col='itemid')
train_obs, test_obs = obs.split_train_test(ratio=0.2)
lfm_rec = LightFMRecommender()


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
