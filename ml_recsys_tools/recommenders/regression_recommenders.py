from abc import abstractmethod

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from ml_recsys_tools.data_handlers.interaction_handlers_base import ObservationsDF
from ml_recsys_tools.recommenders.factorization_base import BaseFactorizationRecommender
from ml_recsys_tools.recommenders.lightfm_recommender import LightFMRecommender
from ml_recsys_tools.recommenders.recommender_base import BasePredictorRecommender


class BaseFactorsRegressor(BasePredictorRecommender):

    def __init__(self, stacking_split=0.5, **kwargs):
        super().__init__(**kwargs)
        self.stacking_split = stacking_split

    @abstractmethod
    def _init_regressor(self, **kwargs):
        self._regressor = BaseEstimator()
        return self

    @abstractmethod
    def _init_factorizer(self, **kwargs):
        self._factorizer = BaseFactorizationRecommender()
        return self

    def _make_features_df(self, df_inds, with_targets=True):
        cols = [self._uid_col, self._iid_col] + ([self._rating_col] if with_targets else [])
        df_inds = df_inds[cols]

        df_inds[self._prediction_col] = self._factorizer._predict(
            df_inds[self._uid_col].values, df_inds[self._iid_col].values)

        df_feat_u = pd.merge(
            df_inds[cols], self.df_user_factors.reset_index(),
            left_on=self._uid_col, right_on='index', how='left'). \
            drop([self._uid_col, 'index'], axis=1)

        df_fit_ui = pd.merge(
            df_feat_u, self.df_item_factors.reset_index(),
            left_on=self._iid_col, right_on='index', how='left'). \
            drop([self._iid_col, 'index'], axis=1)
        return df_fit_ui

    def _df_ids_to_df_inds(self, df):
        df = self._factorizer.sparse_mat_builder.remove_unseen_labels(df)
        return self._factorizer.sparse_mat_builder.add_encoded_cols(df)

    def _make_pos_neg_feat_df(self, df):
        df_inds = self._df_ids_to_df_inds(df)

        if len(df_inds):
            df_pos = self._make_features_df(df_inds, with_targets=True)

            df_inds_neg = df_inds.assign(
                **{'iid_coord': df_inds['iid_coord'].sample(n=len(df_inds)).values,
                   'rating': 0})
            df_neg = self._make_features_df(df_inds_neg, with_targets=True)

            return pd.concat([df_pos, df_neg], axis=0)

    def fit(self, train_obs: ObservationsDF, *args, **kwargs):
        factors_obs, reg_obs = train_obs.split_train_test(ratio=self.stacking_split)
        self._set_data(factors_obs)
        self._fit_factorizer(factors_obs)
        self._fit_regressor(reg_obs)

    def _fit_factorizer(self, factors_obs):
        self._init_factorizer()
        self._factorizer.fit(factors_obs)
        self._rating_col = self.sparse_mat_builder.rating_source_col
        self._uid_col = self.sparse_mat_builder.uid_col
        self._iid_col = self.sparse_mat_builder.iid_col
        self.df_user_factors = self._factorizer.user_factors_dataframe()
        self.df_item_factors = self._factorizer.item_factors_dataframe()

    def _fit_regressor(self, reg_obs):
        df_train_reg = self._make_pos_neg_feat_df(reg_obs.df_obs)
        self._init_regressor()
        self._regressor.fit(df_train_reg.drop('rating', axis=1).values,
                            df_train_reg['rating'].values)

    def _predict(self, user_inds, item_inds):
        df_inds = pd.DataFrame({
            self._uid_col: np.array(user_inds).ravel(),
            self._iid_col: np.array(item_inds).ravel()})
        df_with_feat = self._make_features_df(df_inds, with_targets=False)
        return self._regressor.predict(df_with_feat.values)

    def evaluate_regressor(self, test_dfs, test_names):
        scores = []
        for test_df in test_dfs:
            df_test_ref = self._make_pos_neg_feat_df(test_df)
            if df_test_ref is not None:
                y_true = df_test_ref['rating'].values
                y_pred = self._regressor.predict(df_test_ref.drop('rating', axis=1).values)
                score = r2_score(y_true, y_pred)
            else:
                score = None
            scores.append({'r2': score})
        return pd.DataFrame(data=scores, index=test_names)

    def _get_recommendations_flat(
            self, user_ids, n_rec, item_ids=None, exclude_training=True, **kwargs):

        return self.get_recommendations_exact(
            user_ids=user_ids, item_ids=item_ids, n_rec=n_rec,
            exclude_training=exclude_training, results_format='flat')

    def get_similar_items(self, item_ids=None, target_item_ids=None, n_simil=10,
                          remove_self=True, embeddings_mode=None,
                          simil_mode='cosine', results_format='lists', pbar=None):
        raise NotImplementedError

    def predict_for_user(self, user_id, item_ids, rank_training_last=True, sort=True):
        raise NotImplementedError

    def _predict_rank(self, test_mat, train_mat=None):
        raise NotImplementedError


class RFLFMRegRec(BaseFactorsRegressor):

    def _init_regressor(self, **kwargs):
        self._regressor = RandomForestRegressor(n_estimators=100)
        return self

    def _init_factorizer(self, **kwargs):
        self._factorizer = LightFMRecommender()
        self._factorizer.set_params(**dict(no_components=20, epochs=20))
        return self

