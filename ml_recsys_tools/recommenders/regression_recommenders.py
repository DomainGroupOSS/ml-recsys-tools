from abc import abstractmethod

import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from ml_recsys_tools.data_handlers.interaction_handlers_base import ObservationsDF
from ml_recsys_tools.recommenders.factorization_base import BaseFactorizationRecommender
from ml_recsys_tools.recommenders.lightfm_recommender import LightFMRecommender
from ml_recsys_tools.recommenders.recommender_base import BasePredictorRecommender


class BaseFactorsRegressor(BasePredictorRecommender):

    default_regressor_params = {}
    default_factorizer_params = {}

    factorizer_class = BaseFactorizationRecommender
    regressor_class = RandomForestRegressor  # placeholder for generic regressor

    def __init__(self,
                 stacking_split=0.5,
                 factors_prediction=True,
                 user_factors=True,
                 item_factors=True,
                 item_features=True,
                 target_transform='log',
                 regressor_params=None,
                 item_features_params=None,
                 factorizer_params=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.stacking_split = stacking_split
        self.factors_prediction = factors_prediction
        self.user_factors = user_factors
        self.item_factors = item_factors
        self.item_features = item_features
        self.target_transform_func = target_transform
        self._check_param_keys_conflicts()
        self.item_features_params = item_features_params \
            if item_features_params is not None else {}
        self.regressor_params = self._dict_update(
            self.default_regressor_params, regressor_params)
        self.factorizer_params = self._dict_update(
            self.default_factorizer_params, factorizer_params)

    def _check_param_keys_conflicts(self):
        assert len(
            set(self.default_regressor_params.keys()).intersection(
            set(self.default_factorizer_params.keys()))) == 0

    def set_params(self, **params):
        params = self._pop_set_params(
            params, ['stacking_split', 'factors_prediction',
                     'user_factors', 'item_factors',
                     'target_transform', 'item_features'])
        params = self._pop_set_dict(
            self.regressor_params, params, self.default_regressor_params.keys())
        params = self._pop_set_dict(
            self.factorizer_params, params, self.default_factorizer_params.keys())
        super().set_params(**params)

    def _init_regressor(self):
        self._regressor = self.regressor_class(**self.regressor_params)

    def _init_factorizer(self):
        self._factorizer = self.factorizer_class()
        self._factorizer.set_params(**self.factorizer_params)

    def _transform_targets(self, targets):
        if self.target_transform_func == 'log':
            return np.log(1 + np.min(targets))
        elif self.target_transform_func in [None, 'none']:
            return targets
        else:
            return NotImplementedError('Uknown target transform function')

    def _make_features_df(self, df_inds, with_targets=True):
        cols = [self._uid_col, self._iid_col] + \
               ([self._rating_col] if with_targets else [])
        df_feat = df_inds[cols].copy()

        if with_targets:
            df_feat[self._rating_col] = self._transform_targets(
                df_feat[self._rating_col].values)

        if self.item_features:
            df_feat = pd.merge(
                df_feat, self.df_item_features.reset_index(),
                left_on=self._iid_col, right_on='level_0', how='left'). \
                drop('level_0', axis=1)

        if self.factors_prediction:
            df_feat[self._prediction_col] = self._factorizer._predict_on_inds(
                df_feat[self._uid_col].values, df_feat[self._iid_col].values)

        if self.user_factors:
            df_feat = pd.merge(
                df_feat, self.df_user_factors.reset_index(),
                left_on=self._uid_col, right_on='level_0', how='left'). \
                drop('level_0', axis=1)

        if self.item_factors:
            df_feat = pd.merge(
                df_feat, self.df_item_factors.reset_index(),
                left_on=self._iid_col, right_on='level_0', how='left'). \
                drop('level_0', axis=1)

        df_feat.drop([self._uid_col, self._iid_col], axis=1, inplace=True)

        return df_feat

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

    def _set_item_features_df(self, train_obs):
        if hasattr(train_obs, 'df_items'):
            ext_feat = train_obs.get_item_features_for_obs(**self.item_features_params)
            mat_builder = train_obs.get_sparse_matrix_helper()
            feat_mat = ext_feat.fit_transform_ids_df_to_mat(
                mat_builder.iid_encoder, mode='encode')
            self.df_item_features = pd.DataFrame(
                index=np.arange(feat_mat.shape[0]),
                data=feat_mat)
        else:
            raise ValueError('df_items not present in training ObservationHandler')

    def fit(self, train_obs: ObservationsDF, *args, **kwargs):
        factors_obs, reg_obs = train_obs.split_train_test(ratio=self.stacking_split)
        self._set_item_features_df(train_obs)
        self._set_data(factors_obs)
        self._fit_factorizer(factors_obs)
        self._fit_regressor(reg_obs)

    def _fit_factorizer(self, factors_obs, **fit_params):
        self._init_factorizer()
        self._factorizer.fit(factors_obs, **fit_params)
        self._rating_col = self.sparse_mat_builder.rating_source_col
        self._uid_col = self.sparse_mat_builder.uid_col
        self._iid_col = self.sparse_mat_builder.iid_col
        self.df_user_factors = self._factorizer.user_factors_dataframe()
        self.df_item_factors = self._factorizer.item_factors_dataframe()

    def _fit_regressor(self, reg_obs, **fit_params):
        df_train_reg = self._make_pos_neg_feat_df(reg_obs.df_obs)
        self._init_regressor()
        self._regressor.fit(df_train_reg.drop(self._rating_col, axis=1).values,
                            df_train_reg[self._rating_col].values, **fit_params)

    def _predict_on_inds(self, user_inds, item_inds):
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
                y_true = df_test_ref[self._rating_col].values
                y_pred = self._regressor.predict(df_test_ref.drop(self._rating_col, axis=1).values)
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
                          simil_mode='cosine', results_format='lists'):
        raise NotImplementedError

    def predict_for_user(self, user_id, item_ids, rank_training_last=True,
                         sort=True, combine_original_order=False):
        raise NotImplementedError

    def _predict_rank(self, test_mat, train_mat=None):
        raise NotImplementedError


class BaseLFMRegRec(BaseFactorsRegressor):
    default_factorizer_params = dict(**LightFMRecommender.default_model_params,
                                     **{'epochs': 20})
    factorizer_class = LightFMRecommender


class BaseRFRegRec(BaseFactorsRegressor):
    default_regressor_params = dict(n_estimators=100, n_jobs=-1)
    regressor_class = RandomForestRegressor


class BaseLGBMRecReg(BaseFactorsRegressor):
    default_regressor_params = dict(
        boosting_type="gbdt",
        objective='regression',
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.1,
        n_estimators=100,
        min_child_weight=1e-3,  # what's the right setting for this case?
        min_child_samples=20,  # what's the right setting for this case?
        subsample=1.,
        subsample_freq=1,
        colsample_bytree=1.,
        reg_alpha=0.,
        reg_lambda=0.,
        silent=True,
        verbose=-1,
    )
    regressor_class = LGBMRegressor


class RFonLFMRegRec(BaseLFMRegRec, BaseRFRegRec):
    pass


class LGBMonLFMRegRec(BaseLFMRegRec, BaseLGBMRecReg):
    pass