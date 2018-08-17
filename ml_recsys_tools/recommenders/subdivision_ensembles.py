import warnings

from ml_recsys_tools.data_handlers.interactions_with_features import ObsWithGeoFeatures
from ml_recsys_tools.data_handlers.interaction_handlers_base import RANDOM_STATE
from ml_recsys_tools.recommenders.similarity_recommenders import ItemCoocRecommender
from ml_recsys_tools.recommenders.lightfm_recommender import LightFMRecommender
from ml_recsys_tools.recommenders.ensembles_base import SubdivisionEnsembleBase


class GeoGridEnsembleBase(SubdivisionEnsembleBase):

    def __init__(self,
                 geo_grid_params=None,
                 n_lat=3,
                 n_long=3,
                 overlap_margin=0.5,
                 min_interactions=0,
                 **kwargs):
        self.geo_box = geo_grid_params
        self.n_lat = n_lat
        self.n_long = n_long
        self.min_interactions = min_interactions
        self.overlap_margin = overlap_margin
        super().__init__(**kwargs)

    @property
    def n_recommenders(self):
        return self.n_lat * self.n_long

    @n_recommenders.setter
    def n_recommenders(self, n_recommenders):
        warnings.warn('Cannot set n_recommenders for GeoGridEnsembleBase, set n_lat and n_long instead')

    def set_params(self, **params):
        """
        this is for skopt / sklearn compatibility
        """
        params = self._pop_set_params(
            params, ['n_lat', 'n_long', 'overlap_margin', 'min_interactions'])

        super().set_params(**params.copy())

    def _generate_sub_model_train_data(self, train_obs: ObsWithGeoFeatures):

        if self.geo_box is None:
            self.geo_box = {
                'max_lat': train_obs.df_items[train_obs.lat_col].max(),
                'min_lat': train_obs.df_items[train_obs.lat_col].min(),
                'max_long': train_obs.df_items[train_obs.long_col].max(),
                'min_long': train_obs.df_items[train_obs.long_col].min(),
            }

        self.geo_filters = train_obs.calcluate_equidense_geo_grid(
            n_lat=self.n_lat, n_long=self.n_long,
            overlap_margin=self.overlap_margin, geo_box=self.geo_box)

        for geo_filt in self.geo_filters:
            yield train_obs. \
                filter_by_location_rectangle(*geo_filt). \
                sample_observations(
                min_user_hist=self.min_interactions,
                min_item_hist=self.min_interactions,
                random_state=RANDOM_STATE)


class LFMEnsembleBase(LightFMRecommender, SubdivisionEnsembleBase):

    def __init__(self,
                 use_item_features=False,
                 item_features_params=None,
                 **kwargs):
        self.use_item_features = use_item_features
        self.item_features_params = item_features_params
        super().__init__(**kwargs)

    def _init_recommenders(self, **params):
        self.sub_class_type = LightFMRecommender
        super()._init_recommenders(**{'use_sample_weight': self.use_sample_weight,
                                    # 'sparse_mat_builder': self.sparse_mat_builder,
                                    'model_params': self.model_params,
                                    'fit_params': self.fit_params
                                    }, **params)
        # self._set_sub_model_params()

    def set_params(self, **params):
        """
        this is for skopt / sklearn compatibility
        """
        params = self._pop_set_params(params, ['use_item_features'])

        super().set_params(**params.copy())

    def _fit_sub_model(self, args):
        i_m, obs, fit_params = args

        # external features
        if self.use_item_features:
            fit_params['external_features'] = \
                obs.get_item_features_for_obs(**self.item_features_params)

        # self.recommenders[i_m] = sub_model_fit_func(self.recommenders[i_m], obs)
        self.recommenders[i_m].fit(obs, **fit_params)
        return self.recommenders[i_m]

    fit = SubdivisionEnsembleBase.fit
    get_similar_items = SubdivisionEnsembleBase.get_similar_items
    _get_recommendations_flat = SubdivisionEnsembleBase._get_recommendations_flat
    predict_for_user = SubdivisionEnsembleBase.predict_for_user


class LFMGeoGridEnsemble(GeoGridEnsembleBase, LFMEnsembleBase):
    pass


class GeoClusteringEnsembleBase(SubdivisionEnsembleBase):

    def _generate_sub_model_train_data(self, train_obs: ObsWithGeoFeatures):
        train_obs.geo_cluster_items(n_clusters=len(self.recommenders))

        labels = train_obs.df_items[train_obs.cluster_label_col].unique()

        for label in labels:
            yield train_obs. \
                filter_by_cluster_label(label)


class LFMGeoClusteringEnsemble(GeoClusteringEnsembleBase, LFMEnsembleBase):
    pass


class CoocEnsembleBase(SubdivisionEnsembleBase, ItemCoocRecommender):

    def _init_recommenders(self, **params):
        self.sub_class_type = ItemCoocRecommender
        super()._init_recommenders(fit_params=self.fit_params, **params)

    def _fit_sub_model(self, args):
        i_m, obs, fit_params = args

        # self.recommenders[i_m] = sub_model_fit_func(self.recommenders[i_m], obs)
        self.recommenders[i_m].fit(obs, **fit_params)
        return self.recommenders[i_m]


class CoocGeoGridEnsemble(CoocEnsembleBase, GeoGridEnsembleBase):
    pass
