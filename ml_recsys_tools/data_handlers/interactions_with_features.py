import copy

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import scipy.sparse as sp

from sklearn.preprocessing import LabelBinarizer, normalize
from sklearn_pandas import DataFrameMapper
from ml_recsys_tools.utils.sklearn_extenstions import FloatBinningBinarizer

from ml_recsys_tools.data_handlers.interaction_handlers_base import ObservationsDF, RANDOM_STATE
from ml_recsys_tools.utils.instrumentation import log_time_and_shape
from ml_recsys_tools.utils.geo import ItemsGeoMapper
from ml_recsys_tools.utils.logger import simple_logger as logger


class ExternalFeaturesDF:
    """
    handles external items features and feature engineering
    """

    def __init__(self, feat_df, id_col, num_cols=None, cat_cols=None):
        self.feat_df = feat_df
        self.id_col = id_col
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        if self.num_cols is None and self.cat_cols is None:
            self._infer_categorical_numerical_columns()

    def _infer_categorical_numerical_columns(self,
                                             categorical_unique_ratio=0.05,
                                             categorical_n_unique=20):

        len_df = len(self.feat_df)

        if not len_df:
            raise ValueError('Features DF is empty')

        feat_cols = self.feat_df.columns.difference([self.id_col])

        self.num_cols, self.cat_cols = [], []

        for col in feat_cols:
            if str(self.feat_df[col].dtype) in ['O', 'o']:
                self.cat_cols.append(col)
            else:
                n_unique = self.feat_df[col].nuniques
                unique_ratio = n_unique / len_df
                if n_unique < categorical_n_unique or \
                        unique_ratio <= categorical_unique_ratio:
                    self.cat_cols.append(col)
                else:
                    self.num_cols.append(col)

    def apply_selection_filter(self, selection_filter=None):
        if selection_filter is not None and len(selection_filter) >= 1:
            # no selection applied for None, '', []
            self.cat_cols = [col for col in self.cat_cols if col in selection_filter]
            self.num_cols = [col for col in self.num_cols if col in selection_filter]

        self.feat_df = self.feat_df[[self.id_col] + self.cat_cols + self.num_cols]

        return self

    @log_time_and_shape
    def create_items_features_matrix(self,
                                     items_encoder,
                                     normalize_output=False,
                                     add_identity_mat=True,
                                     numeric_n_bins=100,
                                     feat_weight=1.0):
        """
        creates a sparse feature matrix from item features

        :param items_encoder: the encoder that is used to filter and
            align the features dataframe to the sparse matrices
        :param add_identity_mat: indicator whether to add a sparse identity matrix
            (as used when no features are used), as per LightFM's docs suggestion
        :param normalize_output:
            None (default) - no normalization
            'rows' - normalize rows with l1 norm
            anything else - normalize cols with l1 norm
        :param numeric_n_bins: number of bins for binning numeric features
        :param feat_weight: feature weight relative to identity matrix (can be used to emphasize one or the other)

        :return: sparse feature mat n_items x n_features
        """

        # get numeric cols
        feat_df = self.feat_df[
            [self.id_col] + self.cat_cols + self.num_cols]
        # get only features for relevant items
        feat_df = feat_df[feat_df[self.id_col].isin(items_encoder.classes_)]
        # convert from id to index
        feat_df['item_ind'] = items_encoder.transform(feat_df[self.id_col])

        # reorder in correct index order
        n_items = len(items_encoder.classes_)
        full_feat_df = pd.merge(
            pd.DataFrame({'item_ind': np.arange(n_items)}),
            feat_df.drop([self.id_col], axis=1), on='item_ind', how='left'). \
            drop_duplicates('item_ind'). \
            set_index('item_ind', drop=True)

        # remove nans resulting form join
        # https://stackoverflow.com/questions/34913590/fillna-in-multiple-columns-in-place-in-python-pandas
        full_feat_df = full_feat_df.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna('.'))

        full_feat_df[self.cat_cols] = \
            full_feat_df[self.cat_cols].astype(str)

        if len(full_feat_df):
            feat_mat, df_transformer = self.ohe_hot_encode_features_df(
                full_feat_df, self.num_cols, self.num_cols,
                numeric_n_bins=numeric_n_bins)

            # normalize each row
            if normalize_output:
                axis = int(normalize_output == 'rows')
                feat_mat = normalize(feat_mat, norm='l1', axis=axis, copy=False)

            # weight down the features before adding the identity mat
            feat_mat = feat_mat.astype(np.float32) * feat_weight

            if add_identity_mat:
                # identity mat
                id_mat = sp.identity(n_items, dtype=np.float32, format='csr')

                full_feat_mat = self.concatenate_csc_matrices_by_columns(
                    feat_mat.tocsc(), id_mat.tocsc()).tocsr()
            else:
                full_feat_mat = feat_mat

            return full_feat_mat

        else:
            return None

    @staticmethod
    def concatenate_csc_matrices_by_columns(matrix1, matrix2):
        # https://stackoverflow.com/a/33259578/6485667
        new_data = np.concatenate((matrix1.data, matrix2.data))
        new_indices = np.concatenate((matrix1.indices, matrix2.indices))
        new_ind_ptr = matrix2.indptr + len(matrix1.data)
        new_ind_ptr = new_ind_ptr[1:]
        new_ind_ptr = np.concatenate((matrix1.indptr, new_ind_ptr))

        return sp.csc_matrix((new_data, new_indices, new_ind_ptr), dtype=np.float32)

    @staticmethod
    def ohe_hot_encode_features_df(full_feat_df, categorical_feat_cols, numeric_feat_cols,
                                   numeric_n_bins=50):
        feat_mapper = DataFrameMapper(
            [(cat_col,
              LabelBinarizer(sparse_output=True))
             for cat_col in categorical_feat_cols] +
            [(num_col,
              FloatBinningBinarizer(n_bins=numeric_n_bins, sparse_output=True))
             for num_col in numeric_feat_cols],
            sparse=True
        )
        feat_mat = feat_mapper.fit_transform(full_feat_df)
        feat_mat.eliminate_zeros()
        return feat_mat, feat_mapper


class ObsWithFeatures(ObservationsDF):

    def __init__(self, df_obs, df_items, items_iid_col='item_id', **kwargs):
        super().__init__(df_obs=df_obs, **kwargs)
        self.item_id_col = items_iid_col
        self.cluster_label_col = 'cluster_label'
        self.df_items = self._preprocess_items_df(df_items)

    def _preprocess_items_df(self, df_items):
        # make sure the ID col is of object type
        df_items[self.item_id_col] = df_items[self.item_id_col].astype(str)
        df_items.drop_duplicates(self.item_id_col, inplace=True)
        return df_items

    @log_time_and_shape
    def _filter_relevant_obs_and_items(self):
        items_ids = self.df_items[self.item_id_col].unique()
        obs_ids = self.df_obs[self.iid_col].unique()
        self.df_obs = self.df_obs[self.df_obs[self.iid_col].isin(items_ids)].copy()
        self.df_items = self.df_items[self.df_items[self.item_id_col].isin(obs_ids)].copy()

    def filter_by_cluster_label(self, label):
        assert self.cluster_label_col in self.df_items.columns
        other = copy.deepcopy(self)
        other.df_items = self.df_items[self.df_items[self.cluster_label_col] == label].copy()
        other._filter_relevant_obs_and_items()
        return other

    def _apply_filter(self, mask_filter):
        other = copy.deepcopy(self)
        other.df_items = self.df_items[mask_filter].copy()
        other._filter_relevant_obs_and_items()
        return other

    def sample_observations(self,
                            n_users=None,
                            n_items=None,
                            method='random',
                            min_user_hist=0,
                            min_item_hist=0,
                            random_state=None):
        sample_df = super().sample_observations(n_users=n_users,
                                                n_items=n_items,
                                                method=method,
                                                min_user_hist=min_user_hist,
                                                min_item_hist=min_item_hist,
                                                random_state=random_state)
        other = copy.deepcopy(self)
        other.df_obs = sample_df.df_obs
        other._filter_relevant_obs_and_items()
        return other

    def filter_columns_by_df(self, other_df_obs):
        """
        removes users or items that are not in the other user dataframe
        :param other_df_obs: other dataframe, that has the same structure (column names)
        :return: new observation handler
        """
        other = super().filter_columns_by_df(other_df_obs)
        other._filter_relevant_obs_and_items()
        return other

    def remove_interactions_by_df(self, other_df_obs):
        other = super().remove_interactions_by_df(other_df_obs)
        other._filter_relevant_obs_and_items()
        return other

    def items_filtered_by_ids(self, item_ids):
        return self.df_items[self.df_items[self.item_id_col].isin(item_ids)]

    def get_items_df_for_user(self, user):
        item_ids = self.user_filtered_df(user)[self.iid_col].unique().tolist()
        df_items_view = self.items_filtered_by_ids(item_ids)
        return df_items_view

    def get_item_features_for_obs(self,
                                  categorical_unique_ratio=0.05,
                                  categorical_n_unique=20,
                                  selection_filter=None, **kwargs):
        """

        :param categorical_n_unique: consider categorical if less unique values than this
        :param categorical_unique_ratio: consider categorical if ratio of uniques to length less than this
        :param selection_filter: include only those column, if None or empty includes all
        :param kwargs:
        :return: dataframe of features, list of numeric columns, list of categorical columns
        """
        feat_df = self.df_items

        ext_feat = ExternalFeaturesDF(
            feat_df=feat_df, id_col=self.item_id_col)

        ext_feat.apply_selection_filter(selection_filter)

        return ext_feat


class ObsWithGeoFeatures(ObsWithFeatures):

    def __init__(self, df_obs, df_items, lat_col='lat', long_col='long', **kwargs):
        super().__init__(df_obs=df_obs, df_items=df_items, **kwargs)
        self.lat_col = lat_col
        self.long_col = long_col
        self.df_items = self._preprocess_geo_cols(self.df_items)

    @property
    def geo_cols(self):
        return [self.lat_col, self.long_col]

    def _preprocess_geo_cols(self, df_items):
        # remove nans
        filt_nan = df_items[self.lat_col].notnull() & \
                   ~df_items[self.lat_col].isin(['None']) & \
                   ~df_items[self.lat_col].isin(['nan'])
        df_no_nans = df_items[filt_nan].copy()

        # to float
        df_no_nans[self.lat_col] = df_no_nans[self.lat_col].astype(float)
        df_no_nans[self.long_col] = df_no_nans[self.long_col].astype(float)

        return df_no_nans

    @log_time_and_shape
    def filter_by_location_range(self, min_lat, max_lat, min_long, max_long):
        """ e.g.
        min_lat = -33.851674299999999
        max_lat = -33.767248700000003
        min_long = 151.09386979999999
        max_long = 151.31997699999999
        """

        geo_filt = (self.df_items[self.lat_col] <= max_lat) & (self.df_items[self.lat_col] >= min_lat) & \
                   (self.df_items[self.long_col] <= max_long) & (self.df_items[self.long_col] >= min_long)

        return self._apply_filter(geo_filt)

    def filter_by_location_circle(self, center_lat, center_long, degree_dist):

        geo_filt = ((self.df_items[self.lat_col] - center_lat) ** 2 +
                    (self.df_items[self.long_col] - center_long) ** 2) <= degree_dist

        return self._apply_filter(geo_filt)

    def filter_by_location_square(self, center_lat, center_long, degree_side):
        offset = degree_side / 2
        return self.filter_by_location_range(
            center_lat - offset, center_lat + offset, center_long - offset, center_long + offset)

    def filter_by_location_rectangle(self, center_lat, center_long, lat_side, long_side):
        offset_lat = lat_side / 2
        offset_long = long_side / 2
        return self.filter_by_location_range(
            center_lat - offset_lat, center_lat + offset_lat,
            center_long - offset_long, center_long + offset_long)

    def geo_cluster_items(self, n_clusters=20):
        cls = MiniBatchKMeans(n_clusters=n_clusters,
                              random_state=RANDOM_STATE). \
            fit(self.df_items[self.geo_cols].as_matrix())

        self.df_items[self.cluster_label_col] = cls.labels_

    @log_time_and_shape
    def calcluate_equidense_geo_grid(self, n_lat, n_long, overlap_margin, geo_box):

        df_items = self.filter_by_location_range(**geo_box).df_items

        geo_filters = []
        lat_bins = np.percentile(df_items.lat, np.linspace(0, 100, n_lat + 1))

        for i in range(n_lat):
            cur_lat_bins = lat_bins[i:(i + 2)]

            prop_slice = df_items[(df_items.lat <= cur_lat_bins[1]) & (df_items.lat >= cur_lat_bins[0])]

            long_bins = np.percentile(prop_slice.long, np.linspace(0, 100, n_long + 1))

            lat_center = np.mean(cur_lat_bins)
            side_lat = (cur_lat_bins[1] - cur_lat_bins[0]) + overlap_margin * 2

            for j in range(n_long):
                long_center = np.mean(long_bins[j:(j + 2)])
                side_long = (long_bins[j + 1] - long_bins[j]) + overlap_margin * 2
                geo_filters.append((lat_center, long_center, side_lat, side_long))

        return geo_filters

    @staticmethod
    def calcluate_simple_geo_grid(n_lat, n_long, overlap_margin, geo_box):
        # ranges
        lat_range = geo_box['max_lat'] - geo_box['min_lat']
        long_range = geo_box['max_long'] - geo_box['min_long']

        # distance from center to borders without overlap
        d_lat = lat_range / (n_lat * 2)
        d_long = long_range / (n_long * 2)

        # center locations
        lat_centers = np.linspace(
            geo_box['min_lat'] + d_lat, geo_box['max_lat'] - d_lat, n_lat)
        long_centers = np.linspace(
            geo_box['min_long'] + d_long, geo_box['max_long'] - d_long, n_long)

        # add overlap to distances
        side_lat = 2 * (d_lat + overlap_margin)
        side_long = 2 * (d_long + overlap_margin)

        # create geo filters
        geo_filters = []
        for lat_center in lat_centers:
            for long_center in long_centers:
                geo_filters.append((lat_center, long_center, side_lat, side_long))
        return geo_filters

    def get_mapper(self):
        return ObsGeoFeatMapper(obs_handler=self, mapper_class=ItemsGeoMapper)


class ObsGeoFeatMapper(ObsWithGeoFeatures):

    def __init__(self, obs_handler, mapper_class=ItemsGeoMapper, **kwargs):
        self.mapper_class = mapper_class
        super().__init__(df_obs=obs_handler.df_obs, df_items=obs_handler.df_items, **kwargs)

    def map_items_for_user(self, user):
        return self.mapper_class(self.get_items_df_for_user(user))

    def map_items_by_common_items(self, item_id, default_marker_size=2):

        users = self.items_filtered_df(item_id)[self.uid_col].unique().tolist()

        items_dfs = [self.get_items_df_for_user(user) for user in users]

        # unite all data and get counts
        all_data = pd.concat(items_dfs)
        counts = all_data[self.item_id_col].value_counts()
        all_data = all_data.set_index(self.item_id_col).drop_duplicates()
        all_data['counts'] = counts
        all_data.reset_index(inplace=True)

        mapper = self.mapper_class(all_data)  # for view init

        # add maps for each user's history
        colors = iter(self.mapper_class.get_n_spaced_colors(len(items_dfs)))
        for df in items_dfs:
            color = next(colors)
            mapper.add_markers(df, color=color, size=default_marker_size)
            mapper.add_heatmap(df, color=color, sensitivity=1, opacity=0.4)

        # add common items
        common_items = all_data[all_data['counts'] > 1]
        sizes = list(map(int, np.sqrt(common_items['counts'].as_matrix()) + default_marker_size))
        mapper.add_markers(common_items, color='white', size=sizes)

        return mapper

    def map_cluster_labels(self, df_items=None, sample_n=1000):

        if df_items is None:
            df_items= self.df_items

        unique_labels = df_items[self.cluster_label_col].unique()

        items_dfs = [df_items[df_items[self.cluster_label_col] == label].sample(sample_n)
                        for label in unique_labels]

        all_data = pd.concat(items_dfs)

        mapper = self.mapper_class(all_data)

        colors = iter(self.mapper_class.get_n_spaced_colors(len(items_dfs)))

        for df in items_dfs:
            color = next(colors)
            mapper.add_heatmap(df, color=color, spread=50, sensitivity=3, opacity=0.3)

        return mapper

    def compare_similarity_results(self, item_id, items_lists, scores_lists=None, names=None, print_data=True):
        df_item_source = self.items_filtered_by_ids([item_id])
        items_dfs = [self.items_filtered_by_ids(l) for l in items_lists]

        # add variant and scores field
        if names is None:
            names = [str(i) for i in range(len(items_lists))]
        if scores_lists is None:
            scores_lists = [[5] * len(df) for df in items_dfs]
        all_lists = [df.assign(variant=name, score=scores[:len(df)])
                     for df, name, scores in
                     zip(items_dfs + [df_item_source],
                         names + ['source'], scores_lists + [[0]])]
        all_data = pd.concat(all_lists)

        # add counts
        all_data = all_data.join(
            all_data[self.item_id_col].
                value_counts().to_frame('count'),
            on=self.item_id_col)
        all_data = all_data.set_index(self.item_id_col)

        if print_data:
            logger.info('\n' + str(all_data))

        mapper = self.mapper_class(all_data)  # for view init

        if names is None:
            names = [''] * len(items_dfs)

        colors = self.mapper_class.get_n_spaced_colors(len(items_dfs))

        # poor man's legend
        logger.info("Poor man's legend: " + str(list(zip(colors, names))))

        for i in range(len(items_lists)):

            if scores_lists[i] is None:
                size = 5
            else:
                # filter simil scores only for those items that we have data for
                listings_with_data = items_dfs[i][self.item_id_col].values
                simil_scores = [score for listing_id, score in
                                zip(items_lists[i], scores_lists[i])
                                if listing_id in listings_with_data]
                size = 3 + (10 * np.array(simil_scores) ** 3).astype(np.int32)
                size = [int(el) for el in size]

            mapper.add_markers(items_dfs[i], size=size, color=colors[i])
            mapper.add_heatmap(items_dfs[i], sensitivity=1, opacity=0.4, spread=50, color=colors[i])

        mapper.add_markers(df_item_source, color='black', size=7)

        return mapper

    def map_similar_items(self, item_id, similar_items, similarity_scores=None):
        return self.compare_similarity_results(item_id, [similar_items], [similarity_scores])

    def map_recommendation_variants(
            self, train_items, test_items, recs_variants, colors, default_marker_size=5):
        df_train = self.items_filtered_by_ids(train_items)
        df_test = self.items_filtered_by_ids(test_items)
        rec_dfs = [self.items_filtered_by_ids(rec) for rec in recs_variants]

        all_data = pd.concat(rec_dfs + [df_train, df_test])
        all_data = all_data.set_index(self.item_id_col).drop_duplicates()

        mapper = self.mapper_class(all_data)  # for view init

        # add maps for each user's history
        for df, color in zip(rec_dfs, colors):
            mapper.add_markers(df, color=color, size=default_marker_size)
            mapper.add_heatmap(df, color=color, sensitivity=1, opacity=0.4, spread=50)

        if len(df_train):
            mapper.add_markers(df_train, color='gray', size=6)
            # mapper.add_heatmap(df_train, color='black', sensitivity=1, opacity=0.4)

        if len(df_test):
            mapper.add_markers(df_test, color='black', size=6)
            # mapper.add_heatmap(df_test, color=(250, 50, 50), sensitivity=1, opacity=0.4)

        return mapper