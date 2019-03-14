import copy

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import scipy.sparse as sp

from sklearn.preprocessing import LabelBinarizer, normalize, LabelEncoder
from sklearn_pandas import DataFrameMapper

from ml_recsys_tools.utils.sklearn_extenstions import NumericBinningBinarizer

from ml_recsys_tools.data_handlers.interaction_handlers_base import ObservationsDF, RANDOM_STATE
from ml_recsys_tools.utils.instrumentation import LogCallsTimeAndOutput
from ml_recsys_tools.utils.logger import simple_logger as logger


class ExternalFeaturesDF(LogCallsTimeAndOutput):
    """
    handles external items features and feature engineering
    """

    _numeric_duplicate_suffix = '_num'

    def __init__(self, feat_df, id_col, num_cols=None, cat_cols=None, verbose=True):
        super().__init__(verbose=verbose)
        self.feat_df = feat_df.copy()
        self.id_col = id_col
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self._numeric_duplicate_cols = None
        self._feat_weight = None
        self.df_transformer = None
        if self.num_cols is None and self.cat_cols is None:
            self.infer_categorical_numerical_columns()

    def infer_categorical_numerical_columns(self,
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
                n_unique = self.feat_df[col].nunique()
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

    def _check_intersecting_num_and_cat_columns(self):
        self._numeric_duplicate_cols = list(set(self.cat_cols).intersection(set(self.num_cols)))
        if len(self._numeric_duplicate_cols):
            for col in self._numeric_duplicate_cols:
                alt_name = col + self._numeric_duplicate_suffix
                self.feat_df[alt_name] = self.feat_df[col].copy()
                self.num_cols.remove(col)
                self.num_cols.append(alt_name)

    def fit_transform_ids_df_to_mat(self,
                                    items_encoder,
                                    mode='binarize',
                                    normalize_output=False,
                                    add_identity_mat=False,
                                    numeric_n_bins=128,
                                    feat_weight=1.0):
        """
        creates a sparse feature matrix from item features

        :param items_encoder: the encoder that is used to filter and
            align the features dataframe to the sparse matrices
        :param mode: 'binarize' or 'encode'.
            'binarize' (default) - creates a binary matrix by label binarizing
            categorical and numeric feature.
            'encode' - only encodes the categorical features to integers and leaves numeric as is
        :param add_identity_mat: indicator whether to add a sparse identity matrix
            (as used when no features are used), as per LightFM's docs suggestion
        :param normalize_output:
            None (default) - no normalization
            'rows' - normalize rows with l1 norm
            anything else - normalize cols with l1 norm
        :param numeric_n_bins: number of bins for binning numeric features
        :param feat_weight:
            feature weight relative to identity matrix (can be used to emphasize one or the other)
            can also be a dictionary of weights to be applied to columns e.g. {'column_name': 10}

        :return: sparse feature mat n_items x n_features
        """

        self._check_intersecting_num_and_cat_columns()

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
            self.df_transformer = self.init_df_transformer(
                mode=mode,
                categorical_feat_cols=self.cat_cols,
                numeric_feat_cols=self.num_cols,
                numeric_n_bins=numeric_n_bins)

            feat_mat = self.df_transformer.fit_transform(full_feat_df)

            if sp.issparse(feat_mat):
                feat_mat.eliminate_zeros()

            # weight the features before adding the identity mat
            self._feat_weight = feat_weight
            feat_mat = self._apply_weights_to_matrix(feat_mat)

            # normalize each row
            if normalize_output:
                axis = int(normalize_output == 'rows')
                feat_mat = normalize(feat_mat, norm='l1', axis=axis, copy=False)

            if add_identity_mat:
                # identity mat
                id_mat = sp.identity(n_items, dtype=np.float32, format='csr')

                assert sp.issparse(feat_mat), 'Trying to add identity mat to non-sparse matrix'

                full_feat_mat = self.concatenate_csc_matrices_by_columns(
                    feat_mat.tocsc(), id_mat.tocsc()).tocsr()
            else:
                full_feat_mat = feat_mat

            return full_feat_mat

        else:
            return None

    def transform_df_to_mat(self, feat_df):
        if self._numeric_duplicate_cols is not None:
            for col in self._numeric_duplicate_cols:
                feat_df[col + self._numeric_duplicate_suffix] = feat_df[col].copy()
        feat_df[self.cat_cols] = feat_df[self.cat_cols].astype(str)
        trans_mat = self.df_transformer.transform(feat_df)
        return self._apply_weights_to_matrix(trans_mat)

    def _apply_weights_to_matrix(self, feat_mat):
        if np.isscalar(self._feat_weight):
            feat_mat = feat_mat.astype(np.float32) * self._feat_weight
        elif isinstance(self._feat_weight, dict):
            for col, weight in self._feat_weight.items():
                cols_mask = np.core.defchararray.startswith(
                    self.df_transformer.transformed_names_, col)
                feat_mat[:, cols_mask] *= weight
        else:
            raise ValueError('Uknown feature weight format.')
        return feat_mat

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
    def init_df_transformer(mode, categorical_feat_cols, numeric_feat_cols,
                            numeric_n_bins=64):
        if mode=='binarize':
            feat_mapper = DataFrameMapper(
                [(cat_col,
                  LabelBinarizer(sparse_output=True))
                 for cat_col in categorical_feat_cols] +
                [(num_col,
                  NumericBinningBinarizer(n_bins=numeric_n_bins, sparse_output=True))
                  # GrayCodesNumericBinarizer(n_bins=numeric_n_bins, sparse_output=True))
                 for num_col in numeric_feat_cols],
                sparse=True
            )
        elif mode=='encode':
            feat_mapper = DataFrameMapper(
                [(cat_col,
                  LabelEncoder())
                 for cat_col in categorical_feat_cols],
                default = None  # pass other columns as is
            )
        else:
            raise NotImplementedError('Unknown transform mode')
        return feat_mapper


class ItemsHandler(LogCallsTimeAndOutput):

    def __init__(self, df_items, item_id_col='item_id', **kwargs):
        super().__init__(**kwargs)
        self.item_id_col = item_id_col
        self.df_items = self._preprocess_items_df(df_items)

    def _preprocess_items_df(self, df_items):
        # make sure the ID col is of object type
        df_items[self.item_id_col] = df_items[self.item_id_col].astype(str)
        df_items.drop_duplicates(self.item_id_col, inplace=True)
        return df_items

    def __add__(self, other):
        self.df_items = pd.concat([self.df_items, other.df_items], sort=False)
        self.df_items.drop_duplicates(self.item_id_col, inplace=True)
        return self

    def __repr__(self):
        return super().__repr__() + ', %d Items' % len(self.df_items)

    def items_filtered_by_ids(self, item_ids):
        return self.df_items[self.df_items[self.item_id_col].isin(item_ids)]

    def get_item_features(self,
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


class ObsWithFeatures(ObservationsDF, ItemsHandler):

    def __init__(self, df_obs, df_items, item_id_col='item_id', **kwargs):
        super().__init__(df_obs=df_obs, df_items=df_items, item_id_col=item_id_col, **kwargs)
        self.cluster_label_col = 'cluster_label'
        self._filter_relevant_obs_and_items(stage='init')

    def _filter_relevant_obs_and_items(self, stage=''):
        items_ids = self.df_items[self.item_id_col].unique().astype(str)
        obs_ids = self.df_obs[self.iid_col].unique().astype(str)

        obs_filt = self.df_obs[self.iid_col].astype(str).isin(items_ids)
        item_filt = self.df_items[self.item_id_col].astype(str).isin(obs_ids)

        self.df_obs = self.df_obs[obs_filt].copy()
        self.df_items = self.df_items[item_filt].copy()

        n_dropped_obs = (~obs_filt).sum()
        n_dropped_items = (~item_filt).sum()
        if n_dropped_obs + n_dropped_items:
            logger.info('ObsWithFeatures:_filter_relevant_obs_and_items:%s '
                        'dropped %d observations, %d items' % (stage, n_dropped_obs, n_dropped_items))

    def filter_by_cluster_label(self, label):
        assert self.cluster_label_col in self.df_items.columns
        other = copy.deepcopy(self)
        other.df_items = self.df_items[self.df_items[self.cluster_label_col] == label].copy()
        other._filter_relevant_obs_and_items(stage='filter_by_cluster_label')
        return other

    def _apply_filter(self, mask_filter):
        other = copy.deepcopy(self)
        other.df_items = self.df_items[mask_filter].copy()
        other._filter_relevant_obs_and_items(stage='_apply_filter')
        return other

    def sample_observations(self,
                            n_users=None,
                            n_items=None,
                            method='random',
                            min_user_hist=0,
                            min_item_hist=0,
                            users_to_keep=(),
                            items_to_keep=(),
                            random_state=None):
        sample_df = super().sample_observations(n_users=n_users,
                                                n_items=n_items,
                                                method=method,
                                                min_user_hist=min_user_hist,
                                                min_item_hist=min_item_hist,
                                                users_to_keep=users_to_keep,
                                                items_to_keep=items_to_keep,
                                                random_state=random_state)
        other = copy.deepcopy(self)
        other.df_obs = sample_df.df_obs
        other._filter_relevant_obs_and_items(stage='sample_observations')
        return other

    def filter_columns_by_df(self, other_df_obs):
        """
        removes users or items that are not in the other user dataframe
        :param other_df_obs: other dataframe, that has the same structure (column names)
        :return: new observation handler
        """
        other = super().filter_columns_by_df(other_df_obs)
        other._filter_relevant_obs_and_items(stage='filter_columns_by_df')
        return other

    def filter_interactions_by_df(self, other_df_obs, mode):
        other = super().filter_interactions_by_df(other_df_obs, mode)
        other._filter_relevant_obs_and_items(stage='remove_interactions_by_df')
        return other

    def get_items_df_for_user(self, user):
        item_ids = self.users_filtered_df_obs(user)[self.iid_col].unique().tolist()
        df_items_view = self.items_filtered_by_ids(item_ids)
        return df_items_view



class ObsWithGeoFeatures(ObsWithFeatures):

    def __init__(self, df_obs, df_items, lat_col='lat', long_col='long',
                 remove_nans=False, **kwargs):
        super().__init__(df_obs=df_obs, df_items=df_items, **kwargs)
        self.lat_col = lat_col
        self.long_col = long_col
        self.remove_nans = remove_nans
        self.df_items = self._preprocess_geo_cols(self.df_items)

    @property
    def geo_cols(self):
        return [self.lat_col, self.long_col]

    def _preprocess_geo_cols(self, df):
        if self.remove_nans:
            # remove nans
            filt_nan = df[self.lat_col].notnull() & \
                       ~df[self.lat_col].isin(['None']) & \
                       ~df[self.lat_col].isin(['nan'])
            df = df[filt_nan].copy()

        # to float
        df[self.lat_col] = df[self.lat_col].astype(float)
        df[self.long_col] = df[self.long_col].astype(float)

        return df

    def filter_by_location_range(self, min_lat, max_lat, min_long, max_long):
        """ e.g.
        min_lat = -33.851674299999999
        max_lat = -33.767248700000003
        min_long = 151.09386979999999
        max_long = 151.31997699999999
        """

        geo_filt = (self.df_items[self.lat_col].values <= max_lat) & \
                   (self.df_items[self.lat_col].values >= min_lat) & \
                   (self.df_items[self.long_col].values <= max_long) & \
                   (self.df_items[self.long_col].values >= min_long)

        return self._apply_filter(geo_filt)

    def filter_by_location_circle(self, center_lat, center_long, degree_dist):

        geo_filt = ((self.df_items[self.lat_col].values - center_lat) ** 2 +
                    (self.df_items[self.long_col].values - center_long) ** 2) <= degree_dist

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
            fit(self.df_items[self.geo_cols].values)

        self.df_items[self.cluster_label_col] = cls.labels_

    def calcluate_equidense_geo_grid(self, n_lat, n_long, overlap_margin, geo_box):

        df_items = self.filter_by_location_range(**geo_box).df_items

        geo_filters = []
        lat_bins = np.percentile(df_items.lat, np.linspace(0, 100, n_lat + 1))

        for i in range(n_lat):
            cur_lat_bins = lat_bins[i:(i + 2)]

            prop_slice = df_items[(df_items.lat.values <= cur_lat_bins[1]) &
                                  (df_items.lat.values >= cur_lat_bins[0])]

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



