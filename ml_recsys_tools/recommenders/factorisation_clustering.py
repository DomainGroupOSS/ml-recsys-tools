from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize

from ml_recsys_tools.data_handlers.interaction_handlers_base import RANDOM_STATE
from ml_recsys_tools.recommenders.factorization_base import BaseFactorizationRecommender
from ml_recsys_tools.utils.instrumentation import LogCallsTimeAndOutput
from ml_recsys_tools.utils.parallelism import N_CPUS
from ml_recsys_tools.utils.similarity import top_N_sorted


class FactorClusterMapper(LogCallsTimeAndOutput):

    def __init__(self, factoriser, obs_handler=None, n_clusters=20, **kwargs):
        super().__init__(**kwargs)
        self.factoriser = factoriser
        self.obs_handler = obs_handler
        self.n_clusters = n_clusters
        self.centers = None
        self.u_cluster_labels = None
        self.i_cluster_labels = None
        self.cluster_labels = None

    def cluster_factors(self, verbose=False):
        u_f = self.factoriser.user_factors_dataframe()
        i_f = self.factoriser.item_factors_dataframe()
        # u_f, user_centers = add_clusters(u_f, n_clusters, True)
        # i_f, item_centers = add_clusters(i_f, n_clusters, True)
        df_factors = pd.concat([u_f, i_f], sort=False)
        self._calc_clusters(df_factors, verbose=verbose)

        self.u_cluster_labels = self.cluster_labels[:len(u_f)]
        self.i_cluster_labels = self.cluster_labels[len(u_f):]

        self.center_neighbours, _ = top_N_sorted(
            -cosine_distances(self.centers), len(self.centers))

    def _calc_clusters(self, df_factors, verbose=False):
        # from sklearn import random_projection
        # transformer = random_projection.SparseRandomProjection()
        # X_new = transformer.fit_transform(X)

        # cls = DBSCAN(n_jobs=-1).fit(np.stack(reps_norm))

        # normalize representations
        reps_norm = normalize(df_factors.values, norm='l2', axis=1)

        cls = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=self.n_clusters * 10,
            max_iter=1000,
            verbose=verbose,
            reassignment_ratio=1/self.n_clusters,
            max_no_improvement=100,
            random_state=RANDOM_STATE)

        # cls = KMeans(
        #     n_clusters=self.n_clusters,
        #     max_iter=1000,
        #     verbose=verbose,
        #     n_jobs=-1,
        #     random_state=RANDOM_STATE)

        cls.fit(np.stack(reps_norm))

        self.cluster_labels = cls.labels_
        self.centers = cls.cluster_centers_
        return self

    def plot_cluster_counts(self):
        pd.Series(self.cluster_labels).value_counts().hist(bins=self.n_clusters, alpha=0.5)

    def clustered_items_details(self, n_cluster, n_sample=10):
        if self.obs_handler is not None:
            df_props = self.obs_handler.df_items
            return df_props[df_props.property_id.isin(self.items_for_cluster(n_cluster))].sample(n_sample)

    def clustered_users_items_sample(self, n_cluster, n_sample=10):
        if self.obs_handler is not None:
            df_obs = self.obs_handler.df_obs
            df_props = self.obs_handler.df_items
            items_for_clustered_users = df_obs[df_obs.userid.isin(self.users_for_cluster(n_cluster))].\
                drop_duplicates(subset=self.obs_handler.uid_col)[self.obs_handler.iid_col]
            return df_props[df_props.property_id.isin(items_for_clustered_users)].sample(n_sample)

    def users_for_cluster(self, n_cluster):
        return self.factoriser.user_ids(np.where(self.u_cluster_labels == n_cluster))

    def items_for_cluster(self, n_cluster):
        return self.factoriser.item_ids(np.where(self.i_cluster_labels == n_cluster))

    def clusters_for_users(self, user_ids):
        return self.u_cluster_labels[self.factoriser.user_inds(user_ids=user_ids)]

    def clusters_for_items(self, item_ids):
        return self.i_cluster_labels[self.factoriser.item_inds(item_ids=item_ids)]

    def cluster_neighbours(self, n_cluster, n_neighbours=1, include_self=True):
        ind_start = 0 if include_self else 1
        return self.center_neighbours[n_cluster, ind_start:(n_neighbours + 1)]


class ClusterRecommender(BaseFactorizationRecommender):

    def __init__(self, factoriser, n_clusters=30, neighbour_ratio=0.01,
                 verbose_clustering=False, **kwargs):
        super().__init__(**kwargs)
        self.__dict__.update(factoriser.__dict__)
        self.factoriser = factoriser
        self.n_clusters = n_clusters
        self.neighbour_ratio = neighbour_ratio
        self.cluster_mapper = None
        self.calc_clusters(verbose=verbose_clustering)

    def _predict_rank(self, *args, **kwargs): return self.factoriser._predict_rank(*args, **kwargs)
    def _predict_on_inds(self, *args, **kwargs): return self.factoriser._predict_on_inds(*args, **kwargs)
    def _set_epochs(self, *args, **kwargs): return self.factoriser._set_epochs(*args, **kwargs)
    def fit_partial(self, *args, **kwargs): return self.factoriser.fit_partial(*args, **kwargs)
    def fit(self, *args, **kwargs): return self.factoriser.fit(*args, **kwargs)
    def _prep_for_fit(self, *args, **kwargs): return self.factoriser._prep_for_fit(*args, **kwargs)
    def _get_user_factors(self, *args, **kwargs): return self.factoriser._get_user_factors(*args, **kwargs)
    def _get_item_factors(self, *args, **kwargs): return self.factoriser._get_item_factors(*args, **kwargs)

    def calc_clusters(self, verbose=False):
        self.cluster_mapper = FactorClusterMapper(
            factoriser=self.factoriser,
            n_clusters=self.n_clusters,
            n_neighbours=self.neighbour_ratio)
        self.cluster_mapper.cluster_factors(verbose=verbose)

    def get_recommendations(
            self, user_ids=None, item_ids=None, n_rec=10,
            exclusions=True, results_format='lists',
            **kwargs):

        if user_ids is not None:
            user_ids = self.remove_unseen_users(user_ids, message_prefix='get_recommendations: ')
        else:
            user_ids = self.all_users

        if item_ids is not None:
            item_ids = self.remove_unseen_items(item_ids, message_prefix='get_recommendations: ')
        else:
            item_ids = self.all_items

        u_clusters = self.cluster_mapper.clusters_for_users(user_ids)
        i_clusters = self.cluster_mapper.clusters_for_items(item_ids)

        def _get_cluster_recommendations(n_cluster):

            item_clusters = self.cluster_mapper.cluster_neighbours(
                n_cluster=n_cluster,
                n_neighbours=np.ceil(self.n_clusters * self.neighbour_ratio),
                include_self=True)

            return self._get_recommendations_flat(
                user_ids=user_ids[u_clusters == n_cluster],
                item_ids=item_ids[np.isin(i_clusters, item_clusters)],
                n_rec=n_rec,
                exclusions=exclusions)

        with ThreadPool(N_CPUS) as pool:
            flat_dfs = pool.map(_get_cluster_recommendations, np.unique(u_clusters))

        recos_flat = pd.concat(flat_dfs, sort=False)

        if results_format == 'flat':
            return recos_flat
        else:
            return self._recos_flat_to_lists(recos_flat, n_cutoff=n_rec)



