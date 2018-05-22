import os
from abc import abstractmethod
from copy import deepcopy
from functools import partial

import pandas as pd
import numpy as np

from ml_recsys_tools.data_handlers.interaction_handlers_base import RANDOM_STATE
from ml_recsys_tools.recommenders.recommender_base import BaseDFSparseRecommender
from ml_recsys_tools.utils.automl import early_stopping_runner
from ml_recsys_tools.utils.logger import simple_logger
from ml_recsys_tools.utils.parallelism import map_batches_multiproc
from ml_recsys_tools.utils.similarity import most_similar, top_N_sorted


class BaseFactorizationRecommender(BaseDFSparseRecommender):

    @abstractmethod
    def _get_item_factors(self, mode=None):
        return  np.array([0]), np.array([0])

    @abstractmethod
    def _get_user_factors(self, mode=None):
        return np.array([0]), np.array([0])

    @abstractmethod
    def _prep_for_fit(self, train_obs, **fit_params):
        pass

    @abstractmethod
    def fit_partial(self, train_obs, **fit_params):
        pass

    @abstractmethod
    def _set_epochs(self, epochs):
        pass

    @abstractmethod
    def _predict(self, user_ids, item_ids):
        pass

    @abstractmethod
    def _predict_rank(self, test_mat, train_mat=None):
        pass

    def fit_with_early_stop(self, train_obs, valid_ratio=0.04, refit_on_all=False, metric='AUC',
                            epochs_start=0, epochs_max=200, epochs_step=10, stop_patience=10,
                            plot_convergence=True, decline_threshold=0.05, k=10, valid_split_time_col=None):

        # split validation data
        train_obs_internal, valid_obs = train_obs.split_train_test(
            ratio=valid_ratio ** 0.5 if valid_split_time_col is None else valid_ratio,
            users_ratio=valid_ratio ** 0.5 if valid_split_time_col is None else 1,
            time_split_column=valid_split_time_col,
            random_state=RANDOM_STATE)

        self.model = None
        self.model_checkpoint = None
        all_metrics = pd.DataFrame()

        def update_full_metrics_df(cur_epoch, report_df):
            nonlocal all_metrics
            all_metrics = all_metrics.append(
                report_df.rename(index={'test': cur_epoch}))

        def check_point_func():
            if not refit_on_all:
                self.model_checkpoint = deepcopy(self.model)

        def score_func(cur_epoch, step):
            self.fit_partial(train_obs_internal, epochs=step)
            lfm_report = self.eval_on_test_by_ranking(
                valid_obs.df_obs, include_train=False, prefix='', k=k)
            cur_score = lfm_report.loc['test', metric]
            update_full_metrics_df(cur_epoch, lfm_report)
            return cur_score

        best_epoch = early_stopping_runner(
            score_func=score_func,
            check_point_func=check_point_func,
            epochs_start=epochs_start,
            epochs_max=epochs_max,
            epochs_step=epochs_step,
            stop_patience=stop_patience,
            decline_threshold=decline_threshold,
            plot_graph=plot_convergence
        )
        simple_logger.info('Early stop, all_metrics:\n' + str(all_metrics))

        if plot_convergence:
            all_metrics = all_metrics.divide(all_metrics.max())
            all_metrics.plot()
        self.early_stop_metrics_df = all_metrics

        self._set_epochs(epochs=best_epoch)

        if refit_on_all:
            simple_logger.info('Refitting on whole train data for %d epochs' % best_epoch)
            self.fit(train_obs)
        else:
            simple_logger.info('Loading best model from checkpoint at %d epochs' % best_epoch)
            self.model, self.model_checkpoint = self.model_checkpoint, None

        return self


    def get_similar_items(self, item_ids=None, target_item_ids=None, n_simil=10,
                          remove_self=True, embeddings_mode=None,
                          simil_mode='cosine', results_format='lists', pbar=None):
        """
        uses learned embeddings to get N most similar items

        :param item_ids: vector of item IDs
        :param n_simil: number of most similar items to retrieve
        :param remove_self: whether to remove the the query items from the lists (similarity to self should be maximal)
        :param embeddings_mode: the item representations to use for calculation:
             None (default) - means full representations
             'external_features' - calculation based only external features (assumes those exist)
             'no_features' - calculation based only on internal features (assumed identity mat was part of the features)
        :param simil_mode: mode of similairyt calculation:
            'cosine' (default) - cosine similarity bewtween representations (normalized dot product with no biases)
            'dot' - unnormalized dot product with addition of biases
            'euclidean' - inverse of euclidean distance
            'cooccurance' - no usage of learned features - just cooccurence of items matrix
                (number of 2nd degree connections in user-item graph)
        :param results_format:
            'flat' for dataframe of triplets (source_item, similar_item, similarity)
            'lists' for dataframe of lists (source_item, list of similar items, list of similarity scores)
        :param pbar: name of tqdm progress bar (None means no tqdm)

        :return: a matrix of most similar IDs [n_ids, N], a matrix of score of those similarities [n_ids, N]
        """

        item_ids, target_item_ids = self._check_item_ids_args(item_ids, target_item_ids)

        biases, representations = self._get_item_factors(mode=embeddings_mode)

        best_ids, best_scores = most_similar(
            source_ids=item_ids,
            target_ids=target_item_ids,
            source_encoder=self.sparse_mat_builder.iid_encoder,
            source_mat=representations,
            source_biases=biases,
            n=n_simil+1 if remove_self else n_simil,
            simil_mode=simil_mode,
            pbar=pbar
        )

        simil_df = self._format_results_df(
            item_ids, target_ids_mat=best_ids,
            scores_mat=best_scores, results_format='similarities_flat')

        if remove_self:
            simil_df = self._remove_self_similarities(
                simil_df, col1=self._item_col_simil, col2=self._item_col)

        if 'lists' in results_format:
            simil_df = self._simil_flat_to_lists(simil_df, n_cutoff=n_simil)

        return simil_df

    def get_similar_users(self, user_ids=None, target_user_ids=None, n_simil=10, remove_self=True,
                          simil_mode='cosine', pbar=None):
        """
        same as get_similar_items but for users
        """
        user_ids, target_user_ids = self._check_user_ids_args(user_ids, target_user_ids)

        user_biases, user_representations = self._get_user_factors()

        best_ids, best_scores = most_similar(
            source_ids=user_ids,
            target_ids=target_user_ids,
            source_encoder=self.sparse_mat_builder.uid_encoder,
            source_mat=user_representations,
            source_biases=user_biases,
            n=n_simil,
            simil_mode=simil_mode,
            pbar=pbar
        )

        simil_df = self._format_results_df(
            user_ids, target_ids_mat=best_ids,
            scores_mat=best_scores, results_format='similarities_flat'). \
            rename({self._item_col_simil: self._user_col}, axis=1)
        # this is UGLY, if this function is ever useful, fix this please (the renaming shortcut)

        if remove_self:
            simil_df = self._remove_self_similarities(
                simil_df, col1=self._user_col, col2=self._item_col)

        simil_df = self._recos_flat_to_lists(simil_df, n_cutoff=n_simil)

        return simil_df

    def _get_recommendations_flat(
            self, user_ids, n_rec, item_ids=None, exclude_training=True,
            pbar=None, item_features_mode=None, use_biases=True):

        user_biases, user_representations = self._get_user_factors()
        item_biases, item_representations = self._get_item_factors(mode=item_features_mode)

        if not use_biases:
            user_biases, item_biases = None, None

        best_ids, best_scores = most_similar(
            source_ids=user_ids,
            target_ids=item_ids,
            source_encoder=self.sparse_mat_builder.uid_encoder,
            target_encoder=self.sparse_mat_builder.iid_encoder,
            source_mat=user_representations,
            target_mat=item_representations,
            source_biases=user_biases,
            target_biases=item_biases,
            exclude_mat_sp=self.train_mat if exclude_training else None,
            n=n_rec,
            simil_mode='dot',
            pbar=pbar
        )

        return self._format_results_df(
            source_vec=user_ids, target_ids_mat=best_ids, scores_mat=best_scores,
            results_format='recommendations_flat')

    def predict_on_df(self, df, exclude_training=True, user_col=None, item_col=None):
        if user_col is not None and user_col!=self.sparse_mat_builder.uid_source_col:
            df[self.sparse_mat_builder.uid_source_col] = df[user_col]
        if item_col is not None and item_col!=self.sparse_mat_builder.iid_source_col:
            df[self.sparse_mat_builder.iid_source_col] = df[item_col]

        mat_builder = self.sparse_mat_builder
        df = mat_builder.remove_unseen_labels(df)
        df = mat_builder.add_encoded_cols(df, parallel=False)
        df[self._prediction_col] = self._predict(
            df[mat_builder.uid_col].values, df[mat_builder.iid_col].values)

        if exclude_training:
            exclude_mat_sp_coo = \
                self.train_mat[df[mat_builder.uid_col].values, :] \
                    [:, df[mat_builder.iid_col].values].tocoo()
            df[df[mat_builder.uid_col].isin(exclude_mat_sp_coo.row) &
               df[mat_builder.iid_col].isin(exclude_mat_sp_coo.col)][self._prediction_col] = -np.inf

        df.drop([mat_builder.uid_col, mat_builder.iid_col], axis=1, inplace=True)
        return df

    def eval_on_test_by_ranking_exact(self, test_dfs, test_names=('',),
                                      prefix='lfm ', include_train=True, k=10):

        @self.logging_decorator
        def _get_training_ranks():
            ranks_mat = self._predict_rank(self.train_mat)
            return ranks_mat, self.train_mat

        @self.logging_decorator
        def _get_test_ranks(test_df):
            test_sparse = self.sparse_mat_builder.build_sparse_interaction_matrix(test_df)
            ranks_mat = self.model.predict_rank(test_sparse, self.train_mat)
            return ranks_mat, test_sparse

        return self._eval_on_test_by_ranking_LFM(
            train_ranks_func=_get_training_ranks,
            test_tanks_func=_get_test_ranks,
            test_dfs=test_dfs,
            test_names=test_names,
            prefix=prefix,
            include_train=include_train,
            k=k)

    def get_recommendations_exact(
            self, user_ids, item_ids=None, n_rec=10, exclude_training=True, chunksize=200, results_format='lists'):

        calc_func = partial(
            self._get_recommendations_exact,
            n_rec=n_rec,
            item_ids=item_ids,
            exclude_training=exclude_training,
            results_format=results_format)

        chunksize = int(35000 * chunksize / self.sparse_mat_builder.n_cols)

        ret = map_batches_multiproc(calc_func, user_ids, chunksize=chunksize)
        return pd.concat(ret, axis=0)

    def _get_recommendations_exact(self, user_ids, item_ids=None, n_rec=10, exclude_training=True,
                                   results_format='lists'):

        full_pred_mat = self._predict_for_users_dense(user_ids, item_ids, exclude_training=exclude_training)

        top_inds, top_scores = top_N_sorted(full_pred_mat, n=n_rec)

        best_ids = self.sparse_mat_builder.iid_encoder.inverse_transform(top_inds)

        return self._format_results_df(
            source_vec=user_ids, target_ids_mat=best_ids,
            scores_mat=top_scores, results_format='recommendations_' + results_format)

    def _predict_for_users_dense(self, user_ids, item_ids=None, exclude_training=True):

        if item_ids is None:
            item_ids = self.sparse_mat_builder.iid_encoder.classes_

        user_inds = self.sparse_mat_builder.uid_encoder.transform(user_ids)
        item_inds = self.sparse_mat_builder.iid_encoder.transform(item_ids)

        n_users = len(user_inds)
        n_items = len(item_inds)
        user_inds_mat = user_inds.repeat(n_items)
        item_inds_mat = np.tile(item_inds, n_users)

        full_pred_mat = self._predict(user_inds_mat, item_inds_mat).reshape((n_users, n_items))

        if exclude_training:
            exclude_mat_sp_coo = self.train_mat[user_inds, :][:, item_inds].tocoo()
            full_pred_mat[exclude_mat_sp_coo.row, exclude_mat_sp_coo.col] = -np.inf

        return full_pred_mat

    def predict_for_user(self, user_id, item_ids, rank_training_last=True):
        df = pd.DataFrame()
        df[self.sparse_mat_builder.iid_source_col] = item_ids  # assigning first because determines length
        df[self.sparse_mat_builder.uid_source_col] = user_id
        df[self._prediction_col] = None

        if user_id not in self.sparse_mat_builder.uid_encoder.classes_:
            return df

        new_mask = self.sparse_mat_builder.iid_encoder.find_new_labels(item_ids)

        preds = self._predict_for_users_dense(
            user_ids=[user_id], item_ids=item_ids[~new_mask], exclude_training=rank_training_last)

        df[self._prediction_col].values[~new_mask] = preds.ravel()

        return df

