from copy import deepcopy

import numpy as np
import pandas as pd
from functools import partial

from lightfm import LightFM
import lightfm.lightfm

from ml_recsys_tools.data_handlers.interaction_handlers_base import RANDOM_STATE
from ml_recsys_tools.utils.automl import early_stopping_runner
from ml_recsys_tools.utils.instrumentation import simple_logger
from ml_recsys_tools.utils.parallelism import map_batches_multiproc, N_CPUS
from ml_recsys_tools.utils.similarity import most_similar, top_N_sorted
from ml_recsys_tools.recommenders.recommender_base import BaseDFSparseRecommender

# monkey patch print function
lightfm.lightfm.print = simple_logger.info


class LightFMRecommender(BaseDFSparseRecommender):

    default_fit_params = {
        'epochs': 100,
        'item_features': None,
        'num_threads': N_CPUS,
        'verbose': True,
    }

    default_model_params = {
        'loss': 'warp',
        'learning_schedule': 'adadelta',
        'no_components': 10,
        'max_sampled': 10,
        'item_alpha': 0,
        'user_alpha': 0,
    }

    def __init__(self,
                 use_sample_weight=False,
                 external_features=None,
                 external_features_params=None, **kwargs):
        self.use_sample_weight = use_sample_weight
        self.external_features = external_features
        self.external_features_params = external_features_params or {}
        self.cooc_mat = None
        super().__init__(**kwargs)

    def _prep_for_fit(self, train_obs, **fit_params):
        # assign all observation data
        self._set_data(train_obs)
        fit_params['sample_weight'] = self.train_mat.tocoo() \
            if self.use_sample_weight else None
        self._set_fit_params(fit_params)
        self._add_external_features()
        # init model and set params
        self.model = LightFM(**self.model_params)

    def _add_external_features(self):
        if self.external_features is not None:
            self.external_features_mat = self.external_features.\
                create_items_features_matrix(
                    items_encoder=self.sparse_mat_builder.iid_encoder,
                    **self.external_features_params)
            simple_logger.info('External item features matrix: %s' %
                            str(self.external_features_mat.shape))

        # add external features if specified
        self.fit_params['item_features'] = self.external_features_mat
        if self.external_features_mat is not None:
            simple_logger.info('Fitting using external features mat: %s'
                               % str(self.external_features_mat.shape))

    def fit(self, train_obs, **fit_params):
        self._prep_for_fit(
            train_obs, **fit_params)
        self.model.fit(self.train_mat, **self.fit_params)
        return self

    def fit_partial(self, train_obs, epochs=1):
        fit_params = self._dict_update(self.fit_params, {'epochs': epochs})
        if self.model is None:
            self.fit(train_obs, **fit_params)
        else:
            self.model.fit_partial(
                self.train_mat, **fit_params)
        return self

    def fit_with_early_stop(self, train_obs, valid_ratio=0.04, refit_on_all=False, metric='AUC',
                            epochs_start=0, epochs_max=200, epochs_step=10, stop_patience=10,
                            plot_convergence=True, decline_threshold=0.05, k=10):

        # split validation data
        sqrt_ratio = valid_ratio ** 0.5
        train_obs_internal, valid_obs = train_obs.split_train_test(
            users_ratio=sqrt_ratio, ratio=sqrt_ratio, random_state=RANDOM_STATE)

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

        max_epoch = early_stopping_runner(
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

        if not refit_on_all:
            simple_logger.info('Loading best model from checkpoint at %d epochs' % max_epoch)
            self.fit_params = self._dict_update(self.fit_params, {'epochs': max_epoch})
            self.model = self.model_checkpoint
            self.model_checkpoint = None
        else:
            # refit on whole data
            simple_logger.info('Refitting on whole train data for %d epochs' % max_epoch)
            self.fit(train_obs, epochs=max_epoch)

        return self

    def set_params(self, **params):
        """
        this is for skopt / sklearn compatibility
        """
        if 'epochs' in params:
            self._set_fit_params({'epochs': params.pop('epochs')})
        params = self._pop_set_params(
            params, ['use_sample_weight', 'external_features', 'external_features_params'])
        super().set_params(**params)

    def _get_item_representations(self, mode=None):

        n_items = len(self.sparse_mat_builder.iid_encoder.classes_)

        biases, representations = self.model.get_item_representations(self.fit_params['item_features'])

        if mode is None:
            pass  # default mode

        elif mode == 'external_features':
            external_features_mat = self.external_features_mat

            assert external_features_mat is not None, \
                'Must define and add a feature matrix for "external_features" similarity.'

            representations = external_features_mat

        elif (mode == 'no_features') and (self.fit_params['item_features'] is not None):

            simple_logger.info('LightFM recommender: get_similar_items: "no_features" mode '
                               'assumes ID mat was added and is the last part of the feature matrix.')

            assert self.model.item_embeddings.shape[0] > n_items, \
                'Either no ID matrix was added, or no features added'

            representations = self.model.item_embeddings[-n_items:, :]

        else:
            raise ValueError('Uknown representation mode: %s' % mode)

        return biases, representations

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

        biases, representations = self._get_item_representations(mode=embeddings_mode)

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

        user_biases, user_representations = self.model.get_user_representations()

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
            rename({self._item_col_simil: self._user_col})
        # this is UGLY, if this function is ever useful, fix this please (the renaming shortcut)

        if remove_self:
            simil_df = self._remove_self_similarities(
                simil_df, col1=self._user_col, col2=self._item_col)

        simil_df = self._recos_flat_to_lists(simil_df, n_cutoff=n_simil)

        return simil_df

    def _get_recommendations_flat(
            self, user_ids, n_rec, item_ids=None, exclude_training=True,
            pbar=None, item_features_mode=None, use_biases=True):

        user_biases, user_representations = self.model.get_user_representations()
        item_biases, item_representations = self._get_item_representations(mode=item_features_mode)

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

    def predict_on_df(self, df):
        mat_builder = self.get_prediction_mat_builder_adapter(self.sparse_mat_builder)
        df = mat_builder.remove_unseen_labels(df)
        df = mat_builder.add_encoded_cols(df)
        df[self._prediction_col] = self.model.predict(
            df[mat_builder.uid_col].values,
            df[mat_builder.iid_col].values,
            item_features=self.fit_params['item_features'],
            num_threads=self.fit_params['num_threads'])
        df.drop([mat_builder.uid_col, mat_builder.iid_col], axis=1, inplace=True)
        return df

    def eval_on_test_by_ranking_exact(self, test_dfs, test_names=('',),
                                      prefix='lfm ', include_train=True, k=10):

        @self.logging_decorator
        def _get_training_ranks():
            ranks_mat = self.model.predict_rank(
                self.train_mat,
                item_features=self.fit_params['item_features'],
                num_threads=self.fit_params['num_threads'])
            return ranks_mat, self.train_mat

        @self.logging_decorator
        def _get_test_ranks(test_df):
            test_sparse = self.sparse_mat_builder.build_sparse_interaction_matrix(test_df)
            ranks_mat = self.model.predict_rank(
                test_sparse, train_interactions=self.train_mat,
                item_features=self.fit_params['item_features'],
                num_threads=self.fit_params['num_threads'])
            return ranks_mat, test_sparse

        return self._eval_on_test_by_ranking_LFM(
            train_ranks_func=_get_training_ranks,
            test_tanks_func=_get_test_ranks,
            test_dfs=test_dfs,
            test_names=test_names,
            prefix=prefix,
            include_train=include_train)

    def get_recommendations_exact(
            self, user_ids, n_rec=10, exclude_training=True, chunksize=200, results_format='lists'):

        calc_func = partial(
            self._get_recommendations_exact,
            n_rec=n_rec,
            exclude_training=exclude_training,
            results_format=results_format)

        chunksize = int(35000 * chunksize / self.sparse_mat_builder.n_cols)

        ret = map_batches_multiproc(
            calc_func, user_ids, chunksize=chunksize, pbar='get_recommendations_exact_and_slow')
        return pd.concat(ret, axis=0)

    def _predict_for_users_dense(self, user_ids, exclude_training):

        mat_builder = self.sparse_mat_builder
        n_items = mat_builder.n_cols

        user_inds = mat_builder.uid_encoder.transform(user_ids)

        n_users = len(user_inds)
        user_inds_mat = user_inds.repeat(n_items)
        item_inds_mat = np.tile(np.arange(n_items), n_users)

        full_pred_mat = self.model.predict(
            user_inds_mat,
            item_inds_mat,
            item_features=self.fit_params['item_features'],
            num_threads=self.fit_params['num_threads']). \
            reshape((n_users, n_items))

        train_mat = self.train_mat.tocsr()

        if exclude_training:
            train_mat.sort_indices()
            for pred_ind, user_ind in enumerate(user_inds):
                train_inds = train_mat.indices[
                             train_mat.indptr[user_ind]: train_mat.indptr[user_ind + 1]]
                full_pred_mat[pred_ind, train_inds] = -np.inf

        return full_pred_mat

    def _get_recommendations_exact(self, user_ids, n_rec=10, exclude_training=True,
                                   results_format='lists'):

        full_pred_mat = self._predict_for_users_dense(user_ids, exclude_training=exclude_training)

        top_inds, top_scores = top_N_sorted(full_pred_mat, n=n_rec)

        item_ids = self.sparse_mat_builder.iid_encoder.inverse_transform(top_inds)

        return self._format_results_df(
            source_vec=user_ids, target_ids_mat=item_ids,
            scores_mat=top_scores, results_format='recommendations_' + results_format)
