from builtins import staticmethod
from functools import partial
import numpy as np
import pandas as pd
import copy
from matplotlib import pyplot

from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.ensemble.voting_classifier import VotingClassifier

import skopt.callbacks
from skopt.plots import plot_convergence
from skopt.optimizer import forest_minimize, gp_minimize, gbrt_minimize, dummy_minimize
from skopt.utils import dimensions_aslist, point_asdict
from skopt.space import Real, Categorical, Integer

from ml_recsys_tools.utils.debug import print_time_and_shape
from ml_recsys_tools.utils.logger import simple_logger


# monkey patch print function
skopt.callbacks.print = simple_logger.info


class BayesSearchHoldOut:

    def __init__(self, search_space, pipeline, loss, random_ratio=0.5):
        self.search_space = search_space
        self.loss = loss
        self.pipeline = pipeline
        self.data_dict = None
        self.random_ratio = random_ratio

    class PrintIfBestCB:
        def __init__(self, search_inst):
            # self.prev_result = np.inf
            self.search_inst = search_inst

        def __call__(self, result):
            best_result = result.fun
            cur_result = result.func_vals[-1]
            if best_result >= cur_result:
                # self.prev_result = cur_result
                simple_logger.info('best params, iteration %d' % len(result.func_vals))
            simple_logger.info('params for loss=%f:' % cur_result)
            values = result.x_iters[-1]
            params = point_asdict(self.search_inst.search_space, values)
            simple_logger.info(params)


    def init_pipeline(self, params):
        params_dict = point_asdict(self.search_space, params)
        pipeline = copy.copy(self.pipeline)
        pipeline.set_params(**params_dict)
        return pipeline

    def objective_func(self, params):
        pipeline = self.init_pipeline(params)
        pipeline.fit(self.data_dict['x_train'], self.data_dict['y_train'])
        y_pred = pipeline.predict(self.data_dict['x_valid'])
        return self.loss(self.data_dict['y_valid'], y_pred)

    @print_time_and_shape
    def optimize(self, data_dict, n_calls, n_jobs=-1, optimizer='gb'):
        """
        example code:

        data_dict={ 'x_train': x_train_bo,
                        'y_train': y_train_bo,
                        'x_valid': x_val_bo,
                        'y_valid': y_val_bo,
                        }

        hp_space = {
                    'model': Categorical([LGBMClassifier()]),
                    'model__n_estimators': Integer(10, 50),
                    }

        pipe = Pipeline([
            ('vectorizer', TfidfVectorizer()),
            ('model', None)
        ])

        def custom_loss(y_true, y_pred):
            return 1-np.mean(recall_score(y_true, y_pred, average=None))

        bo = BayesSearchHoldOut(search_space=hp_space, pipeline=pipe, loss=custom_loss)
        res_bo, best_params, best_model = bo.optimize(data_dict=data_dict,n_calls=100)

        """

        if optimizer=='rf':
            opt_func = partial(forest_minimize,
                               # base_estimator="RF",
                               n_points=1000,
                               acq_func="EI")
        elif optimizer=='gb':
            opt_func = gbrt_minimize
        elif optimizer=='random':
            opt_func = dummy_minimize
        elif optimizer == 'gp': # too slow
            opt_func = gp_minimize
        else:
            raise NotImplementedError('unknown optimizer')

        self.data_dict = data_dict

        res_bo = opt_func(
            self.objective_func,
            dimensions_aslist(search_space=self.search_space),
            n_calls=n_calls,
            n_random_starts=int(n_calls * self.random_ratio),
            n_jobs=n_jobs,
            verbose=True,
            callback=self.PrintIfBestCB(self)
        )

        best_values = res_bo.x
        best_params = point_asdict(self.search_space, best_values)
        best_model = self.init_pipeline(best_values)
        plot_convergence(res_bo)
        pyplot.plot(res_bo.func_vals)

        return res_bo, best_params, best_model

    def best_results_summary(self, res_bo, percentile=95):
        return self.best_results_report(
            res_bo, percentile=percentile, search_space=self.search_space)

    @staticmethod
    def best_results_report(res_bo, percentile, search_space):
        cut_off = np.percentile(res_bo.func_vals, 100 - percentile)
        best_x = [x + [res_bo.func_vals[i]] for
                    i, x in enumerate(res_bo.x_iters)
                        if res_bo.func_vals[i] <= cut_off]
        return pd.DataFrame(best_x,
                            columns=sorted(search_space.keys()) + ['target_loss']).\
            sort_values('target_loss')


class VotingEnsemble(VotingClassifier):
    '''
    Adds caching for probabilities to the VotingClassifer to enable quick predictions with updated weights
    (functools lru_cache is no good because it expects immutable inputs)
    '''
    def __init__(self, estimators, voting='hard', weights=None, n_jobs=1,
                 flatten_transform=None):
        super().__init__(estimators, voting, weights, n_jobs, flatten_transform)
        self._proba_cache = None
        self._x_cache = None

    def _collect_probas(self, X):
        if (self._proba_cache is None) or (self._x_cache is None) or \
                (not np.asarray(list(self._x_cache) == list(X)).all()):
            self._proba_cache = VotingClassifier._collect_probas(self, X)
            self._x_cache = X
        return self._proba_cache


class EnsembleTrainer:
    def __init__(self, models_losses, models_params, cutoff_ratio, pipeline, data_dict):
        '''

        example usage:

        # res_bo is result of optimisation run (sklearn format)
        res_y = res_bo.func_vals
        res_x = res_bo.x_iters
        # hp_space is hyper_params space,
        # params are dictionaries of params for initialization of the pipeline
        params = [point_asdict(hp_space, x) for x in res_x]

        ens_selection = EnsembleTrainer(
            models_losses=res_y,
            models_params=params,
            cutoff_ratio=0.2,
            pipeline=pipe,
            data_dict=data_dict
        )

        # than weights can be optimizid in two ways - forward-backward, or bayesian optimisation
        ens, res_ens = ens_selection.optimize_weights_by_skopt(n_iter=1000)
        ens, traj = ens_selection.optimize_weights_by_forward_selection(n_iter=1000)

        '''
        self.data_dict = data_dict
        self.pipeline = pipeline
        self.cutoff_ratio = cutoff_ratio
        self.models_losses = models_losses
        self.models_params = models_params

        self._candidates = self._select_candidates()
        self.ensemble = self._fit_initial_ensemble()

    def _select_candidates(self):
        # sort
        idx = np.argsort(self.models_losses)
        all_results = np.array(self.models_losses)[idx]
        all_params = np.array(self.models_params)[idx]

        # choose ensemble candidates
        cutoff = min(np.percentile(all_results, int(self.cutoff_ratio * 100)),
                     (1 - (1 - self.cutoff_ratio) * (1 - np.min(all_results))))

        # choose the params that satisfy the cutoff
        ens_params = [all_params[i] for i in range(len(all_params))
                      if (all_results[i] <= cutoff)]

        # instantiate the models
        candidates = [copy.deepcopy(self.pipeline.set_params(**ens_params[i]))
                    for i in range(len(ens_params))]

        return candidates

    def _fit_initial_ensemble(self):
        self._select_candidates()

        # define and fit initial ensemeble
        ensemble = VotingEnsemble([(str(i), self._candidates[i])
                              for i in range(len(self._candidates))], voting='soft')

        ensemble.fit(self.data_dict['x_train'], self.data_dict['y_train'])

        return ensemble

    def _loss_function(self, weights):
        self.ensemble.weights = weights
        return 1 - np.mean(
            recall_score(self.data_dict['y_valid'],
                         self.ensemble.predict(self.data_dict['x_valid']),
                         average=None))


    def _initial_guess(self):
        return list(np.array([1] + list(np.zeros(len(self._candidates) - 1))))  # top model

    def optimize_weights_by_skopt(self, n_iter=100):
        # fit ensemble weights by bayesian optimisation (mostly random search though

        res_ens = forest_minimize(
            self._loss_function,
            dimensions=[Real(0.0, 1.0)] * len(self._candidates),
            n_calls=n_iter,
            x0=self._initial_guess(),
            n_random_starts=int(n_iter * 0.5),
            base_estimator="RF",
            n_points=1000
            )

        plot_convergence(res_ens)

        self.ensemble.weights = res_ens.x

        return self.ensemble, res_ens

    def optimize_weights_by_forward_selection(self, n_iter=1000):

        n_models = len(self._candidates)

        cur_weights = list(np.zeros(n_models))
        cur_loss = 1.0
        trajectory = [(cur_loss, 0, 0, 'f')]
        loss_calcs = 0

        # forward
        while loss_calcs <= n_iter:
            new_losses = []
            for ens_i in range(n_models):
                new_weights = cur_weights.copy()
                new_weights[ens_i] += 1
                new_losses.append(self._loss_function(new_weights))
                loss_calcs += 1

            idx_add = int(np.argmin(new_losses))
            new_loss = min(new_losses)

            if new_loss <= cur_loss:
                cur_weights[idx_add] += 1
                cur_loss = new_loss.copy()
                trajectory.append((new_loss, idx_add, loss_calcs, 'f'))

        # backward
        still_pruning = True
        while still_pruning:
            still_pruning = False
            for ens_i in range(n_models):
                new_weights = cur_weights.copy()
                new_weights[ens_i] = 0
                if (sum(new_weights) == 0.0):
                    break

                new_loss = self._loss_function(new_weights)
                loss_calcs += 1

                if new_loss <= cur_loss:
                    cur_weights = new_weights
                    cur_loss = new_loss.copy()
                    trajectory.append((cur_loss, ens_i, loss_calcs, 'b'))
                    still_pruning = True

        self.ensemble.weights = cur_weights
        return self.ensemble, trajectory




# def train_ensemble_by_skopt(models_losses, models_params, cutoff_ratio,
#                             n_iter, pipeline, data_dict):
#     '''
#     trains a voting ensemble by creating one out of top models
#     and than optimizing its weights on a validation set
#
#     '''
#     def ensemble_objective_func(x, *args):
#         ens.weights = x
#         return 1 - np.mean(
#             recall_score(data_dict['y_valid'],
#                          ens.predict(data_dict['x_valid']),
#                          average=None))
#
#     # sort
#     idx = np.argsort(models_losses)
#     all_results = np.array(models_losses)[idx]
#     all_params = np.array(models_params)[idx]
#
#     # choose ensemble candidates
#     cutoff = min(np.percentile(all_results, int(cutoff_ratio * 100)),
#                  cutoff_ratio * (1 - np.min(all_results)))
#
#     # choose the params that satisfy the cutoff
#     ens_params = [all_params[i] for i in range(len(all_params))
#                   if (all_results[i] <= cutoff)]
#
#     # instantiate the models
#     ens_ests = [copy.deepcopy(pipeline.set_params(**ens_params[i]))
#                 for i in range(len(ens_params))]
#
#     # define and fit initial ensemeble
#     ens = VotingEnsemble([(str(i), ens_ests[i])
#                           for i in range(len(ens_ests))], voting='soft')
#
#     ens.fit(data_dict['x_train'], data_dict['y_train'])
#
#     # fit ensemble weights by bayesian optimisation (mostly random search though
#     initial_guess = np.array([1] + list(np.zeros(len(ens_ests) - 1)))  # top model
#     res_ens = forest_minimize(ensemble_objective_func,
#                               dimensions=[Real(0.0, 1.0)] * len(ens_ests),
#                               n_calls=n_iter,
#                               x0=list(initial_guess),
#                               n_random_starts=int(n_iter * 0.5),
#                               )
#
#     plot_convergence(res_ens)
#
#     ens.weights = res_ens.x
#
#     return ens, res_ens
