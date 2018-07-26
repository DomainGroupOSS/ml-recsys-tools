import os
import pprint
from functools import partial
from multiprocessing import Queue, Process
from threading import Thread
from types import SimpleNamespace

import numpy as np
import pandas as pd
import copy

import time
from matplotlib import pyplot
from sklearn.feature_selection import mutual_info_regression

from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.ensemble.voting_classifier import VotingClassifier

from sklearn.preprocessing import LabelEncoder
from skopt import Optimizer
from skopt.callbacks import VerboseCallback
from skopt.learning import RandomForestRegressor
from skopt.plots import plot_convergence
from skopt.optimizer import forest_minimize, gp_minimize, gbrt_minimize, dummy_minimize
from skopt.utils import dimensions_aslist, point_asdict
import skopt.space

from ml_recsys_tools.utils.instrumentation import \
    log_time_and_shape, collect_named_init_params, LogCallsTimeAndOutput
from ml_recsys_tools.utils.logger import simple_logger

# monkey patch print function
skopt.callbacks.print = simple_logger.info


class Integer(skopt.space.Integer):
    # changes the sampled values from numpy ints to python ints
    def rvs(self, n_samples=1, random_state=None):
        return [int(s) for s in super().rvs(n_samples=n_samples, random_state=random_state)]


class Real(skopt.space.Real):
    # changes the sampled values from numpy floats to python floats
    def rvs(self, n_samples=1, random_state=None):
        return [float(s) for s in super().rvs(n_samples=n_samples, random_state=random_state)]


Categorical = skopt.space.Categorical


class SearchSpaceGuess:

    Categorical = Categorical
    Real = Real
    Integer = Integer

    def __init__(self, arg):
        """
        constructs an instance with attributes from the dictionary of names and default values
        :param arg: either a dict of {name: default value} or and instance or a class for which to construct a guess
        """
        if isinstance(arg, dict):
            self.set_from_dict(arg)
        else:
            self.set_from_class(arg)

    def set_from_dict(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, bool):
                variable = self.Categorical([value])
            elif isinstance(value, int):
                variable = self.Integer(value, value + 1)
            elif isinstance(value, float):
                variable = self.Real(value, value + 1)
            else:
                variable = self.Categorical([value])
            setattr(self, key, variable)
        return self

    def __repr__(self):
        return pprint.pformat(self.__dict__)

    def set_from_class(self, obj):
        """
        Guess a class search space from init params and construct an object with
        params as attributes and skopt variable distributions as values
        :param obj: the object from which to set the init search space
        :return: a SearchSpace object with params attributes named as the parameters and skopt variables as values
        """

        if isinstance(obj, type):
            cls = obj
        else:
            cls = type(obj)

        all_init_params = collect_named_init_params(cls)

        search_space = dict([el
                             for d in all_init_params.values()
                             for el in list(d.items())])

        return self.set_from_dict(search_space)


class BayesSearchHoldOut(LogCallsTimeAndOutput):

    target_loss_col = 'target_loss'
    time_taken_col = 'time_taken'

    def __init__(self, search_space, pipeline, loss, random_ratio=0.5,
                 plot_graph=True, interrupt_message_file=None,  n_parallel=1,
                 verbose=True, smooth_transition=False, **kwargs):
        self.search_space = search_space
        self.loss = loss
        self.pipeline = pipeline
        self.train_data = None
        self.validation_data = None
        self.callbacks = None
        self.random_ratio = random_ratio
        self.plot_graph = plot_graph
        self.smooth_transition = smooth_transition
        self.interrupt_message_file = interrupt_message_file
        self.n_parallel = n_parallel
        super().__init__(verbose=verbose, **kwargs)
        self.all_metrics = pd.DataFrame()

    def values_to_dict(self, values):
        return point_asdict(self.search_space, values)

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
            params = self.search_inst.values_to_dict(values)
            simple_logger.info(params)

    def init_pipeline(self, values):
        for i, value in enumerate(values):
            if isinstance(value, np.int_):
                values[i] = int(value)
            elif isinstance(value, np.float_):
                values[i] = float(value)
        params_dict = self.values_to_dict(values)
        pipeline = copy.copy(self.pipeline)
        pipeline.set_params(**params_dict)
        return pipeline

    def _add_loss_and_time_to_report(self, loss, time_taken, report_df=None):
        if report_df is None:
            report_df = pd.DataFrame()
        report_df[self.target_loss_col] = loss
        report_df[self.time_taken_col] = time_taken
        return report_df

    def _fit_predict_loss(self, pipeline):
        start = time.time()
        pipeline.fit(self.train_data['x'], self.train_data['y'])
        y_pred = pipeline.predict(self.validation_data['x'])
        loss = self.loss(self.validation_data['y'], y_pred)
        return loss, self._add_loss_and_time_to_report(loss, time.time() - start)

    def objective_func(self, values):
        try:
            self._check_interrupt()
            pipeline = self.init_pipeline(values)
            loss, report_df = self._fit_predict_loss(pipeline)
            return loss, report_df
        except Exception as e:
            simple_logger.exception(e)
            simple_logger.error(values)
            return 1.0, self._add_loss_and_time_to_report(0, 0)

    def _init_optimizer(self, n_calls):
        return Optimizer(
            dimensions=dimensions_aslist(search_space=self.search_space),
            base_estimator=RandomForestRegressor(n_estimators=10),
            n_initial_points=int(n_calls * self.random_ratio),
            acq_func="EI",
            acq_optimizer="sampling",
            acq_optimizer_kwargs=dict(n_points=1000, n_jobs=-1),
            acq_func_kwargs=dict(xi=0.01, kappa=1.96))

    def _init_callbacks(self, n_calls):
        self.callbacks = [
            self.PrintIfBestCB(self),
            VerboseCallback(n_total=n_calls, n_random=int(n_calls * self.random_ratio))]

    def _eval_callbacks(self, result):
        [c(result) for c in self.callbacks]

    def _smooth_transition(self, optimizer, progress):
        # this should transition from random to BO
        if self.smooth_transition:
            optimizer.n_points = int(2000 ** progress)
        return optimizer

    def optimize(self, train_data, validation_data, n_calls):
        self.train_data = train_data
        self.validation_data = validation_data
        optimizer = self._init_optimizer(n_calls)
        self._init_callbacks(n_calls)

        config_q = Queue(maxsize=1)  # this is to keep the configs fresh
        results_q = Queue()

        def _config_putter(q):
            for i in range(n_calls):
                q.put(optimizer.ask())
            for i in range(self.n_parallel):
                q.put('end')

        jobs = [Thread(target=_config_putter, name='_config_putter', args=(config_q,))]

        if self.n_parallel == 1:
            jobs.append(Thread(target=self._parallel_worker, args=(config_q, results_q)))  # for easier debugging
        else:
            jobs.extend([Process(target=self._parallel_worker, args=(config_q, results_q))
                   for _ in range(self.n_parallel)])
        [j.start() for j in jobs]

        # get results from queue
        for i in range(n_calls):
            next_x, next_y, report_df = results_q.get()
            self._record_all_metrics(report_df=report_df, values=next_x)
            optimizer = self._smooth_transition(optimizer, i / n_calls)
            result = optimizer.tell(next_x, next_y)
            self._eval_callbacks(result)

        [j.join() for j in jobs]
        return self._format_and_plot_result(result)

    def _parallel_worker(self, q_in, q_out):
        while True:
            next_x = q_in.get()
            if next_x == 'end':
                break
            next_y, report_df = self.objective_func(next_x)
            q_out.put((next_x, next_y, report_df))

    def _format_and_plot_result(self, result):
        best_values = result.x
        best_params = self.values_to_dict(best_values)
        best_model = self.init_pipeline(best_values)

        if self.plot_graph:
            pyplot.figure()
            plot_convergence(result)
            pyplot.plot(result.func_vals)

        # return res_bo, best_params, best_model
        return SimpleNamespace(**{
            'optimizer': self,
            'report': self.best_results_summary(),
            'mutual_info_loss': self.params_mutual_info(),
            'mutual_info_time': self.params_mutual_info(self.time_taken_col),
            'result': result,
            'best_params': best_params,
            'best_model': best_model})

    def _check_interrupt(self):
        if self.interrupt_message_file is not None \
                and os.path.exists(self.interrupt_message_file):

            with open(self.interrupt_message_file) as f:
                message = f.readline()

            if 'stop' in message:
                raise InterruptedError('interrupted by "stop" message in %s'
                                       % self.interrupt_message_file)
            elif 'pause' in message:
                simple_logger.warn('Paused by "pause" message in %s'
                            % self.interrupt_message_file)
                while 'pause' in message:
                    time.sleep(1)
                    with open(self.interrupt_message_file) as f:
                        message = f.readline()
                self._check_interrupt()

            elif 'update' in message:
                simple_logger.warn('Updating HP space due to "update" message in %s'
                            % self.interrupt_message_file)
                raise NotImplementedError('not yet implemented')

    def _record_all_metrics(self, values, report_df=None):
        # records the time and the other metrics
        if report_df is None:
            report_df = pd.DataFrame()
        params_dict = self.values_to_dict(values)
        report_df = report_df.assign(**params_dict)
        self.all_metrics = self.all_metrics.append(report_df)

    def best_results_summary(self):
        return self.all_metrics. \
            reset_index(). \
            drop('level_0', axis=1). \
            sort_values(self.target_loss_col)

    def params_mutual_info(self, target_col=None):
        if target_col is None:
            target_col = self.target_loss_col
        mutual_info = {}
        target = self.all_metrics[target_col].values.reshape(-1, 1)
        for feat in self.search_space.keys():
            vec = self.all_metrics[feat].values.reshape(-1, 1)
            try:
                mutual_info[feat] = \
                    mutual_info_regression(vec, target)[0]
            except ValueError:  # categorical feature (string)
                mutual_info[feat] = mutual_info_regression(
                    LabelEncoder().fit_transform(
                        vec.astype(str)).reshape(-1, 1), target)[0]
        return pd.DataFrame([mutual_info])


def early_stopping_runner(
        score_func, check_point_func,
        epochs_start=0, epochs_max=200, epochs_step=10,
        stop_patience=10, decline_threshold=0.05,
        plot_graph=True):
    res_list = []
    max_score = 0
    decline_counter = 0
    cur_epoch = 0
    epochs_list = []
    max_epoch = 0
    # find optimal number of epochs on validation data
    while cur_epoch <= epochs_max:

        cur_step = epochs_start + epochs_step if cur_epoch == 0 else epochs_step

        simple_logger.info('Training epochs %d - %d.' %
                           (cur_epoch, cur_epoch + cur_step))

        cur_epoch += cur_step
        epochs_list.append(cur_epoch)

        cur_score = score_func(cur_epoch, cur_step)
        res_list.append(cur_score)

        # early stopping logic
        if max_score * (1 - decline_threshold) > cur_score:
            decline_counter += cur_step
            if decline_counter >= stop_patience:
                break
        else:
            decline_counter = 0

        if cur_score > max_score:
            max_score = cur_score
            max_epoch = cur_epoch
            check_point_func()

    # print logging info
    scores_str = ','.join(['%d%%(%d)' % (int(100 * s / max_score), e)
                           for s, e in zip(res_list, epochs_list)])

    simple_logger.info('Early stopping: stopped fit after %d '
                       'epochs (max validation score: %f (@%d), all scores: %s)'
                       % (cur_epoch, max_score, max_epoch, scores_str))

    if plot_graph:
        pyplot.figure()
        pyplot.plot(epochs_list, res_list)

    return max_epoch


class VotingEnsemble(VotingClassifier):
    """
    Adds caching for probabilities to the VotingClassifer to enable quick predictions with updated weights
    (functools lru_cache is no good because it expects immutable inputs)
    """

    def __init__(self, estimators, voting='hard', weights=None, n_jobs=1,
                 flatten_transform=None):
        super().__init__(estimators, voting, weights, n_jobs, flatten_transform)
        self._proba_cache = None
        self._x_cache = None

    def _collect_probas(self, x):
        if (self._proba_cache is None) or (self._x_cache is None) or \
                (not np.asarray(list(self._x_cache) == list(x)).all()):
            self._proba_cache = VotingClassifier._collect_probas(self, x)
            self._x_cache = x
        return self._proba_cache


class EnsembleTrainer:
    def __init__(self, models_losses, models_params, cutoff_ratio, pipeline, data_dict):
        """

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

        """
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
                if sum(new_weights) == 0.0:
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
