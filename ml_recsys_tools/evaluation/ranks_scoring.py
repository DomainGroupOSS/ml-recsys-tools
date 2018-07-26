from collections import OrderedDict
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score, reciprocal_rank

from ml_recsys_tools.utils.instrumentation import LogCallsTimeAndOutput
from ml_recsys_tools.utils.parallelism import N_CPUS


class ModelMockRanksCacher:
    """
    this is used in order to use lightfm functions without copying them out and rewriting
    this makes possible to:
        - not calculate ranks every time from scratch (if you have them precalculated)
        - use the functions to score non lightfm models
    """

    def __init__(self, cached_mat):
        self.cached_mat = cached_mat

    def predict_rank(self, *args, **kwargs):
        return self.cached_mat


def mean_scores_report(model, datasets, dataset_names, k=10):
    ranks_list = [model.predict_rank(dataset, num_threads=N_CPUS) for dataset in datasets]
    return mean_scores_report_on_ranks(ranks_list, datasets, dataset_names, k)


def mean_scores_report_on_ranks(ranks_list, datasets, dataset_names, k=10):
    data = []
    for ranks, dataset in zip(ranks_list, datasets):
        full_report = RanksScorer(ranks_mat=ranks, test_mat=dataset, k=k).scores_report()
        res = full_report.describe().loc['mean']
        data.append(res)
    return pd.DataFrame(data=data, index=dataset_names)


class RanksScorer(LogCallsTimeAndOutput):

    def __init__(self, ranks_mat, test_mat, train_mat=None, k=10, verbose=True):
        super().__init__(verbose=verbose)
        self.ranks_mat = ranks_mat
        self.test_mat = test_mat
        self.train_mat = train_mat
        self.k = k
        self.best_ranks_mat = self._best_ranks()

    def _best_ranks(self):
        return best_possible_ranks(self.test_mat)

    def _ranks_kwargs(self):
        return {
            'ranks': self.ranks_mat,
            'test_interactions': self.test_mat,
            'train_interactions': self.train_mat,
            }

    def _best_ranks_kwargs(self):
        return {
            'ranks': self.best_ranks_mat,
            'test_interactions': self.test_mat,
            'train_interactions': self.train_mat,
            }

    def scores_report(self):
        metrics = OrderedDict([
            ('AUC', self.auc),
            ('reciprocal', self.reciprocal),
            ('n-MRR@%d' % self.k, self.n_mrr_at_k),
            ('n-MRR', self.n_mrr),
            ('n-DCG@%d' % self.k, self.n_dcg_at_k),
            ('n-precision@%d' % self.k, self.n_precision_at_k),
            ('precision@%d' % self.k, self.precision_at_k),
            ('recall@%d' % self.k, self.recall_at_k),
            ('n-recall@%d' % self.k, self.n_recall_at_k),
            ('n-i-gini@%d' % self.k, self.n_i_gini_at_k),
            ('n-diversity@%d' % self.k, self.n_diversity_at_k),
        ])

        with ThreadPool(len(metrics)) as pool:
            res = [pool.apply_async(f) for f in metrics.values()]
            for k, r in zip(metrics.keys(), res):
                metrics[k] = r.get()
        return pd.DataFrame(metrics)[list(metrics.keys())]

    def auc(self):
        return auc_score_on_ranks(**self._ranks_kwargs())

    def reciprocal(self):
        return reciprocal_rank_on_ranks(**self._ranks_kwargs())

    def n_mrr_at_k(self):
        return mrr_norm_on_ranks(**self._ranks_kwargs(), k=self.k)

    def n_mrr(self):
        return mrr_norm_on_ranks(**self._ranks_kwargs())

    def n_dcg_at_k(self):
        return dcg_binary_at_k(**self._ranks_kwargs(), k=self.k) / \
               dcg_binary_at_k(**self._best_ranks_kwargs(), k=self.k)

    def n_precision_at_k(self):
        return precision_at_k_on_ranks(**self._ranks_kwargs(), k=self.k) / \
               precision_at_k_on_ranks(**self._best_ranks_kwargs(), k=self.k)

    def precision_at_k(self):
        return precision_at_k_on_ranks(**self._ranks_kwargs(), k=self.k)

    def n_recall_at_k(self):
        return recall_at_k_on_ranks(**self._ranks_kwargs(), k=self.k) / \
               recall_at_k_on_ranks(**self._best_ranks_kwargs(), k=self.k)

    def recall_at_k(self):
        return recall_at_k_on_ranks(**self._ranks_kwargs(), k=self.k)

    def n_i_gini_at_k(self):
        return (1 - gini_coefficient_at_k(**self._ranks_kwargs(), k=self.k)) / \
               (1 - gini_coefficient_at_k(**self._best_ranks_kwargs(), k=self.k))

    def n_diversity_at_k(self):
        return diversity_at_k(**self._ranks_kwargs(), k=self.k) / \
               diversity_at_k(**self._best_ranks_kwargs(), k=self.k)


def best_possible_ranks(test_mat):
    best_ranks = test_mat.tocsr().copy()
    n_users, n_items = test_mat.shape
    item_inds = np.arange(n_items)
    nnz_counts = best_ranks.getnnz(axis=1)
    best_ranks.data = np.concatenate(
        [np.random.choice(item_inds[:n], n, replace=False) if n else []
         for n in nnz_counts]).astype(np.float32)
    return best_ranks


def chance_ranks(test_mat):
    rand_ranks = test_mat.tocsr().copy()
    n_users, n_items = test_mat.shape
    item_inds = np.arange(n_items)
    nnz_counts = rand_ranks.getnnz(axis=1)
    rand_ranks.data = np.concatenate(
        [np.random.choice(item_inds, n, replace=False)
         for n in nnz_counts]).astype(np.float32)
    return rand_ranks


def precision_at_k_on_ranks(
        ranks, test_interactions, train_interactions=None, k=10, preserve_rows=False):
    return precision_at_k(
        model=ModelMockRanksCacher(ranks.copy()),
        test_interactions=test_interactions,
        train_interactions=train_interactions,
        k=k,
        preserve_rows=preserve_rows)


def recall_at_k_on_ranks(
        ranks, test_interactions, train_interactions=None, k=10, preserve_rows=False):
    return recall_at_k(
        model=ModelMockRanksCacher(ranks.copy()),
        test_interactions=test_interactions,
        train_interactions=train_interactions,
        k=k,
        preserve_rows=preserve_rows,
    )


def auc_score_on_ranks(
        ranks, test_interactions, train_interactions=None, preserve_rows=False):
    return auc_score(
        model=ModelMockRanksCacher(ranks.copy()),
        test_interactions=test_interactions,
        train_interactions=train_interactions,
        preserve_rows=preserve_rows,
    )


def reciprocal_rank_on_ranks(
        ranks, test_interactions, train_interactions=None, preserve_rows=False):
    return reciprocal_rank(
        model=ModelMockRanksCacher(ranks.copy()),
        test_interactions=test_interactions,
        train_interactions=train_interactions,
        preserve_rows=preserve_rows,
    )


def mrr_norm_on_ranks(
        ranks, test_interactions, train_interactions=None, preserve_rows=False, k=None):

    def harmonic_number(n):
        # https://stackoverflow.com/questions/404346/python-program-to-calculate-harmonic-series
        """Returns an approximate value of n-th harmonic number.
           http://en.wikipedia.org/wiki/Harmonic_number
        """
        # Euler-Mascheroni constant
        gamma = 0.57721566490153286060651209008240243104215933593992
        return gamma + np.log(n) + 0.5 / n - 1. / (12 * n ** 2) + 1. / (120 * n ** 4)

    # number of test items in each row + epsilon for subsequent reciprocals / devisions
    total_positives = np.diff(test_interactions.tocsr().indptr)

    reciprocals = ranks.copy()
    reciprocals.data = 1.0 / (ranks.data + 1)

    if k:
        reciprocals.data[ranks.data > (k - 1)] = 0
        total_positives[total_positives > k] = k

    # sum the reciprocals and devide by count of test interactions in each row
    mrr = np.squeeze(np.array(reciprocals.sum(axis=1)))

    # the max mrr is the partial sum of the harmonic series divided by number of items:
    #  1/n * (1/1 + 1/2 + 1/3 ... 1/n) for n = number of items
    max_mrr = harmonic_number(total_positives + 0.001)

    mrr_norm = mrr / max_mrr

    if not preserve_rows:
        mrr_norm = mrr_norm[test_interactions.getnnz(axis=1) > 0]

    return mrr_norm


def dcg_binary_at_k(
        ranks, test_interactions, k=10, train_interactions=None, preserve_rows=False):
    ranks = ranks.copy()

    ranks.data += 1
    ranks.data[ranks.data > k] *= 0
    ranks.eliminate_zeros()

    ranks.data = 1 / (np.log2(ranks.data + 1))

    dcg = np.squeeze(np.array(ranks.sum(axis=1)))

    if not preserve_rows:
        dcg = dcg[test_interactions.getnnz(axis=1) > 0]

    return dcg


def diversity_at_k(ranks, test_interactions, k=10, train_interactions=None, preserve_rows=False):
    """
    Diversity metric:
        calculates the percentage of items that
        were recommended @ k for any user out of all possible items
    """
    ranks = ranks.copy().tocsr()
    ranks.data += 1
    ranks.data[ranks.data > k] *= 0
    ranks.eliminate_zeros()

    cols, counts = np.unique(ranks.indices, return_counts=True)
    n_cols = ranks.shape[1]
    percentage = len(counts) / n_cols

    n_rows = np.sum(test_interactions.getnnz(axis=1) > 0) if not preserve_rows else ranks.shape[0]

    return np.repeat(percentage, n_rows)


def gini_coefficient_at_k(ranks, test_interactions, k=10, train_interactions=None, preserve_rows=False):
    """
    Diversity metric:
        calculates the gini coefficient for the
        counts of recommended items @ k for all users
    """

    def gini(x, w=None):
        # https://stackoverflow.com/questions/48999542/more-efficient-weighted-gini-coefficient-in-python/
        # The rest of the code requires numpy arrays.
        x = np.asarray(x)
        if w is not None:
            w = np.asarray(w)
            sorted_indices = np.argsort(x)
            sorted_x = x[sorted_indices]
            sorted_w = w[sorted_indices]
            # Force float dtype to avoid overflows
            cumw = np.cumsum(sorted_w, dtype=float)
            cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
            return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) /
                    (cumxw[-1] * cumw[-1]))
        else:
            sorted_x = np.sort(x)
            n = len(x)
            cumx = np.cumsum(sorted_x, dtype=float)
            # The above formula, with all weights equal to 1 simplifies to:
            return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

    ranks = ranks.copy().tocsr()
    ranks.data += 1
    ranks.data[ranks.data > k] *= 0
    ranks.eliminate_zeros()

    cols, counts = np.unique(ranks.indices, return_counts=True)
    counts = np.concatenate([counts, np.zeros(ranks.shape[1] - len(counts))])

    n_rows = np.sum(test_interactions.getnnz(axis=1) > 0) if not preserve_rows else ranks.shape[0]

    return np.repeat(gini(counts), n_rows)
