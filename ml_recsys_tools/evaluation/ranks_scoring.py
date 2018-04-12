from collections import OrderedDict

import numpy as np
import pandas as pd
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score, reciprocal_rank
from ml_recsys_tools.utils.parallelism import N_CPUS


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


def mean_scores_report(model, datasets, dataset_names):
    ranks_list = [model.predict_rank(dataset, num_threads=N_CPUS) for dataset in datasets]
    return mean_scores_report_on_ranks(ranks_list, datasets, dataset_names)


def mean_scores_report_on_ranks(ranks_list, datasets, dataset_names):
    data = []
    for ranks, dataset in zip(ranks_list, datasets):
        res = all_scores_on_ranks(ranks, dataset).describe().loc['mean']
        data.append(res)
    return pd.DataFrame(data=data, index=dataset_names)


def all_scores_on_ranks(ranks, test_data, train_data=None, k=10):
    ranks_kwargs = \
        {'ranks': ranks,
         'test_interactions': test_data,
         'train_interactions': train_data,
         }

    best_possible_kwargs = \
        {'ranks': best_possible_ranks(test_data),
         'test_interactions': test_data,
         'train_interactions': train_data,
         }

    metrics = OrderedDict([
        ('AUC', auc_score_on_ranks(**ranks_kwargs)),
        ('reciprocal', reciprocal_rank_on_ranks(**ranks_kwargs)),
        ('n-MRR@%d' % k, mrr_norm_on_ranks(**ranks_kwargs, k=k)),
        ('n-MRR', mrr_norm_on_ranks(**ranks_kwargs)),
        ('n-DCG@%d' % k,
         dcg_binary_at_k(**ranks_kwargs, k=k) /
         dcg_binary_at_k(**best_possible_kwargs, k=k)),
        ('n-precision@%d' % k,
         precision_at_k_on_ranks(**ranks_kwargs, k=k) /
         precision_at_k_on_ranks(**best_possible_kwargs, k=k)),
        ('precision@%d' % k, precision_at_k_on_ranks(**ranks_kwargs, k=k)),
        ('recall@%d' % k, recall_at_k_on_ranks(**ranks_kwargs, k=k)),
        ('n-recall@%d' % k,
         recall_at_k_on_ranks(**ranks_kwargs, k=k) /
         recall_at_k_on_ranks(**best_possible_kwargs, k=k)),
        ('n-i-gini@%d' % k,
         (1 - gini_coefficient_at_k(**ranks_kwargs, k=k)) /
         (1 - gini_coefficient_at_k(**best_possible_kwargs, k=k))),
        ('n-diversity@%d' % k,
         diversity_at_k(**ranks_kwargs, k=k) /
         diversity_at_k(**best_possible_kwargs, k=k)),
        ])

    return pd.DataFrame(metrics)[list(metrics.keys())]


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
        # from https://stackoverflow.com/questions/39512260/calculating-gini-coefficient-in-python-numpy
        # Array indexing requires reset indexes.
        x = pd.Series(x).reset_index(drop=True)
        if w is None:
            w = np.ones_like(x)
        w = pd.Series(w).reset_index(drop=True)
        n = x.size
        wxsum = sum(w * x)
        wsum = sum(w)
        sxw = np.argsort(x)
        sx = x[sxw] * w[sxw]
        sw = w[sxw]
        pxi = np.cumsum(sx) / wxsum
        pci = np.cumsum(sw) / wsum
        g = 0.0
        for i in np.arange(1, n):
            g = g + pxi.iloc[i] * pci.iloc[i - 1] - pci.iloc[i] * pxi.iloc[i - 1]
        return g

    ranks = ranks.copy().tocsr()
    ranks.data += 1
    ranks.data[ranks.data > k] *= 0
    ranks.eliminate_zeros()

    cols, counts = np.unique(ranks.indices, return_counts=True)
    counts = np.concatenate([counts, np.zeros(ranks.shape[1] - len(counts))])

    n_rows = np.sum(test_interactions.getnnz(axis=1) > 0) if not preserve_rows else ranks.shape[0]

    return np.repeat(gini(counts), n_rows)
