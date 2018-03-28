import numpy as np
import pandas as pd
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score, reciprocal_rank
from ml_recsys_tools.utils.debug import log_time_and_shape
from ml_recsys_tools.utils.parallelism import N_CPUS


def best_possible_ranks(test_mat):
    best_ranks = test_mat.tocsr().copy()
    nnz_counts = best_ranks.getnnz(axis=1)
    best_ranks.data = np.concatenate([np.arange(n) for n in nnz_counts]).astype(np.float32)
    return best_ranks

def chance_ranks(test_mat):
    n_users, n_items = test_mat.shape
    item_inds = np.arange(n_items)
    rand_ranks = test_mat.tocsr().copy()
    nnz_counts = rand_ranks.getnnz(axis=1)
    rand_ranks.data = np.concatenate([np.random.choice(item_inds, n) for n in nnz_counts]).astype(np.float32)
    return rand_ranks

@log_time_and_shape
def mean_scores_report(model, datasets, dataset_names):
    ranks_list = [model.predict_rank(dataset, num_threads=N_CPUS) for dataset in datasets]
    return mean_scores_report_on_ranks(ranks_list, datasets, dataset_names)

@log_time_and_shape
def mean_scores_report_on_ranks(ranks_list, datasets, dataset_names):
    data = []
    for ranks, dataset in zip(ranks_list, datasets):
        res = all_scores_on_ranks(ranks, dataset).describe().loc['mean']
        data.append(res)
    return pd.DataFrame(data=data, index=dataset_names)

def all_scores_on_ranks(ranks, test_data, train_data=None, k=10):

    ranks_kwargs = \
        {'ranks':ranks,
        'test_interactions': test_data,
        'train_interactions': train_data,
        }

    best_possible_kwargs = \
        {'ranks': best_possible_ranks(test_data),
        'test_interactions': test_data,
        'train_interactions': train_data,
        }

    # chance_ranks_kwargs = {'ranks': chance_ranks(test_data),
    #                    'test_interactions': test_data,
    #                    'train_interactions': train_data,
    #                    }
    metrics = {'recall (k=%d)' % k: recall_at_k_on_ranks(**ranks_kwargs, k=k),
               'n-recall': recall_at_k_on_ranks(**ranks_kwargs, k=k) /
                               recall_at_k_on_ranks(**best_possible_kwargs, k=k),
               # 'recall MAX poss': recall_at_k_on_ranks(**best_possible_kwargs, k=k),
               # 'recall chance': recall_at_k_on_ranks(**chance_ranks_kwargs, k=k),
               'precision (k=%d)' % k: precision_at_k_on_ranks(**ranks_kwargs, k=k),
               'n-precision': precision_at_k_on_ranks(**ranks_kwargs, k=k) /
                                  precision_at_k_on_ranks(**best_possible_kwargs, k=k),
               # 'precision MAX poss': precision_at_k_on_ranks(**best_possible_kwargs, k=k),
               # 'precision chance': precision_at_k_on_ranks(**chance_ranks_kwargs, k=k),
               'AUC': auc_score_on_ranks(**ranks_kwargs),
               'reciprocal': reciprocal_rank_on_ranks(**ranks_kwargs),
               'n-MRR': mrr_norm_on_ranks(**ranks_kwargs),
               'n-DCG': dcg_binary_at_k(**ranks_kwargs, k=k) /
                        dcg_binary_at_k(**best_possible_kwargs, k=k)
               }

    return pd.DataFrame(metrics)


class ModelMockRanksCacher:
    '''
    this is used in order to use lightfm functions without copying them out and rewriting
    this makes possible to:
        - not calculate ranks every time from scratch (if you have them precalculated)
        - use the functions to score non lightfm models
    '''
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
        ranks, test_interactions, train_interactions=None, preserve_rows=False):

    def harmonic_number(n):
        # https://stackoverflow.com/questions/404346/python-program-to-calculate-harmonic-series
        """Returns an approximate value of n-th harmonic number.
           http://en.wikipedia.org/wiki/Harmonic_number
        """
        # Euler-Mascheroni constant
        gamma = 0.57721566490153286060651209008240243104215933593992
        return gamma + np.log(n) + 0.5 / n - 1. / (12 * n ** 2) + 1. / (120 * n ** 4)

    ranks = ranks.copy()

    ranks.data = 1.0 / (ranks.data + 1.0)

    # number of test items in each row + epsilon for subsequent reciprocals / devisions
    test_counts = np.diff(test_interactions.tocsr().indptr) + 0.001

    # sum the reciprocals and devide by count of test interactions in each row
    mrr = np.squeeze(np.array(ranks.sum(axis=1))) / test_counts

    # the max mrr is the partial sum of the harmonic series divided by number of items:
    #  1/n * (1/1 + 1/2 + 1/3 ... 1/n) for n = number of items
    max_mrr = harmonic_number(test_counts) / test_counts

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


