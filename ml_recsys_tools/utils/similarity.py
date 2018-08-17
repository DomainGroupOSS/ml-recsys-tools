import numpy as np
import scipy.sparse as sp
import warnings
from functools import partial
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from ml_recsys_tools.utils.parallelism import map_batches_multiproc
from ml_recsys_tools.utils.instrumentation import log_time_and_shape


def _row_ind_mat(ar):
    # returns a matrix of column indexes of the right shape to enable indexing
    return np.indices(ar.shape)[0]


def top_N_unsorted(mat, n):
    # returns top N values and their indexes for each row in a matrix (axis=1)
    # results are unsorted (to save on sort, when only filtering is needed)

    n = np.min([n, mat.shape[-1]])

    top_inds = np.argpartition(mat, -n)[:, -n:]

    top_values = mat[_row_ind_mat(top_inds), top_inds]

    return np.array(top_inds), np.array(top_values)


def _argsort_mask_descending(mat):
    # gets index mask for sorting a matrix by last axis (sorts rows) in descending order
    sort_inds = (_row_ind_mat(mat), np.argsort(-mat, axis=1))
    return sort_inds


def top_N_sorted(mat, n):
    # returns sorted top N elements and indexes in each row of matrix mat

    top_inds, top_values= top_N_unsorted(mat, n)

    sort_inds = _argsort_mask_descending(top_values)

    return top_inds[sort_inds], top_values[sort_inds]


def _top_N_similar(source_inds, source_mat, target_mat, n,
                   exclude_mat_sp=None, source_biases=None, target_biases=None,
                   simil_mode='cosine'):
    """
    for each row in specified inds in source_mat calculates top N similar items in target_mat
    :param source_inds: indices into source mat
    :param source_mat: matrix of features for similarity calculation (left side)
    :param target_mat: matrix of features for similarity calculation (right side)
    :param n: number of top elements to retreive
    :param exclude_mat_sp: a sparse matrix with interactions to exclude
    :param source_biases: bias terms for source_mat
    :param target_biases: bias terms for target_mat
    :param simil_mode: type of similarity calculation:
        'cosine' dot product of normalized matrices (each row sums to 1), without biases
        'dot' regular dot product, without normalization
    :return:
    """

    if not len(source_inds):
        return np.array([[]]), np.array([[]])

    if simil_mode == 'cosine':
        scores = cosine_similarity(source_mat[source_inds, :], target_mat)

    elif simil_mode == 'euclidean':
        scores = 1 / (euclidean_distances(source_mat[source_inds, :], target_mat) + 0.001)

    elif simil_mode == 'dot':
        scores = np.dot(source_mat[source_inds, :], target_mat.T)

        if source_biases is not None:
            scores = (scores.T + source_biases[source_inds]).T

        if target_biases is not None:
            scores += target_biases

        if sp.issparse(scores):
            scores = scores.toarray()
        else:
            scores = np.array(scores)

    else:
        raise NotImplementedError('unknown similarity mode')

    if exclude_mat_sp is not None:
        exclude_mat_sp_coo = exclude_mat_sp[source_inds, :].tocoo()
        scores[exclude_mat_sp_coo.row, exclude_mat_sp_coo.col] = -np.inf

    best_inds, best_scores = top_N_unsorted(scores, n)

    sort_inds = _argsort_mask_descending(best_scores)

    return best_inds[sort_inds], best_scores[sort_inds]


def most_similar(source_ids, n, source_encoder, source_mat, source_biases=None,
                 target_ids=None, target_encoder=None, target_mat=None, target_biases=None,
                 exclude_mat_sp=None,
                 chunksize=1000, simil_mode='cosine'):
    """
    multithreaded batched version of _top_N_similar() that works with IDs instead of indices
    for each row in specified IDS in source_mat calculates top N similar items in target_mat

    :param source_ids: IDS of query items in source mat
    :param n: number of top items to find for each query item
    :param remove_self: whether to remove first element - for cases when
        source elements are present in target_mat (self similarity)
    :param source_encoder: encoder for transforming IDS to indeces in source_mat
    :param source_mat: features matrix for query items
    :param source_biases: biases for query items
    :param target_ids: subset of target ids to be considered
    :param target_encoder: encoder for transforming IDS to indeces in target_mat
    :param target_mat: features matrix for target items
    :param target_biases: biases for target items
    :param exclude_mat_sp: a sparse mat with interactions to exclude (e.g. training mat)
    :param chunksize: chunksize for batching (in term of query items)
    :param simil_mode: mode of similarity calculation:
        'cosine' dot product of normalized matrices (each row sums to 1), without biases
        'dot' regular dot product, without normalization, with added biases if supplied
        'euclidean' inverse of euclidean distance
    :return:
        best_ids - matrix (n_ids, N) of N top items from target_mat for each item in IDS of source_mat
        best_scores - similarity scores for best_ids (n_ids, N)
    """

    if target_mat is None:
        target_mat = source_mat
        target_encoder = source_encoder
        target_biases = source_biases

    # to index
    source_inds = source_encoder.transform(np.array(source_ids, dtype=str))

    if target_ids is None:
        target_ids = target_encoder.classes_

    target_inds = target_encoder.transform(np.array(target_ids, dtype=str))
    target_inds.sort()

    chunksize = int(35000 * chunksize / max(source_mat.shape))

    calc_func = partial(
        _top_N_similar,
        source_mat=source_mat,
        target_mat=target_mat[target_inds, :],  # only the relevant submatrix
        exclude_mat_sp=exclude_mat_sp[:, target_inds] if exclude_mat_sp is not None else None,
        n=n,
        source_biases=source_biases,
        target_biases=target_biases[target_inds] if target_biases is not None else None,
        simil_mode=simil_mode)

    ret = map_batches_multiproc(calc_func, source_inds,
                                chunksize=chunksize,
                                threads_per_cpu=2)
    sub_mat_best_inds = np.concatenate([r[0] for r in ret], axis=0)
    best_scores = np.concatenate([r[1] for r in ret], axis=0)

    # back to ids
    best_inds = target_inds[sub_mat_best_inds.astype(int)]
    best_ids = target_encoder.inverse_transform(best_inds.astype(int))

    return best_ids, best_scores


@log_time_and_shape
def custom_row_func_on_sparse(source_ids, source_encoder, target_encoder,
                              sparse_mat, row_func, exclude_mat_sp=None,
                              target_ids=None, chunksize=10000):
    source_inds = source_encoder.transform(np.array(source_ids, dtype=str))

    if target_ids is None:
        target_ids = target_encoder.classes_

    target_inds = target_encoder.transform(np.array(target_ids, dtype=str))
    target_inds.sort()

    sub_mat_ll = sparse_mat. \
                     tocsr()[source_inds, :][:, target_inds].tolil()

    exclude_mat_ll = exclude_mat_sp[source_inds, :][:, target_inds].tolil() \
        if exclude_mat_sp is not None else None

    def top_n_inds_batch(inds_batch):
        nonlocal exclude_mat_ll, sub_mat_ll
        inds_list = []
        vals_list = []
        for i in inds_batch:
            exlclude_inds = np.array(exclude_mat_ll.rows[i]) \
                if exclude_mat_ll is not None else None
            inds_, vals_ = row_func(np.array(sub_mat_ll.rows[i]),
                                    np.array(sub_mat_ll.data[i]),
                                    exlclude_inds)
            inds_list.append(inds_)
            vals_list.append(vals_)
        return np.array(np.stack(inds_list)), np.array(np.stack(vals_list))

    batch_res = map_batches_multiproc(
        top_n_inds_batch, np.arange(sub_mat_ll.shape[0]), chunksize=chunksize)

    sub_mat_best_inds = np.concatenate([r[0] for r in batch_res])
    best_scores = np.concatenate([r[1] for r in batch_res])

    # back to ids
    best_inds = target_inds[sub_mat_best_inds.astype(int)]
    best_ids = target_encoder.inverse_transform(best_inds.astype(int))

    return best_ids, best_scores


@log_time_and_shape
def top_N_sorted_on_sparse(source_ids, target_ids, encoder, sparse_mat,
                           n_top=10, chunksize=10000):
    def _pad_k_zeros(vec, k):
        return np.pad(vec, (0, k), 'constant', constant_values=0)

    def top_n_row(row_indices, row_data, exclude_inds):

        n_min = min(n_top, len(row_data))

        if exclude_inds is not None:
            row_data[row_indices == exclude_inds] = -np.inf

        i_sort = np.argsort(-row_data)[:n_min]

        if n_min == n_top:
            return row_indices[i_sort], row_data[i_sort]
        else:
            return \
                _pad_k_zeros(row_indices[i_sort], n_top - n_min), \
                _pad_k_zeros(row_data[i_sort], n_top - n_min),

    return custom_row_func_on_sparse(
        row_func=top_n_row,
        source_ids=source_ids,
        target_ids=target_ids,
        source_encoder=encoder,
        target_encoder=encoder,
        sparse_mat=sparse_mat,
        chunksize=chunksize,
    )
