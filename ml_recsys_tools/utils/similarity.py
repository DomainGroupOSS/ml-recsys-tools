import numpy as np
import warnings
from functools import partial
from sklearn.metrics.pairwise import cosine_similarity
from ml_recsys_tools.utils.parallelism import map_batches_multiproc
from ml_recsys_tools.utils.debug import print_time_and_shape


def _row_ind_mat(ar):
    # returns a matrix of column indexes of the right shape to enable indexing
    return np.indices(ar.shape)[0]

def top_N_unsorted(mat, N):
    # returns top N values and their indexes for each row in a matrix (axis=1)
    # results are unsorted (to save on sort, when only filtering is needed)

    N = np.min([N, mat.shape[-1]])

    top_inds = np.argpartition(mat, -N)[:, -N:]

    top_values = mat[_row_ind_mat(top_inds), top_inds]

    return top_values, top_inds

def _argsort_mask_descending(mat):
    # gets index mask for sorting a matrix by last axis (sorts rows) in descending order
    sort_inds = (_row_ind_mat(mat), np.argsort(-mat, axis=1))
    return sort_inds

def top_N_sorted(mat, N):
    # returns sorted top N elements and indexes in each row of matrix mat

    top_values, top_inds = top_N_unsorted(mat, N)

    sort_inds = _argsort_mask_descending(top_values)

    return top_values[sort_inds], top_inds[sort_inds]

def _top_N_similar(inds, source_mat, target_mat, N, remove_self,
                   source_biases=None, target_biases=None, simil_mode='cosine'):
    '''
    for each row in specified inds in source_mat calculates top N similar items in target_mat
    :param inds: indices into source mat
    :param source_mat: matrix of features for similarity calculation (left side)
    :param target_mat: matrix of features for similarity calculation (right side)
    :param N: number of top elements to retreive
    :param remove_self: whether to remove first element - for cases when
        source elements are present in target_mat (self similarity)
    :param source_biases: bias terms for source_mat
    :param target_biases: bias terms for target_mat
    :param simil_mode: type of similarity calculation:
        'cosine' dot product of normalized matrices (each row sums to 1), without biases
        'dot' regular dot product, without normalization
    :return:
    '''

    if simil_mode=='cosine':
        scores = cosine_similarity(source_mat[inds, :], target_mat)

    elif simil_mode=='dot':
        scores = np.dot(source_mat[inds, :], target_mat.T)

        if source_biases is not None:
            scores = (scores.T + source_biases[inds]).T

        if target_biases is not None:
            scores += target_biases

    else:
        raise NotImplementedError('unknown similarity mode')

    # get best N
    if remove_self:
        N = N + 1

    best_scores, best_inds = top_N_unsorted(scores, N)

    sort_inds = _argsort_mask_descending(best_scores)

    # checks that top similar items are themselves
    if remove_self:
        if not all(best_inds[list(inds[:, 0] for inds in sort_inds)] == np.array(inds)):
            warnings.warn("LightFM: _most_similar_by_cosine: Most similar "
                          "element is not itself, something's wrong!")
        sort_inds = list(inds[:, 1:] for inds in sort_inds)

    return best_inds[sort_inds], best_scores[sort_inds]

def most_similar(ids, N, remove_self, source_encoder, source_mat, source_biases=None,
                 target_encoder=None, target_mat=None,  target_biases=None,
                 chunksize=1000, simil_mode='cosine', pbar=None):
    '''
    multithreaded batched version of _top_N_similar() that works with IDs instead of indices
    for each row in specified IDS in source_mat calculates top N similar items in target_mat

    :param ids: IDS of query items in source mat
    :param N: number of top items to find for each query item
    :param remove_self: whether to remove first element - for cases when
        source elements are present in target_mat (self similarity)
    :param source_encoder: encoder for transforming IDS to indeces in source_mat
    :param source_mat: features matrix for query items
    :param source_biases: biases for query items
    :param target_encoder: encoder for transforming IDS to indeces in target_mat
    :param target_mat: features matrix for target items
    :param target_biases: biases for target items
    :param chunksize: chunksize for batching (in term of query items)
    :param simil_mode: mode of similarity calculation:
        'cosine' dot product of normalized matrices (each row sums to 1), without biases
        'dot' regular dot product, without normalization
    :return:
        best_ids - matrix (n_ids, N) of N top items from target_mat for each item in IDS of source_mat
        best_scores - similarity scores for best_ids (n_ids, N)
    '''

    if target_mat is None:
        target_mat = source_mat
        target_encoder = source_encoder
        target_biases = source_biases

    # to index
    inds = source_encoder.transform(ids)

    chunksize = int(35000 * chunksize / max(source_mat.shape))

    calc_func = partial(
        _top_N_similar, source_mat=source_mat, target_mat=target_mat, N=N,
        remove_self=remove_self, source_biases=source_biases, target_biases=target_biases,
        simil_mode=simil_mode)

    ret = map_batches_multiproc(calc_func, inds,
                                chunksize=chunksize,
                                pbar=pbar,
                                threads_per_cpu=2)
    best_inds = np.concatenate([r[0] for r in ret], axis=0)
    best_scores = np.concatenate([r[1] for r in ret], axis=0)

    # back to ids
    best_ids = target_encoder.inverse_transform(best_inds.astype(int))

    return best_ids, best_scores


@print_time_and_shape
def custom_row_func_on_sparse(ids, source_encoder, target_encoder,
                              sparse_mat, row_func, chunksize=10000, pbar=None):

    source_inds = source_encoder.transform(ids)

    sub_mat_ll = sparse_mat.tocsr()[source_inds, :].tolil()

    def top_n_inds_batch(inds_batch):
        inds_list = []
        vals_list = []
        for i in inds_batch:
            inds_, vals_ = row_func(np.array(sub_mat_ll.rows[i]),
                                    np.array(sub_mat_ll.data[i]))
            inds_list.append(inds_)
            vals_list.append(vals_)
        return np.stack(inds_list), np.stack(vals_list)

    batch_res = map_batches_multiproc(
        top_n_inds_batch, np.arange(sub_mat_ll.shape[0]), chunksize=chunksize, pbar=pbar)

    best_inds = np.concatenate([r[0] for r in batch_res])
    best_scores = np.concatenate([r[1] for r in batch_res])

    # back to ids
    best_ids = target_encoder.inverse_transform(best_inds.astype(int))

    return best_ids, best_scores

@print_time_and_shape
def top_N_sorted_on_sparse(ids, encoder, sparse_mat, n_top=10, chunksize=10000, pbar=None):

    def _pad_k_zeros(vec, k):
        return np.pad(vec, (0, k), 'constant', constant_values=0)

    def top_n_row(row_indices, row_data):

        n_min = min(n_top, len(row_data))

        i_sort = np.argsort(-row_data)[:n_min]

        if n_min == n_top:
            return row_indices[i_sort], row_data[i_sort]
        else:
            return \
                _pad_k_zeros(row_indices[i_sort], n_top - n_min), \
                _pad_k_zeros(row_data[i_sort], n_top - n_min),

    return custom_row_func_on_sparse(
        row_func=top_n_row,
        ids=ids,
        source_encoder=encoder,
        target_encoder=encoder,
        sparse_mat=sparse_mat,
        chunksize=chunksize,
        pbar=pbar
    )
