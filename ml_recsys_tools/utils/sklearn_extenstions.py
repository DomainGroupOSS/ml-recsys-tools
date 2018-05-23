from collections import defaultdict

import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_is_fitted
from pandas.core.algorithms import _get_data_algo, _hashtables, _ensure_object
import importlib

pd_cat_module = importlib.import_module(pd.Categorical.__module__)


class PDLabelEncoder(sklearn.preprocessing.LabelEncoder):
    """
    adapterd and sped up further from here: https://github.com/scikit-learn/scikit-learn/issues/7432
    faster version of encoder that's using Pandas encoding
    which is using hash tables instead of sorted arrays
    """

    def fit(self, y):
        y = column_or_1d(y, warn=True)
        _, self.classes_ = pd.factorize(y, sort=True)
        self._cat_dtype = pd_cat_module.CategoricalDtype(self.classes_)
        self._dtype = self._cat_dtype.categories.dtype
        self._table = self._get_table_for_categories(self.classes_, self._cat_dtype.categories)
        return self

    @staticmethod
    def _get_table_for_categories(values, categories):
        # ripped out of _get_codes_for_values() in pandas Categorical module
        if not pd_cat_module.is_dtype_equal(values.dtype, categories.dtype):
            values = _ensure_object(values)
            categories = _ensure_object(categories)

        (hash_klass, vec_klass), vals = _get_data_algo(values, _hashtables)
        (_, _), cats = _get_data_algo(categories, _hashtables)
        t = hash_klass(len(cats))
        t.map_locations(cats)
        return t

    def transform(self, y, check_labels=True):
        check_is_fitted(self, ['classes_', '_cat_dtype', '_table', '_dtype'])
        y = column_or_1d(y, warn=True)

        # trans_y = pd.Categorical(y, dtype=self._cat_dtype).codes.copy()  # much slower
        trans_y = pd_cat_module.coerce_indexer_dtype(
            indexer=self._table.lookup(y.astype(self._dtype)),
            categories=self._cat_dtype.categories)

        if check_labels:
            mask_new_labels = self._new_labels_locs(trans_y)
            if np.any(mask_new_labels):
                raise ValueError("y contains new labels: %s" %
                                 str(np.unique(y[mask_new_labels])))

        return trans_y

    def _new_labels_locs(self, trans_y):
        return trans_y == -1

    def find_new_labels(self, y):
        return self._new_labels_locs(self.transform(y, check_labels=False))

    def __getstate__(self):
        # custom pickling behaviour because the table object is unpickleable
        state = super().__getstate__()
        state['_table'] = None
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._table = self._get_table_for_categories(self.classes_, self._cat_dtype.categories)


class DictLabelEncoder(sklearn.preprocessing.LabelEncoder):
    """
    python dict based version of the hash table based encoder
    somewhat slower but no pandas private parts and no pickling issues
    """

    def fit(self, y):
        self.classes_ = np.array(list(set(y)))
        self._table = defaultdict(
            lambda: -1, {k: i for i, k in enumerate(self.classes_)})
        return self

    def transform(self, y, check_labels=True):
        y = np.array(y)
        trans_y = np.array([self._table[v] for v in y])

        if check_labels:
            mask_new_labels = self._new_labels_locs(trans_y)
            if np.any(mask_new_labels):
                raise ValueError("y contains new labels: %s" %
                                 str(np.unique(y[mask_new_labels])))
        return trans_y

    def _new_labels_locs(self, trans_y):
        return trans_y == -1

    def find_new_labels(self, y):
        return self._new_labels_locs(self.transform(y, check_labels=False))


class FloatBinningEncoder(sklearn.preprocessing.LabelEncoder):
    """
    class for label-encoding a continuous variable by binning
    """
    def __init__(self, n_bins=50):
        super().__init__()
        self.n_bins = n_bins
        self.bins = None

    def fit(self, y):
        percentiles = list(np.linspace(0, 100, num=(self.n_bins + 1)))
        self.bins = np.percentile(y, percentiles[1:])

        if len(np.unique(self.bins)) != len(self.bins):
            self.bins = list(np.linspace(
                np.min(y) - 0.001, np.max(y) + 0.001, num=(self.n_bins + 1)))

        return self

    def transform(self, y):
        inc_bins = list(self.bins)
        inc_bins[0] = min(inc_bins[0], np.min(y))
        inc_bins[-1] = max(inc_bins[-1], np.max(y))

        y_binned = pd.cut(y, bins=inc_bins, labels=False, include_lowest=True)
        y_ind = y_binned.astype(int, copy=False)
        return y_ind


class FloatBinningBinarizer(sklearn.preprocessing.LabelBinarizer):
    """
    class for one-hot encoding a continuous variable by binning
    """
    def __init__(self, n_bins=50, spillage=2, **kwargs):
        """
        :param n_bins: number of bins
        :param spillage:
            number of neighbouring bins that are also activated in
            order to preserve some "proximity" relationship, default to 2
            e.g. for spillage=1 a result vec would be [.. 0, 0, 0.25, 0.5, 1, 0.5, 0.25, 0, 0 ..]
        :param kwargs:
        """
        super().__init__(**kwargs)
        self._spillage = spillage
        self._binner = FloatBinningEncoder(n_bins=n_bins)

    def fit(self, y):
        self._binner.fit(y)
        super().fit(range(len(self._binner.bins)))
        return self

    def transform(self, y):
        y_binned = self._binner.transform(y)
        binarized = super().transform(y_binned)
        if self._spillage:
            for i in range(1, self._spillage + 1):
                binarized += super().transform(y_binned + i) / 2**i
                binarized += super().transform(y_binned - i) / 2**i
        return binarized
