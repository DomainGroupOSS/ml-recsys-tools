import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_is_fitted


class PDLabelEncoder(sklearn.preprocessing.LabelEncoder):
    """
    from here: https://github.com/scikit-learn/scikit-learn/issues/7432
    faster version of encoder that's using Pandas encoding
    which is using hash tables instead of sorted arrays
    """

    def fit(self, y):
        y = column_or_1d(y, warn=True)
        _, self.classes_ = pd.factorize(y, sort=True)
        return self

    def transform(self, y, check_labels=True):
        check_is_fitted(self, 'classes_')
        y = column_or_1d(y, warn=True)

        if check_labels:
            classes = np.unique(y)
            if len(np.intersect1d(classes, self.classes_)) < len(classes):
                diff = np.setdiff1d(classes, self.classes_)
                raise ValueError("y contains new labels: %s" % str(diff))
        return pd.Categorical(y, categories=self.classes_).codes.copy()


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
