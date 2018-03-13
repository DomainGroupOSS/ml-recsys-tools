import numpy as np
import pandas as pd
import sklearn.preprocessing


class FloatBinningEncoder(sklearn.preprocessing.LabelEncoder):
    def __init__(self, n_bins = 50):
        super().__init__()
        self.n_bins = n_bins
        self.bins = None

    def fit(self, y):
        percentiles = list(np.linspace(0, 100, num=(self.n_bins + 1)))
        self.bins = np.percentile(y, percentiles[1:])

        if len(np.unique(self.bins)) != len(self.bins):
            # non-unique bins using simple linear binning
            self.bins = list(np.linspace(
                np.min(y) - 0.001, np.max(y) + 0.001, num=(self.n_bins + 1)))

        return self

    def transform(self, y):
        # copy bins
        inclusive_bins = list(self.bins)

        # extend the bins to the current edges safely
        inclusive_bins[0] = min(inclusive_bins[0], np.min(y))
        inclusive_bins[-1] = max(inclusive_bins[-1], np.max(y))

        # get the bin indices
        y_binned = pd.cut(
            y, bins=inclusive_bins, labels=False, include_lowest=True)
        y_indeces = y_binned.astype(int, copy=False)
        return y_indeces


class FloatBinningBinarizer(sklearn.preprocessing.LabelBinarizer):
    def __init__(self, n_bins = 50, **kwargs):
        super().__init__(**kwargs)
        self._pre_encoder = FloatBinningEncoder(n_bins=n_bins)

    def fit(self, y):
        self._pre_encoder.fit(y)
        # this is because encoder just returns indices
        # and in order to protect from the binarizer ignoring empty
        # bins and having fewer classes than expected
        super().fit(range(len(self._pre_encoder.bins)))
        return self

    def transform(self, y):
        y_binned = self._pre_encoder.transform(y)
        return super().transform(y_binned)

