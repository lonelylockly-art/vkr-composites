"""Удаление выбросов только на обучающей части, в том числе внутри каждого CV-блока."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone


class TrainIQRRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, estimator, multiplier=1.5, categorical_index=10):
        self.estimator = estimator
        self.multiplier = multiplier
        self.categorical_index = categorical_index

    def fit(self, X, y):
        a = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(a, [25, 75], axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - (self.multiplier or 0) * iqr
        self.upper_ = q3 + (self.multiplier or 0) * iqr
        mask = np.ones(len(a), dtype=bool)
        if self.multiplier is not None:
            valid_columns = iqr > 0
            if self.categorical_index is not None:
                valid_columns[self.categorical_index] = False
            outside = ((a < self.lower_) | (a > self.upper_)) & valid_columns
            mask = ~outside.any(axis=1)
        self.keep_mask_ = mask
        self.n_removed_ = int((~mask).sum())
        self.n_features_in_ = a.shape[1]
        self.estimator_ = clone(self.estimator)
        clean_x = X.iloc[mask] if hasattr(X, 'iloc') else a[mask]
        clean_y = np.asarray(y)[mask]
        self.estimator_.fit(clean_x, clean_y)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)
