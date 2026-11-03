"""Your solution. This is the only file that is run on the hidden test sets.

The three methods below are called for every dataset, on a fresh ``Solution()``:

* ``fit(train)``    receives the incomplete training data, **including** ``price_eur``.
* ``impute(X)``     receives features only (no price) and must return the same
                    dataframe with every missing value filled in.
* ``predict(X)``    receives features only, with missing values, and must return the
                    predicted price in **euros** (one positive number per row).

Rules:

* The test data has the same columns, dtypes and missingness pattern as the
  training data, but no ``price_eur``.
* ``fit`` must work from the dataframe it is given. Do not read files or rely
  on anything saved by an earlier run.
* ``impute`` and ``predict`` must not modify ``X`` in place and must not look at
  other test rows' prices (there are none). They may use anything learned in ``fit``.
* Imputed values must be valid: areas are numbers, ``construction_period`` is
  one of ``CONSTRUCTION_PERIODS``.
* You may add helper modules and install extra packages, but document them.

The code below is a simple baseline (mean/mode imputation and gradient boosting).
Replace it with something better.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from casestudy.data import AREA_COLUMNS, FEATURES, TARGET


class Solution:
    def fit(self, train: pd.DataFrame) -> "Solution":
        self.fill_values_ = {c: train[c].mean() for c in AREA_COLUMNS}
        self.fill_values_["construction_period"] = train["construction_period"].mode()[0]

        X = self.impute(train[FEATURES])
        self.model_ = HistGradientBoostingRegressor(categorical_features="from_dtype",
                                                    random_state=0)
        self.model_.fit(X, np.log(train[TARGET]))
        return self

    def impute(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for column, value in self.fill_values_.items():
            out[column] = out[column].fillna(value)
        return out

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.exp(self.model_.predict(self.impute(X[FEATURES])))
