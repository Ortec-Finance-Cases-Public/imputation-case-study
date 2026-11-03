"""Helpers for validating an imputation method on the data you have.

You only have the incomplete data, so the true values of missing cells are
unknown. You can still evaluate by hiding cells that *are* observed
(``random_mask`` and ``hide_values``) and checking how well they are recovered
(``imputation_scores``). The way you choose which cells to hide matters. Think
about whether it resembles how the real values went missing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from casestudy.data import (
    AREA_COLUMNS, CONSTRUCTION_PERIODS, MISSING_COLUMNS,
)


def train_validation_split(df: pd.DataFrame, validation_fraction: float = 0.2,
                           seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random split into a training and a validation dataframe (index reset)."""
    rng = np.random.default_rng(seed)
    is_valid = np.zeros(len(df), dtype=bool)
    is_valid[rng.choice(len(df), size=int(len(df) * validation_fraction), replace=False)] = True
    return df[~is_valid].reset_index(drop=True), df[is_valid].reset_index(drop=True)


def random_mask(df: pd.DataFrame, rate: float = 0.1, columns: list[str] | None = None,
                seed: int = 0) -> pd.DataFrame:
    """Boolean frame marking a random share of the *observed* cells to hide.

    Every observed cell in ``columns`` is chosen independently with probability
    ``rate``. Write your own function that returns a frame of the same shape to
    hide cells in some other way.
    """
    rng = np.random.default_rng(seed)
    columns = columns or MISSING_COLUMNS
    return pd.DataFrame({c: (rng.random(len(df)) < rate) & df[c].notna().to_numpy()
                         for c in columns}, index=df.index)


def hide_values(df: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``df`` with the cells where ``mask`` is True set to missing."""
    out = df.copy()
    for column in mask.columns:
        out[column] = out[column].mask(mask[column].to_numpy(dtype=bool))
    return out


def imputation_scores(actual: pd.DataFrame, imputed: pd.DataFrame,
                      mask: pd.DataFrame) -> pd.DataFrame:
    """Error of imputed values on the cells in ``mask`` (one row per column).

    * areas: ``rmse`` and ``bias`` (mean of imputed minus actual, in m²),
    * construction period: ``accuracy`` and the ordinal error in steps
      (``mae_steps``, ``bias_steps``).

    ``actual`` holds the values that were hidden, ``imputed`` the method's output.
    """
    rows = {}
    for column in AREA_COLUMNS:
        m = mask[column].to_numpy(dtype=bool)
        if m.any():
            error = imputed.loc[m, column].to_numpy(float) - actual.loc[m, column].to_numpy(float)
            rows[column] = {"n_cells": int(m.sum()), "rmse": float(np.sqrt((error ** 2).mean())),
                            "bias": float(error.mean())}
    m = mask["construction_period"].to_numpy(dtype=bool)
    if m.any():
        a = actual.loc[m, "construction_period"].cat.codes.to_numpy()
        p = imputed.loc[m, "construction_period"].cat.codes.to_numpy()
        rows["construction_period"] = {
            "n_cells": int(m.sum()), "accuracy": float((a == p).mean()),
            "mae_steps": float(np.abs(a - p).mean()), "bias_steps": float((p - a).mean())}
    return pd.DataFrame.from_dict(rows, orient="index")


def log_price_rmse(price_true, price_pred) -> float:
    """Root mean squared error of the log price (both arguments in euros)."""
    price_true, price_pred = np.asarray(price_true, float), np.asarray(price_pred, float)
    if not (np.isfinite(price_pred).all() and (price_pred > 0).all()):
        raise ValueError("Predicted prices must be finite and positive (in euros).")
    return float(np.sqrt(((np.log(price_pred) - np.log(price_true)) ** 2).mean()))


def validate_imputation(before: pd.DataFrame, after: pd.DataFrame) -> None:
    """Check the output of ``Solution.impute``; raise ``ValueError`` if it is invalid.

    The result must have the same rows and columns as the input, no missing
    values in ``MISSING_COLUMNS``, valid construction periods, and unchanged
    values in every cell that was observed.
    """
    if not isinstance(after, pd.DataFrame):
        raise ValueError("impute() must return a pandas DataFrame.")
    if not before.index.equals(after.index) or list(before.columns) != list(after.columns):
        raise ValueError("impute() must keep the same index and columns as its input.")
    if after[MISSING_COLUMNS].isna().any().any():
        raise ValueError("impute() left missing values in "
                         f"{list(after[MISSING_COLUMNS].columns[after[MISSING_COLUMNS].isna().any()])}.")
    if not after["construction_period"].astype(str).isin(CONSTRUCTION_PERIODS).all():
        raise ValueError("impute() returned construction periods that do not exist.")
    for column in MISSING_COLUMNS:
        observed = before[column].notna().to_numpy()
        if not before.loc[observed, column].astype(str).equals(
                after.loc[observed, column].astype(str)):
            raise ValueError(f"impute() changed observed values in {column}.")
