"""Check your ``solution.py`` on your own validation split.

For each dataset this script

1. holds out a share of the training rows as a validation set (the validation
   features are passed to your code without the price, like the hidden test set),
2. fits ``Solution`` on the rest,
3. scores ``predict`` against the validation prices (log-price RMSE, overall and
   for rows with / without missing values),
4. hides some *observed* validation cells at random and scores how well
   ``impute`` recovers them.

Note that step 4 hides cells completely at random. Whether that resembles how the
values in your data went missing is for you to find out. Build your own masks with
``casestudy.validation`` to test other situations.

Usage::

    python evaluate_local.py                      # all datasets
    python evaluate_local.py --dataset dataset_2
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from casestudy.data import FEATURES, TARGET, dataset_names, is_missing, load_dataset
from casestudy.validation import (
    hide_values, imputation_scores, log_price_rmse, random_mask, train_validation_split,
    validate_imputation,
)
from solution import Solution


def evaluate(name: str, seed: int, hide_rate: float, validation_fraction: float) -> None:
    train, valid = train_validation_split(load_dataset(name), validation_fraction, seed)
    X_valid, y_valid = valid[FEATURES], valid[TARGET]
    print(f"\n=== {name}: train {len(train):,} rows, validation {len(valid):,} rows")

    start = time.perf_counter()
    solution = Solution().fit(train)
    print(f"fit: {time.perf_counter() - start:.1f}s")

    # Price prediction on the validation rows as they are (incomplete).
    X_in = X_valid.copy()
    predicted = solution.predict(X_valid)
    if not X_valid.equals(X_in):
        raise ValueError("predict() modified its input.")
    if len(predicted) != len(X_valid):
        raise ValueError("predict() must return one price per row.")
    has_missing = is_missing(X_valid).any(axis=1).to_numpy()
    scores = {"all rows": log_price_rmse(y_valid, predicted)}
    scores["rows with a missing value"] = log_price_rmse(y_valid[has_missing],
                                                         predicted[has_missing])
    scores["rows without"] = log_price_rmse(y_valid[~has_missing], predicted[~has_missing])
    print("log-price RMSE:  " + "   ".join(f"{k}: {v:.4f}" for k, v in scores.items()))

    # Imputation of observed cells that are hidden on purpose.
    mask = random_mask(X_valid, rate=hide_rate, seed=seed)
    X_hidden = hide_values(X_valid, mask)
    X_hidden_in = X_hidden.copy()
    imputed = solution.impute(X_hidden)
    if not X_hidden.equals(X_hidden_in):
        raise ValueError("impute() modified its input.")
    validate_imputation(X_hidden, imputed)
    print(f"imputation of {int(mask.to_numpy().sum()):,} randomly hidden cells:")
    with pd.option_context("display.width", 120, "display.float_format", "{:.3f}".format):
        print(imputation_scores(X_valid, imputed, mask).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default=None, help="default: all datasets")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hide-rate", type=float, default=0.1,
                        help="share of observed cells hidden for the imputation check")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    for name in [args.dataset] if args.dataset else dataset_names():
        evaluate(name, args.seed, args.hide_rate, args.validation_fraction)


if __name__ == "__main__":
    main()
