# Case study: house prices with missing data

You have four datasets of simulated Dutch house sales. In each of them some
feature values are missing, and **each dataset lost its values in its own way**.
Your task is to find out how, to impute the gaps, and to predict prices on
**hidden test data** that went missing in the same way.

## The data

`data/dataset_1_train.csv` ... `dataset_4_train.csv`, each with about 220,000
sales. The four datasets contain different sales, so they cannot be combined to
recover values.

| Column | Type | Description |
|---|---|---|
| `price_eur` | int | **Target.** Sale price in euros. Never missing. Not available in the test data. |
| `living_area_m2` | float | Living area in m². Can be missing. |
| `plot_area_m2` | float | Plot area in m²; 0 when there is no plot (e.g. apartments). Can be missing. |
| `construction_period` | ordered category | `<1900`, `1900-1919`, `1920-1929`, `1930-1944`, `1945`, `1946-1959`, `1960-1969`, `1970-1979`, `1980-1989`, `1990-1999`, `2000-2009`, `2010-2014`, `>=2015`. Can be missing. |
| `dwelling_type` | category | `other (reference)` (mainly detached and terraced houses), `linked house`, `semi-detached`, `gallery flat`, `maisonnette`, `ground-floor apartment`, `upper-floor apartment`, `porch flat`, `penthouse` |
| `is_monument` | 0/1 | Listed monument |
| `repet_flag` | 0/1 | Administrative indicator from the source data |
| `property_type_group` | code | Coarse property-type group (1, 2, 3, 4, 8). Labels, not quantities. |
| `neighbourhood_code`, `municipality_code`, `corop_plus_region_code` | codes | Location, nested from fine to coarse: neighbourhood ⊂ municipality ⊂ region. Categories, not numbers. |
| `quarter_index` | int | Quarter of the sale, 1–49 (about 12 years) |

Only `living_area_m2`, `plot_area_m2` and `construction_period` can be missing.
The other columns are always observed.

## Setup

```
uv sync                      # or: pip install -r requirements.txt
uv run python getting_started.py
uv run python evaluate_local.py
```

Python 3.11 or newer.

## What is in this repository

| Path | Content |
|---|---|
| `data/` | The four training datasets |
| `casestudy/data.py` | Read a dataset with the right dtypes (`load_dataset`), describe what is missing (`missing_summary`, `missing_patterns`, `is_missing`) |
| `casestudy/validation.py` | Split data, hide observed values on purpose (`random_mask`, `hide_values`), score imputations (`imputation_scores`), score prices (`log_price_rmse`) |
| `solution.py` | **Your hand-in.** A template with a simple baseline |
| `evaluate_local.py` | Runs `solution.py` on your own validation split of each dataset |
| `getting_started.py` | A first look at the data and the helpers |

## Tasks

1. **Diagnose.** For each dataset and each column that has gaps, find out
   whether the missingness can be predicted from the other columns. For
   example, train a classifier that predicts "is this value missing?".
   Evaluate it on held-out rows with a metric that fits the situation. Accuracy
   does not: if only a small share of the values is missing, always answering
   "observed" is already highly accurate. Explain what a result above (or at)
   chance level does and does not tell you.
2. **Impute.** Implement `Solution.impute`. The hidden test data has the same
   columns, but no price. Judge your method on cells that you hide yourself
   (see `casestudy/validation.py`). Think about how the cells that are hidden
   in your test should be chosen.
3. **Predict.** Implement `Solution.fit` and `Solution.predict` so that the
   price is predicted well when features are missing. Your method is scored on
   the log price.
4. **Reflect.** Is the best imputation method also the best price model? What
   can and cannot be known about the missing values from the observed data
   alone?

## Hand-in

Hand in `solution.py` (and any helper modules). It is run on the hidden test
sets, with a new `Solution()` for each dataset:

* `fit(train)`: `train` is the incomplete training dataframe with `price_eur`.
* `impute(X)`: features only; return the same dataframe with all missing values filled.
* `predict(X)`: features only, with missing values; return prices in euros.

The details and rules are in the docstring of `solution.py`. In short: `fit`
must work from the dataframe it receives, nothing may be read from disk or from
an earlier run, and no pickled models are handed in. Include a short report on
tasks 1 and 4, and list any packages you add to the environment.
