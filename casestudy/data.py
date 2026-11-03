"""Reading the data and describing its missing values."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

TARGET = "price_eur"
FEATURES = [
    "living_area_m2", "plot_area_m2", "municipality_code", "property_type_group",
    "neighbourhood_code", "is_monument", "dwelling_type", "repet_flag",
    "construction_period", "quarter_index", "corop_plus_region_code",
]
#: Columns in which values can be missing. All other columns are always observed.
MISSING_COLUMNS = ["living_area_m2", "plot_area_m2", "construction_period"]
AREA_COLUMNS = ["living_area_m2", "plot_area_m2"]
#: Integer codes that identify a category; treat them as categorical, not numeric.
CODE_COLUMNS = ["municipality_code", "neighbourhood_code", "corop_plus_region_code",
                "property_type_group"]

CONSTRUCTION_PERIODS = [
    "<1900", "1900-1919", "1920-1929", "1930-1944", "1945", "1946-1959",
    "1960-1969", "1970-1979", "1980-1989", "1990-1999", "2000-2009",
    "2010-2014", ">=2015",
]
DWELLING_TYPES = [
    "other (reference)", "linked house", "semi-detached", "gallery flat", "maisonnette",
    "ground-floor apartment", "upper-floor apartment", "porch flat", "penthouse",
]


def dataset_names() -> list[str]:
    """Names of the available datasets, e.g. ``['dataset_1', 'dataset_2', ...]``."""
    return sorted(p.name.removesuffix("_train.csv") for p in DATA_DIR.glob("dataset_*_train.csv"))


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Give the columns their intended dtypes (a copy is returned).

    * ``dwelling_type`` is categorical,
    * ``construction_period`` is an *ordered* categorical (oldest to newest),
    * the two areas are floats, so that ``NaN`` can mark a missing value.

    Apply this to any dataframe that you read or build yourself, so that
    categories are identical in training and test data.
    """
    out = df.copy()
    out["construction_period"] = pd.Categorical(
        out["construction_period"], categories=CONSTRUCTION_PERIODS, ordered=True)
    out["dwelling_type"] = pd.Categorical(out["dwelling_type"], categories=DWELLING_TYPES)
    for column in AREA_COLUMNS:
        out[column] = out[column].astype(float)
    return out


def load_dataset(name: str) -> pd.DataFrame:
    """Read ``data/<name>_train.csv`` with the right dtypes.

    The result has the columns in ``FEATURES`` and the target ``price_eur``.
    Missing values are ``NaN``.
    """
    path = DATA_DIR / f"{name}_train.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Available: {dataset_names()}")
    return coerce_types(pd.read_csv(path))


def is_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean frame, True where a value is missing (``MISSING_COLUMNS`` only)."""
    return df[MISSING_COLUMNS].isna()


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Number and share of missing values per column that can have them."""
    miss = is_missing(df)
    return pd.DataFrame({"n_missing": miss.sum(), "share_missing": miss.mean()})


def missing_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """How often each combination of missing columns occurs.

    Rows are patterns (``True`` = missing in that column), sorted by frequency.
    Shows, for example, whether columns tend to go missing together.
    """
    miss = is_missing(df)
    counts = miss.value_counts().rename("n_rows").reset_index()
    counts["share"] = counts["n_rows"] / len(df)
    return counts
