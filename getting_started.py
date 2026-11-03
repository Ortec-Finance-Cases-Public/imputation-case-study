# %% [markdown]
# # Getting started
#
# A first look at the data and the helper functions. Run it cell by cell (VS Code,
# Jupyter with jupytext) or as a script: `python getting_started.py`.

# %%
import matplotlib.pyplot as plt

from casestudy.data import (
    FEATURES, MISSING_COLUMNS, TARGET, dataset_names, load_dataset, missing_patterns,
    missing_summary,
)
from casestudy.validation import hide_values, imputation_scores, random_mask

print(dataset_names())

# %% [markdown]
# ## Load a dataset
# Missing values are `NaN`. `construction_period` is an ordered category.

# %%
df = load_dataset("dataset_1")
print(df.shape)
print(df.dtypes)
df.head()

# %% [markdown]
# ## What is missing?

# %%
print(missing_summary(df))
print(missing_patterns(df))

# %% [markdown]
# ## Compare the observed data with the rows that have something missing
# You can compare the *other* columns of rows with and without a missing value.

# %%
miss = df["living_area_m2"].isna()
print(df.groupby(miss)[[TARGET, "quarter_index"]].mean())
print(df.groupby(miss)["dwelling_type"].value_counts(normalize=True).unstack(0).round(3))

# %%
fig, ax = plt.subplots(figsize=(6, 3))
for flag, label in [(False, "living area observed"), (True, "living area missing")]:
    ax.hist(df.loc[miss == flag, TARGET].clip(upper=1_000_000), bins=60, density=True,
            alpha=0.5, label=label)
ax.set_xlabel("price (EUR, clipped at 1 million)")
ax.legend()
plt.show()

# %% [markdown]
# ## Hide observed values to test a method
# The true values of missing cells are unknown, so hide cells that you can see,
# impute them, and compare.

# %%
from solution import Solution

complete_rows = df.dropna(subset=MISSING_COLUMNS)  # rows with nothing missing
solution = Solution().fit(df)

mask = random_mask(complete_rows[FEATURES], rate=0.2, seed=1)
hidden = hide_values(complete_rows[FEATURES], mask)
imputed = solution.impute(hidden)
print(imputation_scores(complete_rows[FEATURES], imputed, mask).round(3))
