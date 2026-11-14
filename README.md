# Task 5 — Feature Engineering & Tuned Ensembles (Ames Housing)

This project continues the Task 1 and Task 3 Ames Housing regression work. It creates eight domain-driven features, re-runs the Task 3 log-Ridge baseline, and tunes Random Forest and XGBoost using 5-fold cross-validation.

## Files

- `feature_engineering.py` — feature creation and Task 3-compatible dataset preparation.
- `tuned_models.py` — 5-fold CV, RandomizedSearchCV, hold-out evaluation and report generation.
- `MODEL_COMPARISON.md` — generated comparison and model-selection verdict.
- `requirements.txt` — required Python packages.
- `outputs/` — generated CV table, hold-out metrics, best parameters and search results.

## Dataset

Use the `ames_housing_cleaned.csv` file produced in Task 1 (Data-Cleaning-Quality-Audit). The dataset is not included in this repository.

## Run locally or in Google Colab

```bash
pip install -r requirements.txt
python tuned_models.py --data ames_housing_cleaned.csv
```

Optional controls:

```bash
python tuned_models.py --data ames_housing_cleaned.csv --n-iter 4 --jobs -1
```

`--n-iter` controls the number of randomized configurations for each ensemble. The default is 4 for a quick reproducible run. Increase it to 10–20 for broader search coverage, with longer runtime.

## Validation and leakage prevention

The script recreates Task 3's exact 80/20 split (`random_state=42`) before fitting any preprocessing. The 5-fold CV and all hyperparameter tuning use only the training partition. Feature creation is deterministic and inside the sklearn pipeline; imputers, scaling and one-hot encoding are also inside the pipeline and are fitted separately within each CV fold. The 20% hold-out set is evaluated only after tuning is complete.

The training target is `log1p(SalePrice)`, matching Task 3's strongest Ridge configuration. `TransformedTargetRegressor` converts predictions back to the original SalePrice scale before RMSE and R² are calculated.
