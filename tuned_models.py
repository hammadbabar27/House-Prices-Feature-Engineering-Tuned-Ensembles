"""Task 5: CV baseline + tuned Random Forest and XGBoost for Ames Housing.

Usage:
    python tuned_models.py --data ames_housing_cleaned.csv

The 80/20 Task 3 split is recreated first (random_state=42). All cross-
validation and hyperparameter tuning use only the 80% training partition.
The 20% hold-out partition is not used until the final evaluation section.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from feature_engineering import engineer_features, prepare_ames_features


RANDOM_STATE = 42
TASK3_HISTORICAL_HOLDOUT_RMSE = 24920.50


def make_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def make_estimator(model, numeric_columns: list[str], categorical_columns: list[str]):
    """Keep feature creation, imputation, scaling and encoding inside each fold."""
    pipeline = Pipeline(
        steps=[
            ("feature_engineering", FunctionTransformer(engineer_features, validate=False)),
            ("preprocessing", make_preprocessor(numeric_columns, categorical_columns)),
            ("model", model),
        ]
    )
    # The Task 3 experiment found log1p(SalePrice) gave its strongest Ridge
    # baseline. TTR inverse-transforms predictions so all CV scores remain in
    # the original SalePrice units, not log-price units.
    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def fmt_money(value: float) -> str:
    return f"{value:,.2f}"


def metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def markdown_table(df: pd.DataFrame, float_columns: list[str] | None = None) -> str:
    view = df.copy()
    float_columns = float_columns or []
    for column in float_columns:
        if column in view.columns:
            view[column] = view[column].map(lambda x: f"{x:,.4f}" if pd.notna(x) else "—")
    headers = list(view.columns)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in view.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_report(
    output_dir: Path,
    data_shape: tuple[int, int],
    n_train: int,
    n_test: int,
    numeric_count: int,
    categorical_count: int,
    cv_table: pd.DataFrame,
    holdout_table: pd.DataFrame,
    best_params: dict[str, dict],
    cv_fold_results: dict[str, list[float]],
    search_times: dict[str, float],
    n_iter: int,
) -> str:
    baseline_rmse = float(cv_table.loc[cv_table["Model"].str.startswith("Task 3 baseline"), "Mean CV RMSE"].iloc[0])
    baseline_std = float(cv_table.loc[cv_table["Model"].str.startswith("Task 3 baseline"), "Std CV RMSE"].iloc[0])
    ensemble_rows = cv_table[cv_table["Model"].isin(["Tuned Random Forest", "Tuned XGBoost"])]
    best_ensemble_row = ensemble_rows.sort_values("Mean CV RMSE").iloc[0]
    best_ensemble_name = str(best_ensemble_row["Model"])
    best_ensemble_rmse = float(best_ensemble_row["Mean CV RMSE"])
    gain_pct = (baseline_rmse - best_ensemble_rmse) / baseline_rmse * 100.0

    base_folds = np.asarray(cv_fold_results["Task 3 baseline"], dtype=float)
    best_folds = np.asarray(cv_fold_results[best_ensemble_name], dtype=float)
    fold_improvements = base_folds - best_folds
    positive_folds = int(np.sum(fold_improvements > 0))
    consistent = positive_folds >= 4
    meaningful = gain_pct >= 2.0 and consistent

    holdout_base = float(holdout_table.loc[holdout_table["Model"] == "Task 3 baseline + engineered features", "RMSE"].iloc[0])
    holdout_best = float(holdout_table.loc[holdout_table["Model"] == best_ensemble_name, "RMSE"].iloc[0])
    holdout_gain_pct = (holdout_base - holdout_best) / holdout_base * 100.0
    holdout_confirms = holdout_best <= holdout_base * 1.02
    historical_delta_pct = (holdout_base - TASK3_HISTORICAL_HOLDOUT_RMSE) / TASK3_HISTORICAL_HOLDOUT_RMSE * 100.0
    rf_holdout = float(holdout_table.loc[holdout_table["Model"] == "Tuned Random Forest", "RMSE"].iloc[0])
    xgb_holdout = float(holdout_table.loc[holdout_table["Model"] == "Tuned XGBoost", "RMSE"].iloc[0])
    rf_holdout_delta = (rf_holdout - holdout_base) / holdout_base * 100.0
    xgb_holdout_delta = (xgb_holdout - holdout_base) / holdout_base * 100.0
    rf_seconds = float(cv_table.loc[cv_table["Model"] == "Tuned Random Forest", "Training time (s)"].iloc[0])
    xgb_seconds = float(cv_table.loc[cv_table["Model"] == "Tuned XGBoost", "Training time (s)"].iloc[0])
    baseline_seconds = float(cv_table.loc[cv_table["Model"].str.startswith("Task 3 baseline"), "Training time (s)"].iloc[0])

    if meaningful and holdout_confirms:
        ship_model = best_ensemble_name
        verdict = (
            f"**Ship {ship_model}.** It reduces mean CV RMSE by {gain_pct:.2f}% versus the Task 3 Ridge baseline, "
            f"beats the baseline on {positive_folds}/5 paired folds, and its one-time hold-out RMSE is "
            f"{holdout_gain_pct:.2f}% lower. The gain clears the project's 2% practical-improvement rule "
            "and is supported by the untouched hold-out check."
        )
    else:
        ship_model = "Task 3 baseline: log-Ridge (alpha=0.1)"
        reasons = []
        if gain_pct < 2.0:
            reasons.append(f"the best ensemble improves mean CV RMSE by only {gain_pct:.2f}% (below the 2% practical threshold)")
        if not consistent:
            reasons.append(f"the ensemble beats the baseline on only {positive_folds}/5 folds")
        if not holdout_confirms:
            reasons.append(f"the hold-out RMSE is {abs(holdout_gain_pct):.2f}% higher than the engineered-feature Ridge baseline")
        if not reasons:
            reasons.append("the measured improvement is not sufficiently consistent to justify the added complexity")
        verdict = (
            f"**Ship the simpler {ship_model}.** Although {best_ensemble_name} has the lowest ensemble CV RMSE, "
            + "; ".join(reasons)
            + ". The ensemble is not justified by the current evidence."
        )

    params_lines = []
    for model_name, params in best_params.items():
        params_lines.append(f"### {model_name}\n\n```json\n{json.dumps(params, indent=2, sort_keys=True)}\n```")

    feature_justifications = [
        ("`TotalLivingArea`", "`GrLivArea + TotalBsmtSF`; combines above-grade living space and basement area into one broad interior-size measure."),
        ("`HouseAgeAtSale`", "`YrSold - YearBuilt`; older homes can differ in condition, materials, maintenance and buyer appeal."),
        ("`RemodelAgeAtSale`", "`YrSold - YearRemodAdd`; captures how recently the property was updated."),
        ("`TotalBathrooms`", "Full bathrooms plus half bathrooms weighted as 0.5, including basement bathrooms; summarizes bathroom capacity."),
        ("`HasRemodeled`", "1 when `YearRemodAdd > YearBuilt`; separates homes with a recorded remodel from homes without one."),
        ("`TotalPorchSF`", "Adds open, enclosed, three-season and screen porch areas to represent combined porch space."),
        ("`GarageAgeAtSale`", "`YrSold - GarageYrBlt`; captures garage age. Missing values are retained for fold-specific imputation."),
        ("`HasGarage`", "1 when `GarageArea > 0`; helps distinguish no garage from an unknown garage year."),
    ]
    feature_md = "\n".join(f"- **{name}:** {why}" for name, why in feature_justifications)

    comparison_md = markdown_table(
        cv_table[["Model", "Mean CV RMSE", "Std CV RMSE", "Mean CV R2", "Training time (s)", "Note"]],
        ["Mean CV RMSE", "Std CV RMSE", "Mean CV R2", "Training time (s)"],
    )
    fold_table = pd.DataFrame({
        "Fold": [1, 2, 3, 4, 5],
        "Task 3 log-Ridge RMSE": cv_fold_results["Task 3 baseline"],
        "Random Forest RMSE": cv_fold_results["Tuned Random Forest"],
        "XGBoost RMSE": cv_fold_results["Tuned XGBoost"],
    })
    fold_md = markdown_table(
        fold_table,
        ["Task 3 log-Ridge RMSE", "Random Forest RMSE", "XGBoost RMSE"],
    )
    holdout_md = markdown_table(
        holdout_table[["Model", "MAE", "RMSE", "R2"]],
        ["MAE", "RMSE", "R2"],
    )

    report = f"""# MODEL_COMPARISON — Feature Engineering & Tuned Ensembles (Ames Housing)

## 1. Objective

This task tests whether domain-driven features and tuned tree ensembles improve house-price prediction beyond Task 3's log-target Ridge baseline. The comparison uses the same 5-fold CV splits for all models and reserves the original Task 3 20% hold-out set for one final evaluation.

## 2. Dataset and leakage controls

- Source: Task 1 cleaned Ames Housing dataset (`ames_housing_cleaned.csv`).
- Dataset shape loaded: **{data_shape[0]:,} rows × {data_shape[1]} columns**.
- Target: `SalePrice`; training target is `log1p(SalePrice)` because Task 3 found this transformation useful.
- Task 3 split reproduced first with `train_test_split(test_size=0.20, random_state=42)`: **{n_train:,} training rows** and **{n_test:,} untouched hold-out rows**.
- Predictors exclude `SalePrice`, `Id`, and the three Task 1 IQR flags. `MSSubClass` and `MoSold` remain categorical.
- The Task 3 missing-value restoration rules are reused. Median/mode imputation, scaling, and one-hot encoding are fitted inside each CV fold's pipeline.
- The engineered features are deterministic calculations from input property and sale-year columns. They do not use `SalePrice` or target-derived statistics.
- The 20% hold-out set is not used for feature selection, hyperparameter tuning, or model selection. It is scored once after CV and tuning are complete.

### Original Task 3 reference

Task 3's best reported model was Ridge Regression on `log1p(SalePrice)` with `alpha=0.1`, with a held-out RMSE of **${TASK3_HISTORICAL_HOLDOUT_RMSE:,.2f}**. In this task, that same model configuration is re-run with the engineered features and 5-fold CV, so its CV results are directly comparable with the tuned ensembles.

## 3. Engineered features and domain reasoning

Eight new features were created:

{feature_md}

All calculations are inside the scikit-learn pipeline. Missing values are not filled during feature creation; the imputer learns its values from each fold's training partition only. These features are hypotheses, not guaranteed improvements. Some are correlated with existing fields, so cross-validation determines whether they add useful signal.

## 4. Validation and tuning method

- Split: the original Task 3 80/20 split (`random_state=42`).
- CV: `KFold(n_splits=5, shuffle=True, random_state=42)` on the 80% training partition only.
- Scoring: RMSE and R² in original `SalePrice` units. `TransformedTargetRegressor` applies `log1p` for fitting and `expm1` for predictions before scoring.
- Baseline: Task 3's `Ridge(alpha=0.1)` with the same feature engineering and preprocessing pipeline.
- Random Forest and XGBoost: tuned with `RandomizedSearchCV`, selecting parameters by mean CV RMSE. Each search samples multiple values across at least four hyperparameters.
- Training time includes the full 5-fold randomized search and final refit for ensemble models; baseline time is its 5-fold CV run. Times are wall-clock seconds and depend on hardware.

## 5. Cross-validation comparison

The table reports the mean and standard deviation of fold RMSE. Lower RMSE is better; higher R² is better.

{comparison_md}

### RMSE on each of the same five folds

{fold_md}

The Task 3 Ridge baseline has a notably large fold-to-fold spread. One fold is much harder for Ridge than the others, which raises its mean and standard deviation. This is why the CV summary is reported as mean ± standard deviation rather than as a single number.

## 6. Best hyperparameters

{chr(10).join(params_lines)}

## 7. Final untouched hold-out evaluation

After selecting the models using training-only CV, each selected estimator was evaluated on the original Task 3 20% hold-out set. This table is a final check, not a tuning input.

{holdout_md}

The historical Task 3 hold-out RMSE was **${TASK3_HISTORICAL_HOLDOUT_RMSE:,.2f}**. The current engineered-feature Ridge result is **${holdout_base:,.2f}**, a **{abs(historical_delta_pct):.2f}% {'higher' if historical_delta_pct > 0 else 'lower'}** error. This is effectively unchanged, so the engineered features alone did not improve the original Task 3 Ridge result on this hold-out set.

## 8. Verdict: what would I ship?

{verdict}

The project's practical-improvement rule is a **2% reduction in mean CV RMSE**, with improvement on at least 4 of the 5 paired folds, and no more than a 2% deterioration on the untouched hold-out set. This is a transparent project decision rule, not a formal statistical significance test. Five folds provide a more reliable estimate than one split, but they do not prove statistical significance on their own.

- Best ensemble by CV RMSE: **{best_ensemble_name}**.
- CV RMSE reduction versus engineered-feature Ridge: **{gain_pct:.2f}%**.
- Paired folds where the ensemble has lower RMSE: **{positive_folds}/5**.
- Hold-out RMSE change versus engineered-feature Ridge: **{holdout_gain_pct:.2f}%** (positive means lower error).
- Recommended model: **{ship_model}**.

### Complexity and runtime trade-off

In this run, the 5-fold baseline took **{baseline_seconds:.2f} seconds**, the full Random Forest search took **{rf_seconds:.2f} seconds**, and the full XGBoost search took **{xgb_seconds:.2f} seconds**. On the untouched hold-out set, Random Forest's RMSE was **{rf_holdout_delta:.2f}% higher** and XGBoost's was **{xgb_holdout_delta:.2f}% higher** than the engineered-feature Ridge baseline. Random Forest adds the largest runtime cost without a hold-out gain; XGBoost is faster in this run, but still has a worse hold-out RMSE and more model complexity. The simpler Ridge baseline is therefore the safer choice based on the evidence available.

The randomized search used **{n_iter} sampled configurations per ensemble** in this reproducible run. Each search spans more than four hyperparameters. This is a useful practical search, but it is not exhaustive; increase `--n-iter` to 10–20 for broader exploration if more compute time is available. The final hold-out set must remain untouched during any expanded search.

## 9. Important interpretation

Random Forest uses **bagging**: many trees are trained on bootstrap samples and their predictions are averaged. This reduces variance and is generally robust, but it can be less effective at extrapolating beyond patterns in the training data.

XGBoost uses **boosting**: trees are built sequentially, with later trees correcting earlier errors. It can model complex interactions and often performs strongly, but it has more tuning choices and can overfit if tree depth, learning rate and regularization are poorly controlled.

A lower score is not automatically a meaningful win. Consider the size and consistency of the CV improvement, the standard deviation across folds, the final hold-out check, and the added training/maintenance cost together.

## 10. Reproducibility

Run:

```bash
pip install -r requirements.txt
python tuned_models.py --data ames_housing_cleaned.csv
```

Generated files are saved in `outputs/`: CV comparison, hold-out results, best parameters, full search results and this report.
"""
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 5 Ames Housing feature engineering and tuned ensembles")
    parser.add_argument("--data", default="ames_housing_cleaned.csv", help="Path to Task 1 cleaned Ames CSV")
    parser.add_argument("--n-iter", type=int, default=4, help="Randomized parameter combinations per ensemble (default: 4; increase for a broader search)")
    parser.add_argument("--jobs", type=int, default=-1, help="Parallel jobs for RandomizedSearchCV (default: -1)")
    parser.add_argument("--output-dir", default="outputs", help="Directory for generated results")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find {data_path}. Put ames_housing_cleaned.csv beside this script "
            "or pass its location with --data."
        )
    if args.n_iter < 1:
        raise ValueError("--n-iter must be at least 1.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(data_path, keep_default_na=False, na_values=["NA", ""])
    X, y = prepare_ames_features(raw)

    # Recreate Task 3's exact split before any learned preprocessing.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    # Inspect deterministic engineered columns only to choose their type lists.
    # This step learns no statistics; all imputers/encoders remain inside CV.
    engineered_train_preview = engineer_features(X_train)
    numeric_columns = engineered_train_preview.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = engineered_train_preview.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"rmse": "neg_root_mean_squared_error", "r2": "r2"}

    # Task 3 baseline, re-run with the engineered features and same 5 folds.
    baseline_estimator = make_estimator(Ridge(alpha=0.1), numeric_columns, categorical_columns)
    start = time.perf_counter()
    baseline_cv = cross_validate(
        baseline_estimator,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=args.jobs,
        return_train_score=False,
    )
    baseline_time = time.perf_counter() - start
    baseline_fold_rmse = -baseline_cv["test_rmse"]

    cv_rows = [{
        "Model": "Task 3 baseline: log-Ridge alpha=0.1",
        "Mean CV RMSE": float(np.mean(baseline_fold_rmse)),
        "Std CV RMSE": float(np.std(baseline_fold_rmse, ddof=1)),
        "Mean CV R2": float(np.mean(baseline_cv["test_r2"])),
        "Training time (s)": float(baseline_time),
        "Note": "Task 3 log-target Ridge; same engineered features and CV folds",
    }]
    cv_fold_results = {"Task 3 baseline": baseline_fold_rmse.tolist()}
    search_times: dict[str, float] = {"Task 3 baseline": float(baseline_time)}
    best_params: dict[str, dict] = {}

    # Random Forest search: six hyperparameters are sampled.
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)
    rf_estimator = make_estimator(rf, numeric_columns, categorical_columns)
    rf_space = {
        "regressor__model__n_estimators": [60, 100, 150, 200],
        "regressor__model__max_depth": [8, 12, 16, 24],
        "regressor__model__min_samples_split": [2, 5, 10],
        "regressor__model__min_samples_leaf": [1, 2, 4],
        "regressor__model__max_features": ["sqrt", 0.5, 0.75],
        "regressor__model__bootstrap": [True, False],
    }
    rf_search = RandomizedSearchCV(
        estimator=rf_estimator,
        param_distributions=rf_space,
        n_iter=args.n_iter,
        scoring=scoring,
        refit="rmse",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=args.jobs,
        verbose=1,
        return_train_score=False,
        error_score="raise",
    )
    start = time.perf_counter()
    rf_search.fit(X_train, y_train)
    rf_time = time.perf_counter() - start
    rf_index = rf_search.best_index_
    rf_fold_rmse = np.array([
        -rf_search.cv_results_[f"split{i}_test_rmse"][rf_index] for i in range(5)
    ], dtype=float)
    cv_rows.append({
        "Model": "Tuned Random Forest",
        "Mean CV RMSE": float(-rf_search.cv_results_["mean_test_rmse"][rf_index]),
        "Std CV RMSE": float(rf_search.cv_results_["std_test_rmse"][rf_index]),
        "Mean CV R2": float(rf_search.cv_results_["mean_test_r2"][rf_index]),
        "Training time (s)": float(rf_time),
        "Note": f"Best of {args.n_iter} randomized configurations; bagging ensemble",
    })
    cv_fold_results["Tuned Random Forest"] = rf_fold_rmse.tolist()
    search_times["Tuned Random Forest"] = float(rf_time)
    best_params["Tuned Random Forest"] = rf_search.best_params_
    pd.DataFrame(rf_search.cv_results_).to_csv(output_dir / "random_forest_search_results.csv", index=False)

    # XGBoost search: eight hyperparameters are sampled.
    xgb = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=RANDOM_STATE,
        n_jobs=1,
        tree_method="hist",
        verbosity=0,
    )
    xgb_estimator = make_estimator(xgb, numeric_columns, categorical_columns)
    xgb_space = {
        "regressor__model__n_estimators": [75, 100, 150, 200],
        "regressor__model__max_depth": [2, 3, 4, 5, 6],
        "regressor__model__learning_rate": [0.02, 0.04, 0.06, 0.08, 0.10],
        "regressor__model__subsample": [0.70, 0.85, 1.0],
        "regressor__model__colsample_bytree": [0.60, 0.80, 1.0],
        "regressor__model__min_child_weight": [1, 3, 5, 7],
        "regressor__model__reg_alpha": [0.0, 0.01, 0.1, 1.0],
        "regressor__model__reg_lambda": [1.0, 3.0, 5.0, 10.0],
    }
    xgb_search = RandomizedSearchCV(
        estimator=xgb_estimator,
        param_distributions=xgb_space,
        n_iter=args.n_iter,
        scoring=scoring,
        refit="rmse",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=args.jobs,
        verbose=1,
        return_train_score=False,
        error_score="raise",
    )
    start = time.perf_counter()
    xgb_search.fit(X_train, y_train)
    xgb_time = time.perf_counter() - start
    xgb_index = xgb_search.best_index_
    xgb_fold_rmse = np.array([
        -xgb_search.cv_results_[f"split{i}_test_rmse"][xgb_index] for i in range(5)
    ], dtype=float)
    cv_rows.append({
        "Model": "Tuned XGBoost",
        "Mean CV RMSE": float(-xgb_search.cv_results_["mean_test_rmse"][xgb_index]),
        "Std CV RMSE": float(xgb_search.cv_results_["std_test_rmse"][xgb_index]),
        "Mean CV R2": float(xgb_search.cv_results_["mean_test_r2"][xgb_index]),
        "Training time (s)": float(xgb_time),
        "Note": f"Best of {args.n_iter} randomized configurations; sequential boosting ensemble",
    })
    cv_fold_results["Tuned XGBoost"] = xgb_fold_rmse.tolist()
    search_times["Tuned XGBoost"] = float(xgb_time)
    best_params["Tuned XGBoost"] = xgb_search.best_params_
    pd.DataFrame(xgb_search.cv_results_).to_csv(output_dir / "xgboost_search_results.csv", index=False)

    cv_table = pd.DataFrame(cv_rows)
    cv_table.to_csv(output_dir / "cv_model_comparison.csv", index=False)
    fold_table = pd.DataFrame({
        "Fold": [1, 2, 3, 4, 5],
        "Task 3 log-Ridge RMSE": cv_fold_results["Task 3 baseline"],
        "Random Forest RMSE": cv_fold_results["Tuned Random Forest"],
        "XGBoost RMSE": cv_fold_results["Tuned XGBoost"],
    })
    fold_table.to_csv(output_dir / "cv_fold_rmse.csv", index=False)

    # Final untouched hold-out evaluation. The test set has not been used by CV
    # or RandomizedSearchCV; each selected model is scored here once.
    baseline_estimator.fit(X_train, y_train)
    selected_estimators = {
        "Task 3 baseline + engineered features": baseline_estimator,
        "Tuned Random Forest": rf_search.best_estimator_,
        "Tuned XGBoost": xgb_search.best_estimator_,
    }
    holdout_rows = []
    for model_name, estimator in selected_estimators.items():
        prediction = estimator.predict(X_test)
        holdout_rows.append({"Model": model_name, **metrics(y_test, prediction)})
    holdout_table = pd.DataFrame(holdout_rows)
    holdout_table.to_csv(output_dir / "holdout_test_results.csv", index=False)

    with open(output_dir / "best_hyperparameters.json", "w", encoding="utf-8") as file:
        json.dump(best_params, file, indent=2, sort_keys=True)

    report = build_report(
        output_dir=output_dir,
        data_shape=raw.shape,
        n_train=len(X_train),
        n_test=len(X_test),
        numeric_count=len(numeric_columns),
        categorical_count=len(categorical_columns),
        cv_table=cv_table,
        holdout_table=holdout_table,
        best_params=best_params,
        cv_fold_results=cv_fold_results,
        search_times=search_times,
        n_iter=args.n_iter,
    )
    report_path = Path("MODEL_COMPARISON.md")
    report_path.write_text(report, encoding="utf-8")
    (output_dir / "MODEL_COMPARISON.md").write_text(report, encoding="utf-8")

    print("\nDATA AND VALIDATION")
    print(f"Dataset shape: {raw.shape}")
    print(f"Train rows: {len(X_train)} | Untouched hold-out rows: {len(X_test)}")
    print(f"Numeric features after engineering: {len(numeric_columns)}")
    print(f"Categorical features after engineering: {len(categorical_columns)}")
    print("\n5-FOLD CV COMPARISON (RMSE in original SalePrice units)")
    print(cv_table.to_string(index=False, formatters={
        "Mean CV RMSE": lambda x: f"{x:,.2f}",
        "Std CV RMSE": lambda x: f"{x:,.2f}",
        "Mean CV R2": lambda x: f"{x:.4f}",
        "Training time (s)": lambda x: f"{x:.2f}",
    }))
    print("\nFINAL UNTOUCHED HOLD-OUT RESULTS")
    print(holdout_table.to_string(index=False, formatters={
        "MAE": lambda x: f"{x:,.2f}",
        "RMSE": lambda x: f"{x:,.2f}",
        "R2": lambda x: f"{x:.4f}",
    }))
    print("\nBest parameters:")
    print(json.dumps(best_params, indent=2, sort_keys=True))
    print(f"\nSaved report: {report_path.resolve()}")
    print(f"Saved outputs in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
