# MODEL_COMPARISON — Feature Engineering & Tuned Ensembles (Ames Housing)

## 1. Objective

This task tests whether domain-driven features and tuned tree ensembles improve house-price prediction beyond Task 3's log-target Ridge baseline. The comparison uses the same 5-fold CV splits for all models and reserves the original Task 3 20% hold-out set for one final evaluation.

## 2. Dataset and leakage controls

- Source: Task 1 cleaned Ames Housing dataset (`ames_housing_cleaned.csv`).
- Dataset shape loaded: **1,460 rows × 89 columns**.
- Target: `SalePrice`; training target is `log1p(SalePrice)` because Task 3 found this transformation useful.
- Task 3 split reproduced first with `train_test_split(test_size=0.20, random_state=42)`: **1,168 training rows** and **292 untouched hold-out rows**.
- Predictors exclude `SalePrice`, `Id`, and the three Task 1 IQR flags. `MSSubClass` and `MoSold` remain categorical.
- The Task 3 missing-value restoration rules are reused. Median/mode imputation, scaling, and one-hot encoding are fitted inside each CV fold's pipeline.
- The engineered features are deterministic calculations from input property and sale-year columns. They do not use `SalePrice` or target-derived statistics.
- The 20% hold-out set is not used for feature selection, hyperparameter tuning, or model selection. It is scored once after CV and tuning are complete.

### Original Task 3 reference

Task 3's best reported model was Ridge Regression on `log1p(SalePrice)` with `alpha=0.1`, with a held-out RMSE of **$24,920.50**. In this task, that same model configuration is re-run with the engineered features and 5-fold CV, so its CV results are directly comparable with the tuned ensembles.

## 3. Engineered features and domain reasoning

Eight new features were created:

- **`TotalLivingArea`:** `GrLivArea + TotalBsmtSF`; combines above-grade living space and basement area into one broad interior-size measure.
- **`HouseAgeAtSale`:** `YrSold - YearBuilt`; older homes can differ in condition, materials, maintenance and buyer appeal.
- **`RemodelAgeAtSale`:** `YrSold - YearRemodAdd`; captures how recently the property was updated.
- **`TotalBathrooms`:** Full bathrooms plus half bathrooms weighted as 0.5, including basement bathrooms; summarizes bathroom capacity.
- **`HasRemodeled`:** 1 when `YearRemodAdd > YearBuilt`; separates homes with a recorded remodel from homes without one.
- **`TotalPorchSF`:** Adds open, enclosed, three-season and screen porch areas to represent combined porch space.
- **`GarageAgeAtSale`:** `YrSold - GarageYrBlt`; captures garage age. Missing values are retained for fold-specific imputation.
- **`HasGarage`:** 1 when `GarageArea > 0`; helps distinguish no garage from an unknown garage year.

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

| Model | Mean CV RMSE | Std CV RMSE | Mean CV R2 | Training time (s) | Note |
|---|---|---|---|---|---|
| Task 3 baseline: log-Ridge alpha=0.1 | 46,140.8359 | 31,859.6271 | 0.5402 | 4.1978 | Task 3 log-target Ridge; same engineered features and CV folds |
| Tuned Random Forest | 29,619.4140 | 4,711.1061 | 0.8508 | 21.7801 | Best of 4 randomized configurations; bagging ensemble |
| Tuned XGBoost | 27,829.0610 | 3,900.9852 | 0.8695 | 4.3409 | Best of 4 randomized configurations; sequential boosting ensemble |

### RMSE on each of the same five folds

| Fold | Task 3 log-Ridge RMSE | Random Forest RMSE | XGBoost RMSE |
|---|---|---|---|
| 1 | 25,616.7457 | 28,387.1781 | 30,727.6686 |
| 2 | 46,318.7543 | 34,757.1677 | 28,911.7815 |
| 3 | 101,196.6156 | 35,516.8648 | 32,814.8428 |
| 4 | 27,149.5579 | 25,056.0582 | 22,493.5476 |
| 5 | 30,422.5059 | 24,379.8012 | 24,197.4645 |

The Task 3 Ridge baseline has a notably large fold-to-fold spread. One fold is much harder for Ridge than the others, which raises its mean and standard deviation. This is why the CV summary is reported as mean ± standard deviation rather than as a single number.

## 6. Best hyperparameters

### Tuned Random Forest

```json
{
  "regressor__model__bootstrap": true,
  "regressor__model__max_depth": 16,
  "regressor__model__max_features": 0.5,
  "regressor__model__min_samples_leaf": 2,
  "regressor__model__min_samples_split": 5,
  "regressor__model__n_estimators": 150
}
```
### Tuned XGBoost

```json
{
  "regressor__model__colsample_bytree": 1.0,
  "regressor__model__learning_rate": 0.1,
  "regressor__model__max_depth": 5,
  "regressor__model__min_child_weight": 3,
  "regressor__model__n_estimators": 200,
  "regressor__model__reg_alpha": 0.01,
  "regressor__model__reg_lambda": 10.0,
  "regressor__model__subsample": 0.85
}
```

## 7. Final untouched hold-out evaluation

After selecting the models using training-only CV, each selected estimator was evaluated on the original Task 3 20% hold-out set. This table is a final check, not a tuning input.

| Model | MAE | RMSE | R2 |
|---|---|---|---|
| Task 3 baseline + engineered features | 16,060.4760 | 24,937.5639 | 0.9189 |
| Tuned Random Forest | 17,236.9027 | 30,074.5233 | 0.8821 |
| Tuned XGBoost | 15,913.8705 | 27,455.8262 | 0.9017 |

The historical Task 3 hold-out RMSE was **$24,920.50**. The current engineered-feature Ridge result is **$24,937.56**, a **0.07% higher** error. This is effectively unchanged, so the engineered features alone did not improve the original Task 3 Ridge result on this hold-out set.

## 8. Verdict: what would I ship?

**Ship the simpler Task 3 baseline: log-Ridge (alpha=0.1).** Although Tuned XGBoost has the lowest ensemble CV RMSE, the hold-out RMSE is 10.10% higher than the engineered-feature Ridge baseline. The ensemble is not justified by the current evidence.

The project's practical-improvement rule is a **2% reduction in mean CV RMSE**, with improvement on at least 4 of the 5 paired folds, and no more than a 2% deterioration on the untouched hold-out set. This is a transparent project decision rule, not a formal statistical significance test. Five folds provide a more reliable estimate than one split, but they do not prove statistical significance on their own.

- Best ensemble by CV RMSE: **Tuned XGBoost**.
- CV RMSE reduction versus engineered-feature Ridge: **39.69%**.
- Paired folds where the ensemble has lower RMSE: **4/5**.
- Hold-out RMSE change versus engineered-feature Ridge: **-10.10%** (positive means lower error).
- Recommended model: **Task 3 baseline: log-Ridge (alpha=0.1)**.

### Complexity and runtime trade-off

In this run, the 5-fold baseline took **4.20 seconds**, the full Random Forest search took **21.78 seconds**, and the full XGBoost search took **4.34 seconds**. On the untouched hold-out set, Random Forest's RMSE was **20.60% higher** and XGBoost's was **10.10% higher** than the engineered-feature Ridge baseline. Random Forest adds the largest runtime cost without a hold-out gain; XGBoost is faster in this run, but still has a worse hold-out RMSE and more model complexity. The simpler Ridge baseline is therefore the safer choice based on the evidence available.

The randomized search used **4 sampled configurations per ensemble** in this reproducible run. Each search spans more than four hyperparameters. This is a useful practical search, but it is not exhaustive; increase `--n-iter` to 10–20 for broader exploration if more compute time is available. The final hold-out set must remain untouched during any expanded search.

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
