"""Task 5: leakage-safe, domain-driven feature engineering for Ames Housing.

All engineered features are deterministic calculations from property/sale-year
columns. No feature uses SalePrice or any statistic calculated from the target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# These are the source columns needed to construct the engineered features.
REQUIRED_SOURCE_COLUMNS = [
    "GrLivArea", "TotalBsmtSF", "YrSold", "YearBuilt", "YearRemodAdd",
    "FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath",
    "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch",
    "GarageYrBlt", "GarageArea",
]


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Return X with eight deterministic, target-independent features added.

    This function is placed inside the sklearn Pipeline. It does not learn
    medians, categories, thresholds, or any other values from the dataset.
    Missing values are left missing for the fold-specific imputers to handle.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("engineer_features expects a pandas DataFrame.")

    missing = [column for column in REQUIRED_SOURCE_COLUMNS if column not in X.columns]
    if missing:
        raise ValueError(
            "The Ames dataset is missing columns required for feature engineering: "
            + ", ".join(missing)
        )

    out = X.copy()

    def numeric(column: str) -> pd.Series:
        return pd.to_numeric(out[column], errors="coerce")

    # 1. Above-grade living area plus basement area: total interior area.
    out["TotalLivingArea"] = numeric("GrLivArea") + numeric("TotalBsmtSF")

    # 2. Approximate property age at the recorded sale year.
    out["HouseAgeAtSale"] = numeric("YrSold") - numeric("YearBuilt")

    # 3. Years since the most recent recorded remodel at sale time.
    out["RemodelAgeAtSale"] = numeric("YrSold") - numeric("YearRemodAdd")

    # 4. Bathroom equivalents: a half bath contributes 0.5.
    out["TotalBathrooms"] = (
        numeric("FullBath")
        + 0.5 * numeric("HalfBath")
        + numeric("BsmtFullBath")
        + 0.5 * numeric("BsmtHalfBath")
    )

    # 5. Whether a remodel year later than the original build year is recorded.
    built = numeric("YearBuilt")
    remodel = numeric("YearRemodAdd")
    remodeled = pd.Series(np.nan, index=out.index, dtype="float64")
    valid_years = built.notna() & remodel.notna()
    remodeled.loc[valid_years] = (remodel.loc[valid_years] > built.loc[valid_years]).astype(float)
    out["HasRemodeled"] = remodeled

    # 6. Combined porch area (not including the separate wood-deck field).
    porch_columns = ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
    porch_values = out[porch_columns].apply(pd.to_numeric, errors="coerce")
    out["TotalPorchSF"] = porch_values.sum(axis=1, min_count=len(porch_columns))

    # 7. Garage age at sale; homes without a recorded garage year remain missing.
    out["GarageAgeAtSale"] = numeric("YrSold") - numeric("GarageYrBlt")

    # 8. A separate garage-presence indicator helps distinguish no garage from
    # an unknown garage construction year. Missing GarageArea remains missing.
    garage_area = numeric("GarageArea")
    has_garage = pd.Series(np.nan, index=out.index, dtype="float64")
    valid_garage = garage_area.notna()
    has_garage.loc[valid_garage] = (garage_area.loc[valid_garage] > 0).astype(float)
    out["HasGarage"] = has_garage

    return out


def prepare_ames_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Apply the same Task 3 data restoration/exclusion rules to Task 1 data."""
    data = df.copy()

    if "SalePrice" not in data.columns:
        raise ValueError("The dataset must contain the SalePrice target column.")

    data["SalePrice"] = pd.to_numeric(data["SalePrice"], errors="coerce")
    data = data.dropna(subset=["SalePrice"]).copy()
    y = data["SalePrice"].astype(float)

    # Task 1's cleaned file contains values imputed with full-dataset statistics.
    # Restore original missing cells using its saved row-level missing indicators,
    # so the modelling pipeline can impute them within each training fold.
    for value_column, flag_column in [
        ("LotFrontage", "LotFrontage_missing"),
        ("Electrical", "Electrical_missing"),
    ]:
        if value_column in data.columns and flag_column in data.columns:
            flag = pd.to_numeric(data[flag_column], errors="coerce").eq(1)
            data.loc[flag, value_column] = np.nan

    # Preserve meaningful "None" values when zero veneer area confirms no veneer.
    if "MasVnrType_missing" in data.columns and "MasVnrArea" in data.columns:
        flag = pd.to_numeric(data["MasVnrType_missing"], errors="coerce").eq(1)
        area = pd.to_numeric(data["MasVnrArea"], errors="coerce")
        restore_type = flag & ~area.eq(0)
        data.loc[restore_type, "MasVnrType"] = np.nan

    if "MasVnrArea_missing" in data.columns and "MasVnrType" in data.columns:
        flag = pd.to_numeric(data["MasVnrArea_missing"], errors="coerce").eq(1)
        no_veneer_type = data["MasVnrType"].eq("None")
        restore_area = flag & ~no_veneer_type
        data.loc[restore_area, "MasVnrArea"] = np.nan

    # Drop target, identifier, and Task 1 IQR flags (their thresholds were
    # calculated before the split and are not safe modelling predictors).
    drop_columns = [
        "SalePrice", "Id", "GrLivArea_outlier", "LotArea_outlier", "GarageArea_outlier"
    ]
    X = data.drop(columns=[c for c in drop_columns if c in data.columns]).copy()

    # Restore Task 1's semantic treatment of numeric-looking category codes.
    for column in ["MSSubClass", "MoSold"]:
        if column in X.columns:
            X[column] = X[column].astype("category")

    return X, y
