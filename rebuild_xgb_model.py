"""
Rebuild artifacts/xgb_model.joblib locally.

The GitHub copy of xgb_model.joblib fails to load on Windows with
`XGBoostError: input stream corrupted`. This script retrains the model exactly
as the notebook does (same split, features, Optuna-selected hyperparameters
from artifacts/config.json, asymmetric objective, early stopping) and saves it.

Run from the project folder:
    python rebuild_xgb_model.py
"""

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit


def asymmetric_mse_objective(y_true, y_pred):
    """
    Custom objective for gradient boosting: penalizes underprediction.

    If the model predicts a tire is faster than reality (smaller delta), the strategist
    keeps the driver out too long, resulting in exponential time loss.
    Overprediction (early pit) is less costly.
    """
    residual = y_pred - y_true
    alpha_under = 2.5
    alpha_over = 1.0

    grad = np.where(residual < 0, alpha_under * residual, alpha_over * residual)
    hess = np.where(residual < 0, alpha_under * np.ones_like(residual), alpha_over * np.ones_like(residual))

    return grad, hess


ART = Path(__file__).resolve().parent / "artifacts"


def main():
    with open(ART / "config.json") as f:
        cfg = json.load(f)
    laps_df = pd.read_parquet(ART / "laps_df.parquet")

    # Group-aware train/test split (identical to the notebook)
    features = cfg["features"]
    target = "LapTimeDelta"
    laps_df["StintID"] = laps_df["Driver"] + "_" + laps_df["Stint"].astype(str)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in gss.split(laps_df, groups=laps_df["StintID"]):
        train_df = laps_df.iloc[train_idx].copy()
        test_df = laps_df.iloc[test_idx].copy()

    X_train = train_df[features].values
    y_train = train_df[target].values
    X_test = test_df[features].values
    y_test = test_df[target].values
    print(f"Train laps: {len(X_train)} | Test laps: {len(X_test)} "
          f"(config: {cfg['n_train']} / {cfg['n_test']})")

    # Train final XGBoost (identical to the notebook)
    val_size = int(0.1 * len(X_train))
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train_final, y_train_final = X_train[:-val_size], y_train[:-val_size]

    xgb_model = xgb.XGBRegressor(
        **cfg["best_xgb_params"],
        objective=asymmetric_mse_objective,
        random_state=42,
        early_stopping_rounds=20,
    )
    xgb_model.fit(X_train_final, y_train_final, eval_set=[(X_val, y_val)], verbose=False)

    xgb_pred_test = xgb_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, xgb_pred_test))
    residual_std = np.std(y_test - xgb_pred_test)
    print(f"Test RMSE: {rmse:.4f} s (expected ~0.663) | residual std: {residual_std:.4f} "
          f"(config: {cfg['residual_std']:.4f})")

    target_path = ART / "xgb_model.joblib"
    backup_path = ART / "xgb_model_original.joblib"
    if target_path.exists() and not backup_path.exists():
        shutil.copy2(target_path, backup_path)
        print(f"Backed up original model to {backup_path.name}")

    joblib.dump(xgb_model, target_path)
    print(f"Saved rebuilt model to {target_path.name}")

    # Round-trip check
    reloaded = joblib.load(target_path)
    assert np.allclose(reloaded.predict(X_test), xgb_pred_test)
    print("Reload check passed.")


if __name__ == "__main__":
    main()
