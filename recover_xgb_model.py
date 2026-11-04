"""
Recover the ORIGINAL trained XGBoost model so it loads on Windows.

Why: the GitHub copy of artifacts/xgb_model.joblib was pickled on Linux. Its
booster blob holds two parts, "Config" (runtime state, including the RNG state)
and "Model" (the trees). Windows XGBoost cannot parse the Linux "Config" part
and raises `XGBoostError: input stream corrupted`, but the trees are intact.

This script unpickles artifacts/xgb_model_original.joblib, decodes the booster's
UBJSON blob in pure Python, loads only the "Model" part (identical trees,
identical predictions), and re-saves the estimator as artifacts/xgb_model.joblib.
It then checks the test RMSE against artifacts/metrics_table.csv.

Run from the project folder (after rebuild_xgb_model.py has made the backup):
    python recover_xgb_model.py
"""

import json
import shutil
import struct
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

# ---------------------------------------------------------------- UBJSON decoder
_NUM = {b"i": (">b", 1), b"U": (">B", 1), b"I": (">h", 2), b"l": (">i", 4),
        b"L": (">q", 8), b"d": (">f", 4), b"D": (">d", 8)}


class _UBJReader:
    def __init__(self, data):
        self.d = bytes(data)
        self.p = 0

    def _type(self):
        c = self.d[self.p:self.p + 1]
        self.p += 1
        return c

    def _num(self, c):
        fmt, n = _NUM[c]
        v = struct.unpack(fmt, self.d[self.p:self.p + n])[0]
        self.p += n
        return v

    def _str(self):
        n = self._num(self._type())
        s = self.d[self.p:self.p + n].decode("utf-8")
        self.p += n
        return s

    def value(self, c=None):
        c = c or self._type()
        if c in _NUM:
            return self._num(c)
        if c == b"S":
            return self._str()
        if c in (b"T", b"F", b"Z"):
            return {b"T": True, b"F": False, b"Z": None}[c]
        if c == b"{":
            out = {}
            while self.d[self.p:self.p + 1] != b"}":
                k = self._str()
                out[k] = self.value()
            self.p += 1
            return out
        if c == b"[":
            typ = cnt = None
            if self.d[self.p:self.p + 1] == b"$":
                self.p += 1
                typ = self._type()
            if self.d[self.p:self.p + 1] == b"#":
                self.p += 1
                cnt = self._num(self._type())
            if cnt is not None:
                if typ in _NUM:
                    fmt, n = _NUM[typ]
                    vals = list(struct.unpack(">" + fmt[1] * cnt, self.d[self.p:self.p + n * cnt]))
                    self.p += n * cnt
                    return vals
                return [self.value(typ) for _ in range(cnt)]
            out = []
            while self.d[self.p:self.p + 1] != b"]":
                out.append(self.value())
            self.p += 1
            return out
        raise ValueError(f"Unexpected UBJSON type {c!r} at byte {self.p - 1}")


def _booster_setstate_model_only(self, state):
    """Replacement for Booster.__setstate__ that skips the unreadable Config part."""
    blob = _UBJReader(state["handle"]).value()
    booster = xgb.Booster()
    booster.load_model(bytearray(json.dumps(blob["Model"]).encode("utf-8")))
    self.__dict__.update(booster.__dict__)
    booster.handle = None  # ownership moved to self


def main():
    original = ART / "xgb_model_original.joblib"
    target = ART / "xgb_model.joblib"
    if not original.exists():
        shutil.copy2(target, original)
        print(f"Backed up {target.name} to {original.name}")

    patched = xgb.core.Booster.__setstate__
    xgb.core.Booster.__setstate__ = _booster_setstate_model_only
    try:
        model = joblib.load(original)
    finally:
        xgb.core.Booster.__setstate__ = patched

    # Verify against the notebook's recorded test metrics
    with open(ART / "config.json") as f:
        cfg = json.load(f)
    laps_df = pd.read_parquet(ART / "laps_df.parquet")
    laps_df["StintID"] = laps_df["Driver"] + "_" + laps_df["Stint"].astype(str)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx = next(gss.split(laps_df, groups=laps_df["StintID"]))
    test_df = laps_df.iloc[test_idx]
    pred = model.predict(test_df[cfg["features"]].values)
    rmse = np.sqrt(mean_squared_error(test_df["LapTimeDelta"].values, pred))
    recorded = pd.read_csv(ART / "metrics_table.csv").set_index("Model").loc["XGBoost (Asymmetric)", "RMSE (s)"]
    print(f"Recovered model test RMSE: {rmse:.6f} s | recorded in metrics_table.csv: {recorded:.6f} s")
    if not np.isclose(rmse, recorded, atol=1e-5):
        raise SystemExit("RMSE does not match the recorded value; not saving.")

    rebuilt = ART / "xgb_model_rebuilt.joblib"
    if target.exists() and not rebuilt.exists():
        shutil.copy2(target, rebuilt)
        print(f"Kept the retrained model from rebuild_xgb_model.py as {rebuilt.name}")

    joblib.dump(model, target)
    reloaded = joblib.load(target)
    assert np.allclose(reloaded.predict(test_df[cfg["features"]].values), pred)
    print(f"Saved the original model (Windows-loadable) to {target.name}. Reload check passed.")


if __name__ == "__main__":
    main()
