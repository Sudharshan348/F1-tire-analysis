"""
Generate the figures and prediction tables used in the final report.

Reads only the saved artifacts in ../artifacts (no retraining) and writes
PNG figures plus small JSON tables into report/figures and report/tables.

Run from the project folder:
    python report/make_figures.py
"""

import json
from pathlib import Path

import __main__
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.model_selection import GroupShuffleSplit


def asymmetric_mse_objective(y_true, y_pred):
    residual = y_pred - y_true
    alpha_under = 2.5
    alpha_over = 1.0
    grad = np.where(residual < 0, alpha_under * residual, alpha_over * residual)
    hess = np.where(residual < 0, alpha_under * np.ones_like(residual), alpha_over * np.ones_like(residual))
    return grad, hess


__main__.asymmetric_mse_objective = asymmetric_mse_objective

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
FIG = ROOT / "report" / "figures"
TAB = ROOT / "report" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

ACCENT = "#1f6fd1"
GREY = "#8c8c8c"
DARK = "#222222"
SOFT_C = "#d62728"
HARD_C = "#444444"
MED_C = "#e6b800"
MODEL_COLORS = {"Linear": "#9e9e9e", "Poly": "#bdbdbd", "XGB": "#ff7f0e", "RF": "#2ca02c", "GP": ACCENT}

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

cfg = json.loads((ART / "config.json").read_text())
opt = cfg["OPTIMIZATION"]
laps = pd.read_parquet(ART / "laps_df.parquet")
features = cfg["features"]


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- Fig 1: pipeline
def fig_pipeline():
    steps = [
        ("FastF1 data", "Bahrain GP 2023\nlaps + weather"),
        ("Cleaning", "SC, pit, dirty-air\nlaps, outliers out"),
        ("Fuel\ncorrection", "0.055 s/kg\n→ LapTimeDelta"),
        ("5 ML models", "Linear, Poly,\nXGB, RF, GP"),
        ("Monte Carlo", "Race simulator\n2 engines"),
        ("Differential\nEvolution", "Optimal pit lap\n(laps 31–56)"),
        ("Dashboard", "Streamlit\nlive what-if"),
    ]
    fig, ax = plt.subplots(figsize=(10, 2.3))
    ax.set_xlim(0, len(steps) * 1.45)
    ax.set_ylim(0, 2.2)
    ax.axis("off")
    for i, (name, sub) in enumerate(steps):
        x = 0.1 + i * 1.45
        hl = name.startswith("Differential")
        name_y = 1.45 if "\n" in name else 1.5
        box = FancyBboxPatch((x, 0.35), 1.2, 1.5, boxstyle="round,pad=0.02,rounding_size=0.08",
                             fc="#e8f0fb" if hl else "white", ec=ACCENT if hl else DARK, lw=1.6 if hl else 1)
        ax.add_patch(box)
        ax.text(x + 0.6, name_y, name, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x + 0.6, 0.78, sub, ha="center", va="center", fontsize=8.5, color="#333333")
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 1.45, 1.1), xytext=(x + 1.2, 1.1),
                        arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1))
    save(fig, "fig01_pipeline.png")


# ---------------------------------------------------------------- Fig 2: degradation
def fig_degradation():
    low = pd.read_csv(ART / "lowess_curves.csv").drop_duplicates()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for comp, c in [("SOFT", SOFT_C), ("MEDIUM", MED_C), ("HARD", HARD_C)]:
        d = laps[laps["Compound"] == comp]
        if d.empty:
            continue
        ax.scatter(d["TyreLife"], d["LapTimeDelta"], s=9, alpha=0.25, color=c)
        curve = low[low["Compound"] == comp].sort_values("TyreLife")
        if len(curve):
            ax.plot(curve["TyreLife"], curve["SmoothedDelta"], color=c, lw=2.4,
                    label=f"{comp.title()} ({len(d)} laps)")
    ax.set_xlabel("Tyre age (laps)")
    ax.set_ylabel("Lap-time loss vs stint best (s)")
    ax.set_title("Lap time rises steadily with tyre age; softs pull away after ~18 laps")
    ax.legend(frameon=False)
    save(fig, "fig02_degradation.png")


# ---------------------------------------------------------------- split + predictions
def test_split():
    df = laps.copy()
    df["StintID"] = df["Driver"] + "_" + df["Stint"].astype(str)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["StintID"]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def load_models():
    return {k: joblib.load(ART / f"{v}.joblib") for k, v in {
        "linear": "linear_model", "poly": "poly_model", "poly_t": "poly_transformer",
        "xgb": "xgb_model", "rf": "rf_model", "gp": "gp_model", "gp_s": "gp_scaler"}.items()}


def predict_all(m, X):
    gp_mean, gp_std = m["gp"].predict(m["gp_s"].transform(X), return_std=True)
    return {
        "Linear": m["linear"].predict(X),
        "Poly": m["poly"].predict(np.hstack([m["poly_t"].transform(X[:, :3]), X[:, 3:]])),
        "XGB": m["xgb"].predict(X),
        "RF": m["rf"].predict(X),
        "GP": gp_mean,
    }, gp_std


def fig_and_tables_predictions(models):
    _, test = test_split()
    X = test[features].values
    y = test["LapTimeDelta"].values
    preds, gp_std = predict_all(models, X)

    # Fig 3: predicted vs actual for the two simulation engines
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
    for ax, key, name in [(axes[0], "XGB", "XGBoost (asymmetric loss)"), (axes[1], "GP", "Gaussian Process")]:
        p = preds[key]
        rmse = float(np.sqrt(np.mean((y - p) ** 2)))
        ax.scatter(y, p, s=14, alpha=0.6, color=MODEL_COLORS[key])
        lim = [min(y.min(), p.min()) - 0.2, max(y.max(), p.max()) + 0.2]
        ax.plot(lim, lim, color=GREY, lw=1, ls="--")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(f"{name}\nRMSE {rmse:.3f} s on 112 unseen laps")
        ax.set_xlabel("Actual LapTimeDelta (s)")
    axes[0].set_ylabel("Predicted LapTimeDelta (s)")
    save(fig, "fig03_pred_vs_actual.png")

    # Table: sample prediction output (8 test laps across compounds/ages)
    test = test.assign(**{f"pred_{k}": v for k, v in preds.items()}, gp_std=gp_std)
    sample = (test.sort_values("TyreLife")
                  .groupby("Compound", group_keys=False)
                  .apply(lambda g: g.iloc[np.linspace(0, len(g) - 1, min(4, len(g))).astype(int)]))
    sample = sample.sort_values(["Compound", "TyreLife"]).head(8)
    rows = []
    for _, r in sample.iterrows():
        rows.append({
            "Driver": r["Driver"], "Lap": int(r["LapNumber"]), "Compound": r["Compound"].title(),
            "TyreAge": int(r["TyreLife"]), "Actual": round(float(r["LapTimeDelta"]), 3),
            "Linear": round(float(r["pred_Linear"]), 3), "Poly": round(float(r["pred_Poly"]), 3),
            "XGB": round(float(r["pred_XGB"]), 3), "RF": round(float(r["pred_RF"]), 3),
            "GP": round(float(r["pred_GP"]), 3), "GPstd": round(float(r["gp_std"]), 3),
        })
    (TAB / "sample_predictions.json").write_text(json.dumps(rows, indent=1))
    print("saved sample_predictions.json")

    # Table: predicted degradation by tyre age (scenario forecast)
    scen = []
    for comp_name, enc in [("Soft", 0), ("Hard", 2)]:
        for age in [1, 5, 10, 15, 20, 25]:
            x = np.array([[age, age ** 2, age ** 3, enc, cfg["MEAN_TRACK_TEMP"], cfg["MEAN_AIR_TEMP"], 40]])
            p, s = predict_all(models, x)
            scen.append({"Compound": comp_name, "TyreAge": age, "XGB": round(float(p["XGB"][0]), 3),
                         "GP": round(float(p["GP"][0]), 3), "GPstd": round(float(s[0]), 3)})
    (TAB / "scenario_predictions.json").write_text(json.dumps(scen, indent=1))
    print("saved scenario_predictions.json")


# ---------------------------------------------------------------- Fig 4: model metrics
def fig_metrics():
    m = pd.read_csv(ART / "metrics_table.csv")
    short = ["Linear", "Poly (deg 2)", "XGBoost", "Random Forest", "Gaussian Proc."]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, col, title, best_fn in [
        (axes[0], "RMSE (s)", "RMSE (s): Random Forest lowest", np.argmin),
        (axes[1], "SIS (s)", "Strategy Impact Score (s/stint): GP lowest", np.argmin),
    ]:
        vals = m[col].values
        best = best_fn(vals)
        colors = [ACCENT if i == best else "#c7c7c7" for i in range(len(vals))]
        bars = ax.bar(short, vals, color=colors)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.01, f"{v:.3f}" if col == "RMSE (s)" else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.set_ylim(0, vals.max() * 1.18)
    save(fig, "fig04_model_metrics.png")


# ---------------------------------------------------------------- Fig 5: PER stint trace
def fig_stint_trace():
    t = pd.read_csv(ART / "per_stint_trace.csv")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(t["LapNumber"], t["Actual"], color=DARK, lw=2, marker="o", ms=4, label="Actual (fuel-corrected)")
    ax.plot(t["LapNumber"], t["XGB"], color=MODEL_COLORS["XGB"], lw=1.6, label="XGBoost")
    ax.plot(t["LapNumber"], t["GP"], color=ACCENT, lw=1.6, label="Gaussian Process")
    ax.plot(t["LapNumber"], t["RF"], color=MODEL_COLORS["RF"], lw=1.2, ls="--", label="Random Forest")
    ax.set_xlabel("Lap number")
    ax.set_ylabel("Fuel-corrected lap time (s)")
    ax.set_title("Model tracks Perez's final stint lap by lap (laps 36–57)")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    save(fig, "fig05_per_stint.png")


# ---------------------------------------------------------------- Fig 6: MC grid
def fig_mc_grid():
    w = pd.read_csv(ART / "win_probability_comparison.csv")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(len(w))
    ax.bar(x - 0.2, w["WinPct_XGB"] * 100, 0.4, color=MODEL_COLORS["XGB"], label="XGBoost engine")
    ax.bar(x + 0.2, w["WinPct_GP"] * 100, 0.4, color=ACCENT, label="Gaussian Process engine")
    for i, r in w.iterrows():
        ax.text(i - 0.2, r["WinPct_XGB"] * 100 + 1.5, f"{r['WinPct_XGB']*100:.1f}%", ha="center", fontsize=8.5)
        ax.text(i + 0.2, r["WinPct_GP"] * 100 + 1.5, f"{r['WinPct_GP']*100:.1f}%", ha="center", fontsize=8.5)
    ax.set_xticks(x, [f"Lap {int(l)}" for l in w["ActualPitLap"]])
    ax.set_ylabel("Win probability (%)")
    ax.set_ylim(0, 112)
    ax.set_title("Review 2 baseline: 4-point grid, 10,000 races per strategy")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    save(fig, "fig06_mc_grid.png")


# ---------------------------------------------------------------- Fig 7: DE landscape
def fig_landscape():
    land = pd.read_csv(ART / "optimization_landscape.csv")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.axvspan(42, 51, color="#efefef", zorder=0)
    ax.text(46.5, 103, "Old grid window (42–51)", ha="center", fontsize=9, color="#555555")
    ax.plot(land["PitLap"], land["Ensemble_WinPct"] * 100, color=GREY, lw=2, marker="o", ms=3.5,
            label="Ensemble win probability")
    ax.plot(land["PitLap"], land["XGB_WinPct"] * 100, color=MODEL_COLORS["XGB"], lw=1, ls=":", label="XGBoost engine")
    ax.plot(land["PitLap"], land["GP_WinPct"] * 100, color=ACCENT, lw=1, ls=":", label="GP engine")
    grid = land[land["PitLap"].isin([42, 45, 48, 51])]
    ax.scatter(grid["PitLap"], grid["Ensemble_WinPct"] * 100, s=60, facecolor="white", edgecolor=DARK, zorder=3,
               label="Grid candidates")
    best = land.loc[land["PitLap"] == opt["best_pit_lap"]].iloc[0]
    ax.scatter([best["PitLap"]], [best["Ensemble_WinPct"] * 100], s=90, color=ACCENT, zorder=4)
    ax.annotate(f"DE optimum: lap {int(best['PitLap'])}\n{best['Ensemble_WinPct']*100:.2f}%",
                (best["PitLap"], best["Ensemble_WinPct"] * 100), xytext=(31.2, 62),
                arrowprops=dict(arrowstyle="->", color=DARK), fontsize=10, fontweight="bold")
    g42 = land.loc[land["PitLap"] == 42].iloc[0]
    ax.annotate(f"Grid best: lap 42\n{g42['Ensemble_WinPct']*100:.2f}%", (42, g42["Ensemble_WinPct"] * 100),
                xytext=(43.5, 86), arrowprops=dict(arrowstyle="->", color=DARK), fontsize=9.5)
    ax.set_xlabel("Pit lap (current lap 30)")
    ax.set_ylabel("Win probability (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Full-window landscape: the peak is at lap 34, outside the old grid")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    save(fig, "fig07_landscape.png")


# ---------------------------------------------------------------- Fig 8: algorithm benchmark
def fig_benchmark():
    c = pd.read_csv(ART / "optimization_comparison.csv")
    labels = ["Grid search\n(4 points)", "Golden-section\n(full window)", "Golden-section\n(old bracket)",
              "Newton–Raphson\n(from lap 42)", "Differential\nEvolution"]
    vals = c["Ensemble_WinPct"].values * 100
    colors = ["#c7c7c7", "#c7c7c7", "#c7c7c7", "#e08585", ACCENT]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.bar(labels, vals, color=colors)
    for b, v, lap in zip(bars, vals, c["BestPitLap"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"lap {int(lap)}\n{v:.2f}%", ha="center", fontsize=9)
    ax.set_ylabel("Ensemble win probability (%)")
    ax.set_ylim(0, 118)
    ax.set_title("Only DE (and golden-section given the right bracket) reaches lap 34")
    save(fig, "fig08_benchmark.png")


# ---------------------------------------------------------------- Fig 9: validation
def fig_validation():
    v = pd.read_csv(ART / "optimization_validation.csv")
    order = ["Grid best (baseline)", "XGB-only optimum", "GP-only optimum", "DE optimum"]
    v = v.set_index("Scenario").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 3.3))
    colors = ["#c7c7c7", "#c7c7c7", "#c7c7c7", ACCENT]
    y = np.arange(len(v))
    ax.barh(y, v["Ensemble_WinPct"] * 100, xerr=v["Ensemble_WinPct_CI"] * 100, color=colors)
    for i, r in v.iterrows():
        ax.text(r["Ensemble_WinPct"] * 100 + 1, i, f"lap {int(r['PitLap'])}: {r['Ensemble_WinPct']*100:.2f}%",
                va="center", fontsize=9.5)
    ax.set_yticks(y, order)
    ax.set_xlim(0, 122)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Ensemble win probability (%), 5 fresh seeds × 20,000 races, 95% CI")
    ax.set_title("Independent validation confirms the DE optimum")
    save(fig, "fig09_validation.png")


if __name__ == "__main__":
    models = load_models()
    fig_pipeline()
    fig_degradation()
    fig_and_tables_predictions(models)
    fig_metrics()
    fig_stint_trace()
    fig_mc_grid()
    fig_landscape()
    fig_benchmark()
    fig_validation()
