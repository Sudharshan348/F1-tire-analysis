import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
ART = BASE_DIR / "artifacts"

SKY = "#8ecae6"
BLUE = "#219ebc"
DEEP = "#023047"
AMBER = "#ffb703"
ORANGE = "#fb8500"

COMPOUND_COLORS = {"SOFT": ORANGE, "MEDIUM": AMBER, "HARD": SKY}
ENGINE_COLORS = {"XGBoost": ORANGE, "GaussianProcess": BLUE}
FEATURES = ["TyreLife", "TyreAgeSq", "TyreAgeCubed", "CompoundEncoded",
            "TrackTemp", "AirTemp", "LapNumber"]

st.set_page_config(page_title="F1 Tire Degradation — Pit Wall Analytics",
                   page_icon="🏁", layout="wide")

st.markdown(f"""
<style>
    h1, h2, h3 {{ color: {BLUE}; }}
    div[data-testid="stMetric"] {{
        background-color: {DEEP};
        border: 1px solid rgba(33,158,188,0.35);
        border-radius: 10px; padding: 12px 16px;
    }}
    div[data-testid="stSidebar"] * {{ color: {SKY}; }}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    return {
        "config": json.loads((ART / "config.json").read_text()),
        "laps": pd.read_parquet(ART / "laps_df.parquet"),
        "metrics": pd.read_csv(ART / "metrics_table.csv"),
        "poly_sel": pd.read_csv(ART / "poly_model_selection.csv"),
        "sim_xgb": pd.read_parquet(ART / "sim_xgb.parquet"),
        "sim_gp": pd.read_parquet(ART / "sim_gp.parquet"),
        "win_cmp": pd.read_csv(ART / "win_probability_comparison.csv"),
        "win_tables": pd.read_csv(ART / "win_tables.csv"),
        "lowess": pd.read_csv(ART / "lowess_curves.csv"),
        "conditions": pd.read_parquet(ART / "lap_conditions.parquet"),
        "heatmap": pd.read_csv(ART / "temp_heatmap.csv"),
        "trace": pd.read_csv(ART / "per_stint_trace.csv"),
        "pits": pd.read_csv(ART / "real_pit_stops.csv"),
    }


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


@st.cache_resource
def load_models():
    return {
        "linear": joblib.load(ART / "linear_model.joblib"),
        "poly_model": joblib.load(ART / "poly_model.joblib"),
        "poly_transformer": joblib.load(ART / "poly_transformer.joblib"),
        "xgb": joblib.load(ART / "xgb_model.joblib"),
        "rf": joblib.load(ART / "rf_model.joblib"),
        "gp": joblib.load(ART / "gp_model.joblib"),
        "gp_scaler": joblib.load(ART / "gp_scaler.joblib"),
    }


def base_layout(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=SKY,
        colorway=[ORANGE, AMBER, SKY, BLUE, "#7E57C2"],
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_xaxes(gridcolor="rgba(142,202,230,0.15)")
    fig.update_yaxes(gridcolor="rgba(142,202,230,0.15)")
    return fig


def stint_state_sequence(pit_lap, start_age, compound_before, compound_after,
                         current_lap, race_total_laps):
    ages, compounds, laps = [], [], []
    age, compound = start_age, compound_before
    for lap in range(current_lap + 1, race_total_laps + 1):
        age += 1
        if lap == pit_lap:
            age, compound = 1, compound_after
        ages.append(age)
        compounds.append(compound)
        laps.append(lap)
    return np.array(ages), np.array(compounds), np.array(laps)


def gp_predict_states(ages, compounds, laps, mean_track_temp, mean_air_temp, models):
    m = np.column_stack([
        ages, ages ** 2, ages ** 3, compounds,
        np.full(len(ages), mean_track_temp),
        np.full(len(ages), mean_air_temp), laps,
    ])
    X = models["gp_scaler"].transform(m)
    mean, std = models["gp"].predict(X, return_std=True)
    return mean, std


def run_gp_monte_carlo(pit_offset, models, config, n_sims=10_000, seed=7):
    mc = config["MC"]
    current_lap = mc["CURRENT_LAP"]
    race_total = mc["RACE_TOTAL_LAPS"]
    remaining = race_total - current_lap
    actual_pit_lap = current_lap + pit_offset

    our_ages, our_comp, our_laps = stint_state_sequence(
        actual_pit_lap, mc["OUR_TYRE_AGE_START"], mc["OUR_COMPOUND_BEFORE"],
        mc["OUR_COMPOUND_AFTER"], current_lap, race_total)
    rival_ages, rival_comp, rival_laps = stint_state_sequence(
        45, mc["RIVAL_TYRE_AGE"], 1, 2, current_lap, race_total)

    our_mean, our_std = gp_predict_states(
        our_ages, our_comp, our_laps, config["MEAN_TRACK_TEMP"],
        config["MEAN_AIR_TEMP"], models)
    rival_mean, rival_std = gp_predict_states(
        rival_ages, rival_comp, rival_laps, config["MEAN_TRACK_TEMP"],
        config["MEAN_AIR_TEMP"], models)

    n_checkpoints = len(range(0, remaining, 10))
    rng = np.random.default_rng(seed)
    sc_count = (rng.random((n_sims, n_checkpoints)) < mc["SAFETY_CAR_PROBABILITY"]).sum(axis=1)

    our_total = (our_mean.sum() + mc["PIT_STOP_DELTA"]
                 + (rng.standard_normal((n_sims, remaining)) * our_std).sum(axis=1)
                 + 3 * 45.0 * sc_count)
    rival_total = (rival_mean.sum() + mc["PIT_STOP_DELTA"]
                   + (rng.standard_normal((n_sims, remaining)) * rival_std).sum(axis=1)
                   + 3 * 45.0 * sc_count)

    gap = (rival_total + mc["RIVAL_GAP_SECONDS"]) - our_total
    return gap


def main():
    if not (ART / "config.json").exists():
        st.error("Artifacts not found. Run the notebook once — it exports everything "
                 "the dashboard needs into `artifacts/`.")
        st.code("jupyter nbconvert --to notebook --execute --inplace "
                "f1-tire-degradation-performance-decay-analysis.ipynb")
        return

    data = load_data()
    cfg = data["config"]
    mc = cfg["MC"]

    st.sidebar.markdown("## 🏁 Pit Wall Analytics")
    st.sidebar.markdown("Tire degradation & undercut optimization — "
                        "**Bahrain Grand Prix 2023**")
    st.sidebar.divider()

    win_xgb = data["win_tables"].query("Engine == 'XGBoost'")
    best_strategy = int(win_xgb.loc[win_xgb["WinPct"].idxmax(), "Strategy"])
    best_model = data["metrics"].loc[data["metrics"]["SIS (s)"].idxmin(), "Model"]

    st.sidebar.metric("Recommended pit lap", f"Lap {mc['CURRENT_LAP'] + best_strategy}")
    st.sidebar.metric("Lowest strategy damage model", best_model)
    st.sidebar.metric("Simulations per strategy", f"{mc['N_SIMULATIONS']:,}")
    st.sidebar.caption("Artifacts are read-only exports of the notebook — "
                       "the app never retrains models.")

    st.title("🏁 F1 Tire Degradation & Performance Decay Analysis")
    st.caption("Bahrain Grand Prix 2023 · Physics-based fuel correction · "
               "Asymmetric-loss XGBoost · Random Forest · Gaussian Process · "
               "Monte Carlo strategy simulation")

    tab_overview, tab_method, tab_eda, tab_models, tab_mc, tab_proof = st.tabs(
        ["Overview", "Data & Method", "EDA", "Models", "Monte Carlo", "Proof & Inference"])

    # ------------------------------------------------------------------ OVERVIEW
    with tab_overview:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Laps modeled", f"{cfg['n_laps_modeled']:,}")
        c2.metric("Drivers / stints", f"{cfg['n_drivers']} / {cfg['n_stints']}")
        c3.metric("Recommended pit lap",
                  f"Lap {mc['CURRENT_LAP'] + best_strategy}",
                  f"+{best_strategy} laps from now")
        c4.metric("Best model (SIS)", best_model)

        st.markdown("### Key findings")
        win_xgb_top = win_xgb.sort_values("WinPct", ascending=False)
        f1, f2, f3 = st.columns(3)
        f1.markdown(f"""
        <div style="border:1px solid {ORANGE}; border-radius:10px; padding:16px;">
        <h4 style="color:{AMBER}; margin:0;">Optimal undercut window</h4>
        <p style="color:{SKY};">Pitting at <b>lap {mc['CURRENT_LAP'] + best_strategy}</b>
        maximizes win probability
        (<b>{win_xgb_top.iloc[0]['WinPct']*100:.1f}%</b>) against a rival pitting at lap 45.
        Earlier laps trade tire life for track position.</p></div>
        """, unsafe_allow_html=True)
        f2.markdown(f"""
        <div style="border:1px solid {AMBER}; border-radius:10px; padding:16px;">
        <h4 style="color:{AMBER}; margin:0;">Soft tire cliff ≈ lap 16</h4>
        <p style="color:{SKY};">LOWESS-smoothed degradation shows the soft compound's
        pace drop-off accelerating sharply around tire age 16 — the cliff every
        strategy must respect.</p></div>
        """, unsafe_allow_html=True)
        f3.markdown(f"""
        <div style="border:1px solid {BLUE}; border-radius:10px; padding:16px;">
        <h4 style="color:{AMBER}; margin:0;">Uncertainty matters</h4>
        <p style="color:{SKY};">The GP variant propagates <b>per-lap σ</b> into the
        race simulation: probabilities compress toward ~50% and P10–P90 gap bands
        widen — the models disagree on how confident the undercut is, and the GP
        prices in that model risk.</p></div>
        """, unsafe_allow_html=True)

        st.markdown("### Methodology at a glance")
        st.markdown(f"""
        Raw FastF1 laps → clean-air filtering → physics-based **fuel correction**
        (K = {cfg['FUEL_EFFECT_K']} s/lap/kg) → target `LapTimeDelta` (tire-only
        degradation) → **5 models** (Linear, Polynomial d{cfg['optimal_degree']},
        XGBoost asymmetric, Random Forest, Gaussian Process) →
        **{mc['N_SIMULATIONS']:,}-run Monte Carlo** on 4 pit strategies →
        probabilistic optimal pit window, validated against real-world pit stops.
        """)

    # ------------------------------------------------------------------ METHOD
    with tab_method:
        st.markdown("### Physics-based fuel burn correction")
        k = cfg["FUEL_EFFECT_K"]
        st.latex(r"\text{FuelCorrectedLapTime} = \text{LapTime} - (R_{laps} - N_{lap}) \cdot \Delta_{fuel}")
        st.latex(rf"\Delta_{{fuel}} = K \times \frac{{F_{{start}}}}{{R_{{laps}}}} = {k} \times \frac{{110}}{{{cfg['RACE_LAPS']}}} \approx {cfg['DELTA_PER_LAP']:.4f}\ \text{{s/lap}}")
        st.markdown(f"""
        Each lap, the car burns ~{cfg['FUEL_PER_LAP']:.2f} kg of fuel, worth
        ≈{cfg['DELTA_PER_LAP']*1000:.0f} ms of lap time. Removing this linear effect
        isolates the **non-linear tire wear signal** underneath.
        """)

        st.markdown("### Target variable — isolating tire wear from car pace")
        st.latex(r"\text{LapTimeDelta}_{lap} = \text{FuelCorrectedLapTime}_{lap} - \min_{\text{stint}}(\text{FuelCorrectedLapTime})")

        st.markdown("### Cleaning funnel")
        p1, p2, p3 = st.columns(3)
        p1.metric("Accurate laps only", "IsAccurate == True")
        p2.metric("Removed", "SC laps · pit laps · dirty air · ±2.5σ outliers")
        p3.metric("Retained", f"{cfg['n_laps_modeled']:,} laps · {cfg['n_stints']} stints ≥ 3 laps")

        st.markdown("### Leakage-safe split")
        st.markdown(f"""
        `GroupShuffleSplit` on **StintID** (80/20) — all laps of a stint stay on the
        same side, so the model can never peek at a stint's own later laps.
        Features: `{', '.join(cfg['features'])}`
        """)

        st.markdown("### Compound mix (modeling set)")
        comp_df = pd.DataFrame(
            [{"Compound": c, "Laps": n} for c, n in cfg["compound_counts"].items()])
        fig = px.bar(comp_df, x="Compound", y="Laps",
                     color="Compound",
                     color_discrete_map=COMPOUND_COLORS)
        base_layout(fig)
        st.plotly_chart(fig, width="stretch")

    # ------------------------------------------------------------------ EDA
    with tab_eda:
        st.markdown("### Pure tire degradation by compound (fuel-corrected Δ)")
        fig = go.Figure()
        for compound in data["lowess"]["Compound"].unique():
            curve = data["lowess"].query("Compound == @compound")
            fig.add_scatter(x=curve["TyreLife"], y=curve["SmoothedDelta"],
                            mode="lines", name=compound,
                            line=dict(color=COMPOUND_COLORS.get(compound, SKY), width=3))
        fig.add_vline(x=16, line_dash="dash", line_color=ORANGE,
                      annotation_text="Soft tire cliff (~lap 16)")
        fig.update_layout(xaxis_title="Tire life (laps)",
                          yaxis_title="Pace drop-off (s)", legend_title="Compound")
        base_layout(fig)
        st.plotly_chart(fig, width="stretch")
        st.markdown("💬 **Inference:** the soft compound degrades fastest and hits its "
                    "cliff near age 16; the medium offers the flattest long-run slope — "
                    "the backbone of a one-stop race.")

        st.markdown("### Lap time variance: clean air vs dirty air")
        fig = px.box(data["conditions"], x="Condition", y="LapTimeSeconds",
                     color="Condition",
                     color_discrete_map={"Clean Air": BLUE, "Dirty Air (< 2s gap)": ORANGE})
        fig.update_layout(showlegend=False, yaxis_title="Lap time (s)")
        base_layout(fig)
        st.plotly_chart(fig, width="stretch")
        st.markdown("💬 **Inference:** laps within 2 s of the car ahead are systematically "
                    "slower and noisier — justifying the clean-air filter before modeling "
                    "pure tire degradation.")

        st.markdown("### Track temperature × tire age heat map (mean Δ)")
        hm = data["heatmap"].copy()
        hm = hm.set_index("TrackTempBin")
        hm.columns = [str(c) for c in hm.columns]
        fig = px.imshow(hm, color_continuous_scale="RdYlBu_r", aspect="auto",
                        labels=dict(color="Mean Δ (s)"))
        base_layout(fig)
        st.plotly_chart(fig, width="stretch")
        st.markdown("💬 **Inference:** degradation penalties compound in the high-temp × "
                    "old-tire corner — thermal load and tire age are multiplicative, "
                    "not additive.")

    # ------------------------------------------------------------------ MODELS
    with tab_models:
        st.markdown("### Five-model comparison (strategy-aware metrics)")
        metrics = data["metrics"].copy()
        metrics["SIS (s)"] = metrics["SIS (s)"].astype(float)
        metrics["SIS (s)"] = metrics["SIS (s)"].abs()
        styled = metrics.style.format({
            "RMSE (s)": "{:.4f}", "MAE (s)": "{:.4f}", "R²": "{:.4f}",
            "TW-RMSE (s)": "{:.4f}", "SIS (s)": "{:.3f}", "Time (s)": "{:.2f}",
        }).highlight_min(subset=["RMSE (s)", "MAE (s)", "TW-RMSE (s)", "SIS (s)"],
                         color="#1a4a35").highlight_max(subset=["R²"], color="#1a4a55")
        st.dataframe(styled, width="stretch", hide_index=True)

        m1, m2 = st.columns(2)
        fig = px.bar(metrics, x="Model", y="RMSE (s)", color="Model", text_auto=".3f",
                     color_discrete_sequence=[ORANGE, AMBER, BLUE, SKY, "#7E57C2"])
        fig.update_layout(showlegend=False, title="RMSE — pure math accuracy")
        m1.plotly_chart(base_layout(fig), width="stretch")

        fig = px.bar(metrics, x="Model", y="SIS (s)", color="Model", text_auto=".2f",
                     color_discrete_sequence=[ORANGE, AMBER, BLUE, SKY, "#7E57C2"])
        fig.update_layout(showlegend=False, title="SIS — race seconds lost to optimistic errors")
        m2.plotly_chart(base_layout(fig), width="stretch")
        best_rmse_model = metrics.loc[metrics["RMSE (s)"].idxmin(), "Model"]
        best_sis_model = metrics.loc[metrics["SIS (s)"].idxmin(), "Model"]
        st.markdown(f"""
        💬 **Inference:** **{best_rmse_model}** leads raw RMSE, while **{best_sis_model}**
        posts the lowest Strategy Impact Score (race seconds lost to optimistic
        predictions). The asymmetric XGBoost objective curbs optimism relative to
        the linear/polynomial baselines — trading a little accuracy for fewer
        strategy-critical errors.
        """)

        st.markdown("### Polynomial degree selection (AIC / BIC / CV-RMSE)")
        ps = data["poly_sel"]
        fig = go.Figure()
        fig.add_scatter(x=ps["degree"], y=ps["AIC"], mode="lines+markers", name="AIC")
        fig.add_scatter(x=ps["degree"], y=ps["BIC"], mode="lines+markers", name="BIC")
        fig.add_scatter(x=ps["degree"], y=ps["CV_RMSE"], mode="lines+markers",
                        name="CV RMSE", line=dict(dash="dot"))
        fig.add_vline(x=cfg["optimal_degree"], line_dash="dash", line_color=AMBER,
                      annotation_text=f"chosen: degree {cfg['optimal_degree']}")
        fig.update_layout(xaxis_title="Polynomial degree",
                          yaxis_title="Criterion value", legend_title="")
        base_layout(fig)
        st.plotly_chart(fig, width="stretch")
        st.markdown("💬 **Inference:** AIC/BIC elbow at degree 2 — quadratic curvature "
                    "captures the wear ramp; higher degrees only chase noise.")

        st.markdown("### Reality trace — PER, stint 3 (all models vs actual)")
        trace = data["trace"]
        fig = go.Figure()
        fig.add_scatter(x=trace["LapNumber"], y=trace["Actual"], mode="markers",
                        name="Actual", marker=dict(color="#B0BEC5", size=7))
        for col, color, dash in [("Linear", SKY, "dash"), ("Poly", AMBER, "dash"),
                                 ("XGB", ORANGE, "solid"), ("RF", "#66BB6A", "dot"),
                                 ("GP", "#7E57C2", "solid")]:
            fig.add_scatter(x=trace["LapNumber"], y=trace[col], mode="lines",
                            name=col, line=dict(color=color, dash=dash, width=2.5))
        models = load_models()
        laps = data["laps"]
        per = laps[(laps["Driver"] == "PER") & (laps["Stint"] == 3)].sort_values("LapNumber")
        if not per.empty:
            X = per[FEATURES].values
            gp_mean, gp_std = models["gp"].predict(models["gp_scaler"].transform(X),
                                                   return_std=True)
            anchor = per["PersonalBestInStint"].iloc[0]
            fig.add_scatter(x=per["LapNumber"], y=anchor + gp_mean + gp_std,
                            mode="lines", line=dict(width=0), showlegend=False,
                            hoverinfo="skip", name="GP+σ")
            fig.add_scatter(x=per["LapNumber"], y=anchor + gp_mean - gp_std,
                            mode="lines", fill="tonexty", fillcolor="rgba(126,87,194,0.15)",
                            line=dict(width=0), name="GP ±1σ")
        fig.update_layout(xaxis_title="Lap", yaxis_title="Fuel-corrected lap time (s)")
        base_layout(fig)
        st.plotly_chart(fig, width="stretch")

    # ------------------------------------------------------------------ MONTE CARLO
    with tab_mc:
        st.markdown(f"### Strategy win probabilities — {mc['N_SIMULATIONS']:,} runs each")
        wt = data["win_tables"].copy()
        wt["PitLap"] = mc["CURRENT_LAP"] + wt["Strategy"]
        wt["Win%"] = wt["WinPct"] * 100
        fig = px.bar(wt, x="PitLap", y="Win%", color="Engine", barmode="group",
                     color_discrete_map=ENGINE_COLORS,
                     category_orders={"PitLap": sorted(wt["PitLap"].unique())})
        fig.update_layout(xaxis_title="Our pit lap", yaxis_title="Win probability (%)",
                          legend_title="Simulation engine")
        base_layout(fig)
        st.plotly_chart(fig, width="stretch")

        cmp_cols = ["Strategy", "ActualPitLap"]
        ren = {}
        for col in data["win_cmp"].columns:
            if col not in cmp_cols and col != "Strategy":
                ren[col] = col.replace("_XGB", " ·XGB").replace("_GP", " ·GP")
        cmp_show = data["win_cmp"].rename(columns=ren)
        st.dataframe(cmp_show.style.format("{:.2f}",
                      subset=[c for c in cmp_show.columns if "·" in c]),
                      width="stretch", hide_index=True)
        st.markdown("""
        💬 **Inference:** the two engines **disagree on confidence, not on the
        direction of the undercut** — XGBoost's fixed low residual noise produces
        sharp win probabilities, while the GP's per-lap σ compresses them toward
        ~50% and widens the P10–P90 bands. XGBoost remains the primary result;
        the GP prices in model uncertainty that fixed noise assumes away.
        """)

        st.markdown("### Gap distributions")
        engine = st.radio("Engine", ["XGBoost", "Gaussian Process"],
                          horizontal=True)
        sim = data["sim_xgb"] if engine == "XGBoost" else data["sim_gp"]
        fig = go.Figure()
        strat_colors = [ORANGE, AMBER, BLUE, SKY]
        for i, s in enumerate(sorted(sim["Strategy"].unique())):  # ordered strategies
            g = sim.query("Strategy == @s")["FinalGapSeconds"]
            fig.add_histogram(x=g, name=f"Pit lap {mc['CURRENT_LAP'] + s}",
                              histnorm="probability density", opacity=0.45,
                              marker_color=strat_colors[i % len(strat_colors)])
        fig.update_layout(barmode="overlay", xaxis_title="Final gap (s, positive = ahead)",
                          yaxis_title="Density")
        base_layout(fig)
        st.plotly_chart(fig, width="stretch")

        st.markdown("---")
        st.markdown("### 🧪 What-if simulator (live GP Monte Carlo)")
        w1, w2, w3 = st.columns(3)
        with w1:
            pit_offset = st.slider("Pit lap", 40, 56, 42,
                                   help="Lap on which our car pits (Medium → Hard)")
        with w2:
            our_age = st.slider("Our current tire age", 5, 30, 15)
        with w3:
            rival_gap = st.slider("Rival gap ahead (s)", 0.0, 10.0, float(mc["RIVAL_GAP_SECONDS"]), 0.1)
        if st.button("Run 10,000 simulations", type="primary"):
            models = load_models()
            gap = run_gp_monte_carlo(pit_offset - mc["CURRENT_LAP"], models, cfg)
            win_pct = (gap > 0).mean() * 100
            p10, med, p90 = np.percentile(gap, [10, 50, 90])
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Win probability", f"{win_pct:.1f}%")
            r2.metric("Median gap", f"{med:+.2f} s")
            r3.metric("P10 gap", f"{p10:+.2f} s")
            r4.metric("P90 gap", f"{p90:+.2f} s")
            fig = px.histogram(x=gap, nbins=80, histnorm="probability density",
                               color_discrete_sequence=[ORANGE])
            fig.add_vline(x=0, line_dash="dash", line_color=SKY)
            fig.update_layout(showlegend=False, xaxis_title="Final gap (s)",
                              yaxis_title="Density", title="Finishing-gap distribution")
            base_layout(fig)
            st.plotly_chart(fig, width="stretch")
            verdict = "🟢 We likely finish ahead — commit." if win_pct >= 50 else \
                      "🔴 We likely finish behind — consider an earlier undercut."
            st.markdown(verdict)

    # ------------------------------------------------------------------ PROOF
    with tab_proof:
        st.markdown("### Proof 1 — Model selection is statistically grounded")
        st.markdown(f"""
        Polynomial degree chosen by **AIC/BIC + 5-fold CV** (not eyeballed), the
        XGBoost hyper-parameters by **100-trial Optuna** search, the Random Forest
        by a 30-trial sweep, and the GP by **marginal-likelihood kernel optimization**:

        Learned GP kernel: `{cfg['gp_kernel']}`
        """)
        st.markdown(f"XGBoost best params: `{cfg['best_xgb_params']}`")
        st.markdown(f"Random Forest best params: `{cfg['best_rf_params']}`")

        st.markdown("### Proof 2 — Model tracks reality lap-by-lap")
        trace = data["trace"]
        fig = go.Figure()
        fig.add_scatter(x=trace["LapNumber"], y=trace["Actual"].cumsum(),
                        mode="lines+markers", name="Actual cumulative")
        fig.add_scatter(x=trace["LapNumber"], y=trace["XGB"].cumsum(),
                        mode="lines+markers", name="Model cumulative", line=dict(dash="dash"))
        drift = trace["XGB"].cumsum() - trace["Actual"].cumsum()
        fig.add_bar(x=trace["LapNumber"], y=drift, name="Error drift",
                    marker_color=np.where(drift > 0, BLUE, ORANGE), yaxis="y2")
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="Drift (s)"),
                          xaxis_title="Lap",
                          yaxis_title="Cumulative seconds in stint")
        base_layout(fig)
        st.plotly_chart(fig, width="stretch")
        total_drift = abs((trace["XGB"].cumsum() - trace["Actual"].cumsum()).iloc[-1])
        st.markdown(f"💬 **Inference:** after {len(trace)} laps the model's total track-"
                    f"position error is **{total_drift:.2f} s** — tight enough for "
                    "strategy calls where margins are typically >2 s.")

        st.markdown("### Proof 3 — Model recommendations align with real pit decisions")
        pits = data["pits"]["LapNumber"]
        fig = px.histogram(x=pits, nbins=15, color_discrete_sequence=[BLUE])
        xgb_best = int(data["win_cmp"].loc[data["win_cmp"]["WinPct_XGB"].idxmax(), "ActualPitLap"])
        fig.add_vline(x=xgb_best, line_dash="dash", line_color=ORANGE,
                      annotation_text=f"Model P90 optimal (lap {xgb_best})")
        fig.update_layout(showlegend=False, xaxis_title="Race lap",
                          yaxis_title="Real pit stops")
        base_layout(fig)
        st.plotly_chart(fig, width="stretch")
        st.markdown(f"💬 **Inference:** real teams clustered their final stops inside "
                    f"laps 35–50, bracketing our model's recommendation of lap "
                    f"{xgb_best} — the simulation recommends a *defensible aggressive* "
                    "undercut, not an outlier.")

        st.markdown("### Proof 4 — Uncertainty is quantified, not assumed away")
        st.markdown(f"""
        The XGBoost residual σ = **{cfg['residual_std']:.4f} s** feeds the baseline
        simulation as uniform lap noise, while the GP supplies **per-lap σ** that
        widens with tire age — the Monte Carlo comparison table in the previous tab
        quantifies exactly how much risk the two views disagree on.
        """)

        st.markdown("---")
        st.caption("Authors: B Sudharshan, Arush Srivastava, Dharanitharan D · Predictive Analytics project · "
                   "Data: FastF1 (Bahrain 2023 Race) · Dashboard is a read-only view of "
                   "notebook-exported artifacts.")


if __name__ == "__main__":
    main()
