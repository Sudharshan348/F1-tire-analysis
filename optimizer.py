"""Full-window pit-strategy optimization for the Bahrain 2023 project.

The notebook's original strategy search evaluated four pit laps
(42, 45, 48, 51). This module replaces that fixed grid with a proper
optimization over the full feasible window, using a model-averaged
win-probability objective and Differential Evolution.

The evaluator uses common random numbers (CRN): every candidate pit lap
sees the same safety-car draws and the same standard-normal lap-noise
draws. The resulting objective is deterministic and comparable across
candidates; independent seeds are used later for validation.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize_scalar, newton


def asymmetric_mse_objective(y_true: np.ndarray, y_pred: np.ndarray):
    """Custom XGBoost objective used when the serialized model was trained."""
    residual = y_pred - y_true
    alpha_under = 2.5
    alpha_over = 1.0
    grad = np.where(residual < 0, alpha_under * residual, alpha_over * residual)
    hess = np.where(
        residual < 0,
        alpha_under * np.ones_like(residual),
        alpha_over * np.ones_like(residual),
    )
    return grad, hess


import __main__

if not hasattr(__main__, "asymmetric_mse_objective"):
    __main__.asymmetric_mse_objective = asymmetric_mse_objective


ARTIFACT_DIR = Path("artifacts")
DEFAULT_OPT_SIMULATIONS = 50_000
DEFAULT_VALIDATION_SIMULATIONS = 20_000
DEFAULT_VALIDATION_SEEDS = (101, 202, 303, 404, 505)
DEFAULT_DE_SEEDS = (42, 7, 2026)
SC_LAP_SECONDS = 45.0
SC_LAPS = 3


def _clip(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


@dataclass
class RaceScenario:
    """A vectorized dual-engine race simulator with shared random numbers."""

    config: Dict[str, Any]
    models: Dict[str, Any]
    n_sims: int = DEFAULT_OPT_SIMULATIONS
    seed: int = 42

    def __post_init__(self) -> None:
        mc = self.config["MC"]
        self.current_lap = int(mc["CURRENT_LAP"])
        self.race_total_laps = int(mc["RACE_TOTAL_LAPS"])
        self.remaining = self.race_total_laps - self.current_lap
        self.pit_stop_delta = float(mc["PIT_STOP_DELTA"])
        self.sc_probability = float(mc["SAFETY_CAR_PROBABILITY"])
        self.rival_gap = float(mc["RIVAL_GAP_SECONDS"])
        self.rival_pit_lap = 45
        self.residual_std = float(self.config["residual_std"])
        self.mean_track_temp = float(self.config["MEAN_TRACK_TEMP"])
        self.mean_air_temp = float(self.config["MEAN_AIR_TEMP"])
        self.min_pit_lap = self.current_lap + 1
        self.max_pit_lap = self.race_total_laps - 1

        self.xgb = self.models["xgb"]
        self.gp = self.models["gp"]
        self.gp_scaler = self.models["gp_scaler"]

        rng = np.random.default_rng(self.seed)
        n_checkpoints = len(range(0, self.remaining, 10))
        sc_count = (
            rng.random((self.n_sims, n_checkpoints)) < self.sc_probability
        ).sum(axis=1)
        self.sc_penalty = SC_LAPS * SC_LAP_SECONDS * sc_count
        self.noise_our = rng.standard_normal((self.n_sims, self.remaining))
        self.noise_rival = rng.standard_normal((self.n_sims, self.remaining))

        self.rival_features = self._features_for(
            self.rival_pit_lap,
            int(mc["RIVAL_TYRE_AGE"]),
            1,
            2,
        )
        self.rival_xgb_mean = float(self.xgb.predict(self.rival_features).sum())
        rival_gp_mean, rival_gp_std = self.gp.predict(
            self.gp_scaler.transform(self.rival_features), return_std=True
        )
        self.rival_gp_mean = float(rival_gp_mean.sum())
        self.rival_gp_std = rival_gp_std
        self.rival_total_xgb = (
            self.rival_xgb_mean
            + self.pit_stop_delta
            + (self.noise_rival * self.residual_std).sum(axis=1)
            + self.sc_penalty
        )
        self.rival_total_gp = (
            self.rival_gp_mean
            + self.pit_stop_delta
            + (self.noise_rival * self.rival_gp_std).sum(axis=1)
            + self.sc_penalty
        )

        self._cache: Dict[Tuple[int, bool], Dict[str, Any]] = {}
        self._evaluated_laps: set[int] = set()

    @property
    def n_unique(self) -> int:
        return len(self._evaluated_laps)

    def _features_for(
        self,
        pit_lap: int,
        start_age: int,
        compound_before: int,
        compound_after: int,
    ) -> np.ndarray:
        ages: List[int] = []
        compounds: List[int] = []
        laps: List[int] = []
        age = start_age
        compound = compound_before
        for lap in range(self.current_lap + 1, self.race_total_laps + 1):
            age += 1
            if lap == pit_lap:
                age = 1
                compound = compound_after
            ages.append(age)
            compounds.append(compound)
            laps.append(lap)
        ages_arr = np.asarray(ages)
        compounds_arr = np.asarray(compounds)
        laps_arr = np.asarray(laps)
        return np.column_stack(
            [
                ages_arr,
                ages_arr**2,
                ages_arr**3,
                compounds_arr,
                np.full(len(ages_arr), self.mean_track_temp),
                np.full(len(ages_arr), self.mean_air_temp),
                laps_arr,
            ]
        )

    def _our_features(self, pit_lap: int) -> np.ndarray:
        mc = self.config["MC"]
        return self._features_for(
            pit_lap,
            int(mc["OUR_TYRE_AGE_START"]),
            int(mc["OUR_COMPOUND_BEFORE"]),
            int(mc["OUR_COMPOUND_AFTER"]),
        )

    def deterministic_gap(self, pit_lap: int) -> float:
        """Mean final gap with no lap noise and no safety car."""
        our_features = self._our_features(pit_lap)
        our_xgb = float(self.xgb.predict(our_features).sum())
        our_gp = float(self.gp.predict(self.gp_scaler.transform(our_features)).sum())
        gap_xgb = self.rival_xgb_mean + self.rival_gap - our_xgb
        gap_gp = self.rival_gp_mean + self.rival_gap - our_gp
        return 0.5 * (gap_xgb + gap_gp)

    def evaluate(self, pit_lap: int, return_gaps: bool = False) -> Dict[str, Any]:
        pit_lap = int(round(pit_lap))
        if pit_lap < self.min_pit_lap or pit_lap > self.max_pit_lap:
            raise ValueError(
                f"pit_lap must be in [{self.min_pit_lap}, {self.max_pit_lap}]"
            )
        cache_key = (pit_lap, bool(return_gaps))
        if cache_key in self._cache:
            return self._cache[cache_key]

        features = self._our_features(pit_lap)
        our_xgb = float(self.xgb.predict(features).sum())
        our_gp_mean, our_gp_std = self.gp.predict(
            self.gp_scaler.transform(features), return_std=True
        )
        our_gp_mean = float(our_gp_mean.sum())

        our_total_xgb = (
            our_xgb
            + self.pit_stop_delta
            + (self.noise_our * self.residual_std).sum(axis=1)
            + self.sc_penalty
        )
        our_total_gp = (
            our_gp_mean
            + self.pit_stop_delta
            + (self.noise_our * our_gp_std).sum(axis=1)
            + self.sc_penalty
        )
        gap_xgb = self.rival_total_xgb + self.rival_gap - our_total_xgb
        gap_gp = self.rival_total_gp + self.rival_gap - our_total_gp

        win_xgb = float((gap_xgb > 0).mean())
        win_gp = float((gap_gp > 0).mean())
        median_xgb = float(np.median(gap_xgb))
        median_gp = float(np.median(gap_gp))
        ensemble_win = 0.5 * (win_xgb + win_gp)
        ensemble_median = 0.5 * (median_xgb + median_gp)

        result: Dict[str, Any] = {
            "PitLap": pit_lap,
            "XGB_WinPct": win_xgb,
            "GP_WinPct": win_gp,
            "Ensemble_WinPct": ensemble_win,
            "XGB_MedianGap": median_xgb,
            "GP_MedianGap": median_gp,
            "Ensemble_MedianGap": ensemble_median,
            "XGB_MeanGap": float(gap_xgb.mean()),
            "GP_MeanGap": float(gap_gp.mean()),
            "XGB_P10": float(np.percentile(gap_xgb, 10)),
            "XGB_P90": float(np.percentile(gap_xgb, 90)),
            "GP_P10": float(np.percentile(gap_gp, 10)),
            "GP_P90": float(np.percentile(gap_gp, 90)),
            "Objective": ensemble_win + 1e-6 * ensemble_median,
        }
        if return_gaps:
            result["GapXGB"] = gap_xgb
            result["GapGP"] = gap_gp
        self._evaluated_laps.add(pit_lap)
        self._cache[cache_key] = result
        return result


class ContinuousObjective:
    """Linear interpolation of the integer pit-lap objective for golden-section."""

    def __init__(self, scenario: RaceScenario):
        self.scenario = scenario

    def __call__(self, x: float) -> float:
        x = _clip(float(x), self.scenario.min_pit_lap, self.scenario.max_pit_lap)
        low = int(np.floor(x))
        high = int(np.ceil(x))
        low = max(self.scenario.min_pit_lap, min(self.scenario.max_pit_lap, low))
        high = max(self.scenario.min_pit_lap, min(self.scenario.max_pit_lap, high))
        if low == high:
            return self.scenario.evaluate(low)["Objective"]
        weight = x - low
        return (1.0 - weight) * self.scenario.evaluate(low)["Objective"] + weight * self.scenario.evaluate(high)["Objective"]


def _best_row(
    frame: pd.DataFrame, win_col: str, gap_col: str
) -> pd.Series:
    best_win = frame[win_col].max()
    candidates = frame[np.isclose(frame[win_col], best_win, atol=1e-12)]
    return candidates.loc[candidates[gap_col].idxmax()]


def run_grid(
    scenario: RaceScenario, grid_laps: Sequence[int]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    started = time.perf_counter()
    rows = [scenario.evaluate(pit_lap) for pit_lap in grid_laps]
    runtime = time.perf_counter() - started
    best = max(rows, key=lambda row: (row["Objective"], row["Ensemble_MedianGap"]))
    return rows, {
        "best": best,
        "evaluations": len(grid_laps),
        "runtime_s": runtime,
    }


def run_golden_section(
    scenario: RaceScenario, bounds: Tuple[int, int]
) -> Dict[str, Any]:
    started = time.perf_counter()
    objective = ContinuousObjective(scenario)
    result = minimize_scalar(
        lambda x: -objective(x),
        method="bounded",
        bounds=(float(bounds[0]), float(bounds[1])),
        options={"xatol": 1e-4},
    )
    best_pit_lap = int(round(_clip(result.x, bounds[0], bounds[1])))
    best = scenario.evaluate(best_pit_lap)
    return {
        "best": best,
        "continuous_x": float(result.x),
        "evaluations": scenario.n_unique,
        "runtime_s": time.perf_counter() - started,
        "success": bool(result.success),
    }


def run_newton_surrogate(
    scenario: RaceScenario, starts: Sequence[int] = (31, 38, 42, 48, 54)
) -> Dict[str, Any]:
    started = time.perf_counter()
    roots: List[Dict[str, Any]] = []
    surrogate_evaluations = 0
    for start in starts:
        start = int(_clip(start, scenario.min_pit_lap, scenario.max_pit_lap))
        low = max(scenario.min_pit_lap, start - 3)
        high = min(scenario.max_pit_lap, start + 3)
        pit_laps = np.arange(low, high + 1, dtype=float)
        surrogate_evaluations += len(pit_laps)
        values = np.asarray([scenario.deterministic_gap(int(p)) for p in pit_laps])
        if len(pit_laps) < 3:
            continue
        coefficients = np.polyfit(pit_laps, values, 2)
        a, b = float(coefficients[0]), float(coefficients[1])
        try:
            if abs(a) < 1e-12:
                raise RuntimeError("quadratic surrogate is flat")
            root = float(
                newton(
                    lambda x: 2.0 * a * x + b,
                    x0=float(start),
                    fprime=lambda x: 2.0 * a,
                    maxiter=20,
                )
            )
        except Exception as exc:
            roots.append(
                {
                    "Start": start,
                    "Root": np.nan,
                    "Feasible": False,
                    "Note": str(exc),
                }
            )
            continue
        feasible = bool(scenario.min_pit_lap <= root <= scenario.max_pit_lap)
        roots.append(
            {
                "Start": start,
                "Root": root,
                "RoundedPitLap": int(round(root)) if feasible else np.nan,
                "Feasible": feasible,
                "Note": "" if feasible else "root outside feasible pit-lap window",
            }
        )

    feasible_rows = [
        row for row in roots if row["Feasible"] and not np.isnan(row["Root"])
    ]
    best = None
    if feasible_rows:
        ranked = sorted(
            feasible_rows,
            key=lambda row: scenario.evaluate(int(round(row["Root"])))["Objective"],
            reverse=True,
        )
        best_pit_lap = int(round(ranked[0]["Root"]))
        best = scenario.evaluate(best_pit_lap)

    root_values = [row["Root"] for row in roots if not np.isnan(row["Root"])]
    root_note = (
        f"roots ranged {min(root_values):.1f} to {max(root_values):.1f}"
        if root_values
        else "no valid roots"
    )
    from_grid_row = next((row for row in roots if row["Start"] == 42), None)
    from_grid_pit_lap = None
    from_grid_result = None
    if from_grid_row is not None and from_grid_row["Feasible"]:
        from_grid_pit_lap = int(round(from_grid_row["Root"]))
        from_grid_result = scenario.evaluate(from_grid_pit_lap)
    return {
        "best": best,
        "roots": pd.DataFrame(roots),
        "root_note": root_note,
        "from_grid_pit_lap": from_grid_pit_lap,
        "from_grid_result": from_grid_result,
        "evaluations": surrogate_evaluations,
        "runtime_s": time.perf_counter() - started,
    }


def _run_de_once(
    scenario: RaceScenario,
    seed: int,
    popsize: int,
    maxiter: int,
    tolerance: float,
) -> Dict[str, Any]:
    history: List[Dict[str, Any]] = []

    def objective(x: np.ndarray) -> float:
        pit_lap = int(
            round(_clip(x[0], scenario.min_pit_lap, scenario.max_pit_lap))
        )
        return -scenario.evaluate(pit_lap)["Objective"]

    def callback(xk: np.ndarray, convergence: float) -> None:
        pit_lap = int(
            round(_clip(xk[0], scenario.min_pit_lap, scenario.max_pit_lap))
        )
        row = scenario.evaluate(pit_lap)
        history.append(
            {
                "Generation": len(history) + 1,
                "BestPitLap": pit_lap,
                "BestObjective": row["Objective"],
                "BestWinPct": row["Ensemble_WinPct"],
                "BestMedianGap": row["Ensemble_MedianGap"],
                "Convergence": float(convergence),
            }
        )

    started = time.perf_counter()
    result = differential_evolution(
        objective,
        bounds=[(float(scenario.min_pit_lap), float(scenario.max_pit_lap))],
        integrality=True,
        strategy="best1bin",
        popsize=popsize,
        maxiter=maxiter,
        tol=tolerance,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=seed,
        polish=False,
        updating="immediate",
        workers=1,
        callback=callback,
    )
    runtime = time.perf_counter() - started
    best_pit_lap = int(
        round(_clip(result.x[0], scenario.min_pit_lap, scenario.max_pit_lap))
    )
    best = scenario.evaluate(best_pit_lap)
    return {
        "seed": seed,
        "best": best,
        "objective": best["Objective"],
        "unique_evaluations": scenario.n_unique,
        "nfev": int(result.nfev),
        "nit": int(result.nit),
        "runtime_s": runtime,
        "history": pd.DataFrame(history),
        "success": bool(result.success),
        "message": str(result.message),
        "result": result,
    }


def run_de_seeds(
    config: Dict[str, Any],
    models: Dict[str, Any],
    seeds: Sequence[int],
    n_sims: int,
    popsize: int,
    maxiter: int,
    tolerance: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    runs: List[Dict[str, Any]] = []
    for seed in seeds:
        scenario = RaceScenario(config, models, n_sims=n_sims, seed=int(seed))
        runs.append(
            _run_de_once(
                scenario,
                seed=int(seed),
                popsize=popsize,
                maxiter=maxiter,
                tolerance=tolerance,
            )
        )
    best = max(
        runs,
        key=lambda run: (run["best"]["Objective"], run["best"]["Ensemble_MedianGap"]),
    )
    return best, runs


def run_landscape(
    config: Dict[str, Any], models: Dict[str, Any], n_sims: int, seed: int
) -> pd.DataFrame:
    scenario = RaceScenario(config, models, n_sims=n_sims, seed=seed)
    rows = [
        scenario.evaluate(pit_lap)
        for pit_lap in range(scenario.min_pit_lap, scenario.max_pit_lap + 1)
    ]
    return pd.DataFrame(rows)


def _aggregate_validation(seed_frame: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["Scenario", "PitLap"]
    rows: List[Dict[str, Any]] = []
    for (scenario_name, pit_lap), group in seed_frame.groupby(group_cols):
        row: Dict[str, Any] = {
            "Scenario": scenario_name,
            "PitLap": int(pit_lap),
            "Seeds": int(len(group)),
        }
        for metric in [
            "XGB_WinPct",
            "GP_WinPct",
            "Ensemble_WinPct",
            "XGB_MedianGap",
            "GP_MedianGap",
            "XGB_P10",
            "XGB_P90",
            "GP_P10",
            "GP_P90",
            "XGB_MedianGapGain",
            "GP_MedianGapGain",
        ]:
            values = group[metric].astype(float)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            ci = 1.96 * std / np.sqrt(len(values)) if len(values) > 1 else 0.0
            row[metric] = mean
            row[f"{metric}_CI"] = ci
        rows.append(row)
    return pd.DataFrame(rows)


def run_validation(
    config: Dict[str, Any],
    models: Dict[str, Any],
    candidates: Dict[str, int],
    n_sims: int,
    seeds: Sequence[int] = DEFAULT_VALIDATION_SEEDS,
    grid_pit_lap: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        scenario = RaceScenario(config, models, n_sims=n_sims, seed=int(seed))
        grid_gap = scenario.evaluate(grid_pit_lap, return_gaps=True)
        for scenario_name, pit_lap in candidates.items():
            result = scenario.evaluate(int(pit_lap), return_gaps=True)
            xgb_gain = float(
                np.median(result["GapXGB"] - grid_gap["GapXGB"])
            )
            gp_gain = float(np.median(result["GapGP"] - grid_gap["GapGP"]))
            seed_rows.append(
                {
                    "Scenario": scenario_name,
                    "PitLap": int(pit_lap),
                    "Seed": int(seed),
                    "XGB_WinPct": result["XGB_WinPct"],
                    "GP_WinPct": result["GP_WinPct"],
                    "Ensemble_WinPct": result["Ensemble_WinPct"],
                    "XGB_MedianGap": result["XGB_MedianGap"],
                    "GP_MedianGap": result["GP_MedianGap"],
                    "XGB_P10": result["XGB_P10"],
                    "XGB_P90": result["XGB_P90"],
                    "GP_P10": result["GP_P10"],
                    "GP_P90": result["GP_P90"],
                    "XGB_MedianGapGain": xgb_gain,
                    "GP_MedianGapGain": gp_gain,
                }
            )
    seed_frame = pd.DataFrame(seed_rows)
    return seed_frame, _aggregate_validation(seed_frame)


def _format_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def run_full_optimization(
    artifact_dir: Path | str = ARTIFACT_DIR,
    n_sims: int = DEFAULT_OPT_SIMULATIONS,
    validation_sims: int = DEFAULT_VALIDATION_SIMULATIONS,
    seed: int = 42,
    popsize: int = 12,
    maxiter: int = 30,
    tolerance: float = 1e-4,
    save: bool = True,
    verbose: bool = True,
) -> "OptimizationResult":
    artifact_dir = Path(artifact_dir)
    config = json.loads((artifact_dir / "config.json").read_text())
    models = {
        "xgb": joblib.load(artifact_dir / "xgb_model.joblib"),
        "gp": joblib.load(artifact_dir / "gp_model.joblib"),
        "gp_scaler": joblib.load(artifact_dir / "gp_scaler.joblib"),
    }
    mc = config["MC"]
    current_lap = int(mc["CURRENT_LAP"])
    race_total_laps = int(mc["RACE_TOTAL_LAPS"])
    min_pit_lap = current_lap + 1
    max_pit_lap = race_total_laps - 1
    grid_laps = [
        int(np.clip(current_lap + int(offset), min_pit_lap, max_pit_lap))
        for offset in mc["PIT_STRATEGIES"]
    ]

    grid_scenario = RaceScenario(config, models, n_sims=n_sims, seed=seed)
    _, grid_run = run_grid(grid_scenario, grid_laps)

    golden_full_scenario = RaceScenario(config, models, n_sims=n_sims, seed=seed)
    golden_full = run_golden_section(
        golden_full_scenario, (min_pit_lap, max_pit_lap)
    )

    old_low = max(min_pit_lap, min(grid_laps))
    old_high = min(max_pit_lap, max(grid_laps))
    golden_old_scenario = RaceScenario(config, models, n_sims=n_sims, seed=seed)
    golden_old = run_golden_section(golden_old_scenario, (old_low, old_high))

    newton_scenario = RaceScenario(config, models, n_sims=n_sims, seed=seed)
    newton_run = run_newton_surrogate(newton_scenario)

    de_best, de_runs = run_de_seeds(
        config,
        models,
        seeds=DEFAULT_DE_SEEDS,
        n_sims=n_sims,
        popsize=popsize,
        maxiter=maxiter,
        tolerance=tolerance,
    )

    landscape = run_landscape(config, models, n_sims=n_sims, seed=seed)
    xgb_row = _best_row(landscape, "XGB_WinPct", "XGB_MedianGap")
    gp_row = _best_row(landscape, "GP_WinPct", "GP_MedianGap")
    ensemble_row = _best_row(
        landscape, "Ensemble_WinPct", "Ensemble_MedianGap"
    )

    best = de_best["best"]
    grid_best = grid_run["best"]
    candidates = {
        "Grid best (baseline)": int(grid_best["PitLap"]),
        "DE optimum": int(best["PitLap"]),
        "XGB-only optimum": int(xgb_row["PitLap"]),
        "GP-only optimum": int(gp_row["PitLap"]),
    }
    validation_seeds, validation_summary = run_validation(
        config,
        models,
        candidates,
        n_sims=validation_sims,
        seeds=DEFAULT_VALIDATION_SEEDS,
        grid_pit_lap=int(grid_best["PitLap"]),
    )

    comparison_rows = [
        {
            "Algorithm": "Grid search (baseline)",
            "SearchRange": f"{min(grid_laps)}-{max(grid_laps)} ({len(grid_laps)} points)",
            "BestPitLap": int(grid_best["PitLap"]),
            "XGB_WinPct": grid_best["XGB_WinPct"],
            "GP_WinPct": grid_best["GP_WinPct"],
            "Ensemble_WinPct": grid_best["Ensemble_WinPct"],
            "XGB_MedianGap": grid_best["XGB_MedianGap"],
            "GP_MedianGap": grid_best["GP_MedianGap"],
            "Evaluations": grid_run["evaluations"],
            "Runtime_s": grid_run["runtime_s"],
            "HandlesInteger": "Yes",
            "NeedsDerivative": "No",
            "Global": "No",
            "Verdict": "Baseline only: 4-point sample misses the full-window peak.",
        },
        {
            "Algorithm": "Golden-section (full window)",
            "SearchRange": f"{min_pit_lap}-{max_pit_lap}",
            "BestPitLap": int(golden_full["best"]["PitLap"]),
            "XGB_WinPct": golden_full["best"]["XGB_WinPct"],
            "GP_WinPct": golden_full["best"]["GP_WinPct"],
            "Ensemble_WinPct": golden_full["best"]["Ensemble_WinPct"],
            "XGB_MedianGap": golden_full["best"]["XGB_MedianGap"],
            "GP_MedianGap": golden_full["best"]["GP_MedianGap"],
            "Evaluations": golden_full["evaluations"],
            "Runtime_s": golden_full["runtime_s"],
            "HandlesInteger": "No (continuous relaxation)",
            "NeedsDerivative": "No",
            "Global": "No (unimodal assumption)",
            "Verdict": "Finds the peak here, but cannot certify it; bracket-sensitive.",
        },
        {
            "Algorithm": "Golden-section (old grid bracket)",
            "SearchRange": f"{old_low}-{old_high}",
            "BestPitLap": int(golden_old["best"]["PitLap"]),
            "XGB_WinPct": golden_old["best"]["XGB_WinPct"],
            "GP_WinPct": golden_old["best"]["GP_WinPct"],
            "Ensemble_WinPct": golden_old["best"]["Ensemble_WinPct"],
            "XGB_MedianGap": golden_old["best"]["XGB_MedianGap"],
            "GP_MedianGap": golden_old["best"]["GP_MedianGap"],
            "Evaluations": golden_old["evaluations"],
            "Runtime_s": golden_old["runtime_s"],
            "HandlesInteger": "No (continuous relaxation)",
            "NeedsDerivative": "No",
            "Global": "No (unimodal assumption)",
            "Verdict": "Trapped at the old bracket edge: exactly the grid bias.",
        },
    ]

    newton_best = newton_run["from_grid_result"] or newton_run["best"]
    if newton_best is not None:
        newton_row = {
            "Algorithm": "Newton-Raphson (local quadratic surrogate)",
            "SearchRange": "local surrogate around start",
            "BestPitLap": int(newton_best["PitLap"]),
            "XGB_WinPct": newton_best["XGB_WinPct"],
            "GP_WinPct": newton_best["GP_WinPct"],
            "Ensemble_WinPct": newton_best["Ensemble_WinPct"],
            "XGB_MedianGap": newton_best["XGB_MedianGap"],
            "GP_MedianGap": newton_best["GP_MedianGap"],
            "Evaluations": newton_run["evaluations"],
            "Runtime_s": newton_run["runtime_s"],
            "HandlesInteger": "No",
            "NeedsDerivative": "Yes",
            "Global": "No (local)",
            "Verdict": (
                f"From the old-grid start (lap 42) it lands on lap "
                f"{newton_best['PitLap']}; {newton_run['root_note']}; rejected."
            ),
        }
    else:
        newton_row = {
            "Algorithm": "Newton-Raphson (local quadratic surrogate)",
            "SearchRange": "local surrogate around start",
            "BestPitLap": np.nan,
            "XGB_WinPct": np.nan,
            "GP_WinPct": np.nan,
            "Ensemble_WinPct": np.nan,
            "XGB_MedianGap": np.nan,
            "GP_MedianGap": np.nan,
            "Evaluations": newton_run["evaluations"],
            "Runtime_s": newton_run["runtime_s"],
            "HandlesInteger": "No",
            "NeedsDerivative": "Yes",
            "Global": "No (local)",
            "Verdict": f"No stable feasible root: {newton_run['root_note']}; rejected.",
        }
    comparison_rows.append(newton_row)
    comparison_rows.append(
        {
            "Algorithm": "Differential Evolution (selected)",
            "SearchRange": f"{min_pit_lap}-{max_pit_lap}",
            "BestPitLap": int(best["PitLap"]),
            "XGB_WinPct": best["XGB_WinPct"],
            "GP_WinPct": best["GP_WinPct"],
            "Ensemble_WinPct": best["Ensemble_WinPct"],
            "XGB_MedianGap": best["XGB_MedianGap"],
            "GP_MedianGap": best["GP_MedianGap"],
            "Evaluations": de_best["unique_evaluations"],
            "Runtime_s": de_best["runtime_s"],
            "HandlesInteger": "Yes",
            "NeedsDerivative": "No",
            "Global": "Yes (population-based)",
            "Verdict": "Selected: global, integer-native, assumption-light.",
        }
    )
    comparison = pd.DataFrame(comparison_rows)

    convergence = de_best["history"].copy()
    if convergence.empty:
        convergence = pd.DataFrame(
            [
                {
                    "Generation": 1,
                    "BestPitLap": int(best["PitLap"]),
                    "BestObjective": best["Objective"],
                    "BestWinPct": best["Ensemble_WinPct"],
                    "BestMedianGap": best["Ensemble_MedianGap"],
                    "Convergence": 0.0,
                }
            ]
        )
    de_seed_frame = pd.DataFrame(
        [
            {
                "Seed": int(run["seed"]),
                "BestPitLap": int(run["best"]["PitLap"]),
                "Objective": run["best"]["Objective"],
                "XGB_WinPct": run["best"]["XGB_WinPct"],
                "GP_WinPct": run["best"]["GP_WinPct"],
                "Ensemble_WinPct": run["best"]["Ensemble_WinPct"],
                "MedianGap": run["best"]["Ensemble_MedianGap"],
                "UniqueEvaluations": run["unique_evaluations"],
                "NFev": run["nfev"],
                "NIter": run["nit"],
                "Runtime_s": run["runtime_s"],
            }
            for run in de_runs
        ]
    )

    improvement_pp = 100.0 * (best["Ensemble_WinPct"] - grid_best["Ensemble_WinPct"])
    optimization_config = {
        "algorithm": "Differential Evolution",
        "objective": (
            "Mean win probability across the XGBoost and Gaussian Process engines, "
            "tie-broken by the mean median final gap"
        ),
        "pit_lap_bounds": [min_pit_lap, max_pit_lap],
        "best_pit_lap": int(best["PitLap"]),
        "best_offset": int(best["PitLap"] - current_lap),
        "xgb_win_pct": float(best["XGB_WinPct"]),
        "gp_win_pct": float(best["GP_WinPct"]),
        "ensemble_win_pct": float(best["Ensemble_WinPct"]),
        "xgb_median_gap": float(best["XGB_MedianGap"]),
        "gp_median_gap": float(best["GP_MedianGap"]),
        "xgb_p10_gap": float(best["XGB_P10"]),
        "xgb_p90_gap": float(best["XGB_P90"]),
        "gp_p10_gap": float(best["GP_P10"]),
        "gp_p90_gap": float(best["GP_P90"]),
        "grid_best_pit_lap": int(grid_best["PitLap"]),
        "grid_best_xgb_win_pct": float(grid_best["XGB_WinPct"]),
        "grid_best_gp_win_pct": float(grid_best["GP_WinPct"]),
        "grid_best_ensemble_win_pct": float(grid_best["Ensemble_WinPct"]),
        "grid_best_xgb_median_gap": float(grid_best["XGB_MedianGap"]),
        "grid_best_gp_median_gap": float(grid_best["GP_MedianGap"]),
        "improvement_pp": float(improvement_pp),
        "xgb_median_gap_gain": float(
            best["XGB_MedianGap"] - grid_best["XGB_MedianGap"]
        ),
        "gp_median_gap_gain": float(
            best["GP_MedianGap"] - grid_best["GP_MedianGap"]
        ),
        "xgb_only_pit_lap": int(xgb_row["PitLap"]),
        "gp_only_pit_lap": int(gp_row["PitLap"]),
        "ensemble_landscape_pit_lap": int(ensemble_row["PitLap"]),
        "n_simulations": int(n_sims),
        "validation_simulations": int(validation_sims),
        "validation_seeds": [int(s) for s in DEFAULT_VALIDATION_SEEDS],
        "n_evaluations": int(de_best["unique_evaluations"]),
        "n_function_evals": int(de_best["nfev"]),
        "n_iterations": int(de_best["nit"]),
        "runtime_s": float(de_best["runtime_s"]),
        "seed": int(seed),
        "de_seeds": [int(s) for s in DEFAULT_DE_SEEDS],
        "popsize": int(popsize),
        "maxiter": int(maxiter),
        "common_random_numbers": True,
    }

    result = OptimizationResult(
        landscape=landscape,
        comparison=comparison,
        convergence=convergence,
        de_seeds=de_seed_frame,
        validation_seeds=validation_seeds,
        validation_summary=validation_summary,
        config=optimization_config,
        best=best,
        de_best=de_best,
        de_runs=de_runs,
        grid_best=grid_best,
        newton_roots=newton_run["roots"],
    )

    if save:
        result.save(artifact_dir)
        config["OPTIMIZATION"] = optimization_config
        (artifact_dir / "config.json").write_text(
            json.dumps(config, indent=2, default=str)
        )

    if verbose:
        print("=" * 78)
        print("FULL-WINDOW PIT-STRATEGY OPTIMIZATION")
        print("=" * 78)
        print(f"Baseline grid best : lap {grid_best['PitLap']} | "
              f"ensemble {_format_pct(grid_best['Ensemble_WinPct'])} | "
              f"XGB {_format_pct(grid_best['XGB_WinPct'])} | "
              f"GP {_format_pct(grid_best['GP_WinPct'])}")
        print(f"DE optimum         : lap {best['PitLap']} | "
              f"ensemble {_format_pct(best['Ensemble_WinPct'])} | "
              f"XGB {_format_pct(best['XGB_WinPct'])} | "
              f"GP {_format_pct(best['GP_WinPct'])}")
        print(f"Gain               : {improvement_pp:+.2f} pp ensemble win | "
              f"XGB margin {best['XGB_MedianGap'] - grid_best['XGB_MedianGap']:+.2f} s | "
              f"GP margin {best['GP_MedianGap'] - grid_best['GP_MedianGap']:+.2f} s")
        print(f"DE effort          : {de_best['unique_evaluations']} unique lap evaluations, "
              f"{de_best['runtime_s']:.2f} s")
        print(f"DE seeds           : {', '.join(str(s) for s in DEFAULT_DE_SEEDS)}")
        print("=" * 78)

    return result


@dataclass
class OptimizationResult:
    landscape: pd.DataFrame
    comparison: pd.DataFrame
    convergence: pd.DataFrame
    de_seeds: pd.DataFrame
    validation_seeds: pd.DataFrame
    validation_summary: pd.DataFrame
    config: Dict[str, Any]
    best: Dict[str, Any]
    de_best: Dict[str, Any]
    de_runs: List[Dict[str, Any]]
    grid_best: Dict[str, Any]
    newton_roots: pd.DataFrame

    def save(self, artifact_dir: Path) -> None:
        artifact_dir = Path(artifact_dir)
        self.landscape.to_csv(artifact_dir / "optimization_landscape.csv", index=False)
        self.comparison.to_csv(
            artifact_dir / "optimization_comparison.csv", index=False
        )
        self.convergence.to_csv(
            artifact_dir / "optimization_convergence.csv", index=False
        )
        self.de_seeds.to_csv(artifact_dir / "optimization_de_seeds.csv", index=False)
        self.validation_seeds.to_csv(
            artifact_dir / "optimization_validation_seeds.csv", index=False
        )
        self.validation_summary.to_csv(
            artifact_dir / "optimization_validation.csv", index=False
        )
        self.newton_roots.to_csv(
            artifact_dir / "optimization_newton_roots.csv", index=False
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize the Bahrain 2023 pit lap with Differential Evolution."
    )
    parser.add_argument("--artifact-dir", default=str(ARTIFACT_DIR))
    parser.add_argument("--n-sims", type=int, default=DEFAULT_OPT_SIMULATIONS)
    parser.add_argument(
        "--validation-sims", type=int, default=DEFAULT_VALIDATION_SIMULATIONS
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--popsize", type=int, default=12)
    parser.add_argument("--maxiter", type=int, default=30)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    run_full_optimization(
        artifact_dir=args.artifact_dir,
        n_sims=args.n_sims,
        validation_sims=args.validation_sims,
        seed=args.seed,
        popsize=args.popsize,
        maxiter=args.maxiter,
        tolerance=args.tolerance,
        save=not args.no_save,
    )


if __name__ == "__main__":
    main()
