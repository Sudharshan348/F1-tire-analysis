# F1 Tire Degradation & Performance Decay Analysis
## Complete Project Documentation — Bahrain Grand Prix 2023

**Authors:** B Sudharshan, Arush Srivastava, Dharanitharan D
**Course context:** Predictive Analytics Project
**Deliverables:** Jupyter notebook (analysis), Streamlit dashboard (presentation), FastF1 data pipeline

---

## 1. Executive Summary (the 60-second pitch)

This project answers a single, high-stakes question that every Formula 1 pit wall faces
several times per race:

> *"When exactly should we pit our driver, given how the tires are actually degrading,
> to finish ahead of the rival car?"*

We take raw F1 timing and telemetry (FastF1, Bahrain GP 2023), remove every confound that
is **not** tire wear (fuel load, safety cars, pit laps, traffic), and train **five**
regression models to predict the pure lap-time penalty caused by tire aging. We then feed
those predictions into a **Monte Carlo race simulator** that prices in pit-stop loss, rival
behavior, and safety-car risk. The baseline strategy study samples four pit laps
(`{42, 45, 48, 51}`); the final layer replaces that hand-designed grid with a full-window
**Differential Evolution** optimizer over laps 31–56. It raises the dual-engine win
probability from **78.0%** at the grid's best lap to **99.99%** at **lap 34**, and adds
**+5.1 s** (XGBoost) / **+18.7 s** (GP) of median final margin. The output is a
probabilistic optimal pit lap, an uncertainty band, a convergence/landscape proof, and a
live what-if dashboard.

The novel parts are: (a) physics-based fuel correction, (b) a **custom asymmetric loss
function** that punishes the specific mistakes that cost races (optimistic tire
predictions), (c) **strategy-aware evaluation metrics** instead of plain RMSE,
(d) a **dual-engine uncertainty study** comparing fixed-noise vs Gaussian-Process
per-lap uncertainty in the race simulation, and (e) **full-window stochastic strategy
optimization** with Differential Evolution, common random numbers, and independent-seed
validation. **Section 20 gives the short before/after summary of exactly what was added in
the final version.**

---

## 2. Problem Statement & Motivation

### 2.1 Why tire degradation matters

In modern F1, overtaking is hard. Races are frequently decided by the **undercut**: pit
earlier than your rival, use the fresh-tire pace advantage to jump them when they pit.
But pit too early and your tire falls off a "cliff" before the end; pit too late and the
rival jumps you. The decision margin is often **under 2 seconds over a 90-minute race**.

Human intuition cannot reliably separate:
- How much lap time is lost to **fuel burn** (the car gets ~0.1 s/lap faster all race)
- How much is **tire degradation** (non-linear, compound-specific)
- How much is **dirty air** (following another car)
- How much is **driver/car baseline pace differences**

Our project separates these signals and converts the tire signal into a
**probabilistic strategy decision**.

### 2.2 The decision problem formalized

```
Given:
  - Current lap L=30 of 57 (27 laps remaining)
  - We are 2.1 s BEHIND our rival on track
  - Both cars must make one final pit stop (22 s time loss)
  - Rival's baseline plan: pit at lap 45
  - Tire degradation model: delta(age, compound, temperature)

Choose:
  baseline grid:  p ∈ {42, 45, 48, 51}
  final optimizer: p ∈ {31, 32, ..., 56}

Maximize:
  J(p) = 1/2 [ P_XGB(win | p) + P_GP(win | p) ]   [dual-engine win probability]
```

The four-point grid is kept as the comparison baseline. The final recommendation comes
from the full-window Differential Evolution search in Section 8.6.

---

## 3. Dataset

| Property | Value |
|---|---|
| Source | FastF1 API — `fastf1.get_session(2023, 'Bahrain', 'R')` |
| Event | 2023 Bahrain Grand Prix, Race session (~57 laps) |
| Raw laps | all classified laps (timing, compounds, pit in/out, track status) |
| Weather | air temperature, track temperature, humidity per timestamp |
| Extra data used | pit stops, safety-car status, driver/team, positions |
| Arms-length artifacts | `F1_Bahrain_2023_Dataset.xlsx`, `cache/` (FastF1 HTTP cache) |

FastF1 merges official timing loops with telemetry, giving per-lap: lap time, compound
(SOFT/MEDIUM/HARD), tire life in laps, stint number, pit-in/pit-out times, track status
(green/yellow/SC/VSC/red), and `IsAccurate` flags for timing glitches.

---

## 4. End-to-End Pipeline (big picture)

```
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │                          RAW DATA (FastF1)                                   │
 │   lap times · compounds · tire life · pit times · track status · weather     │
 └───────────────────────────────────┬──────────────────────────────────────────┘
                                     │
                                     ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │                     DATA CLEANING  (keep only "pure" laps)                  │
 │  • IsAccurate == True              (drop timing glitches)                    │
 │  • remove Safety Car / VSC laps    (not representative pace)                 │
 │  • remove pit-in / pit-out laps    (contain pit-lane time)                   │
 │  • remove dirty-air laps           (gap to car ahead < 2 s)                  │
 │  • remove ±2.5σ outliers per stint (lock-ups, mistakes)                      │
 │  • require stints of ≥ 3 laps      (enough to fit a trend)                   │
 │                                                                              │
 │  RESULT: 564 laps · 19 drivers · 52 stints                                   │
 │          HARD 361 · SOFT 195 · MEDIUM 8                                      │
 └───────────────────────────────────┬──────────────────────────────────────────┘
                                     │
                                     ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │              FEATURE ENGINEERING  (physics + target design)                  │
 │                                                                              │
 │  1. Fuel correction  (remove fuel-burn trend)                                │
 │        FuelCorrectedLapTime = LapTime − (57 − LapNumber) × 0.1061            │
 │                                                                              │
 │  2. Target = LapTimeDelta = FuelCorrectedLapTime − min(same stint)           │
 │     ⇒ car/driver baseline pace cancels; what remains is TIRE WEAR            │
 │                                                                              │
 │  3. Features: TyreLife, TyreLife², TyreLife³, CompoundEncoded,               │
 │               TrackTemp, AirTemp, LapNumber                                  │
 └───────────────────────────────────┬──────────────────────────────────────────┘
                                     │
                                     ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │            LEAKAGE-SAFE SPLIT   (GroupShuffleSplit on StintID 80/20)         │
 │            train = 452 laps        test = 112 laps                           │
 └───────────────────────────────────┬──────────────────────────────────────────┘
                                     │
                                     ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │                    FIVE-MODEL ZOO  (regression)                              │
 │   Linear │ Polynomial(2) │ XGBoost(asymmetric) │ Random Forest │ GP           │
 └───────────────────────────────────┬──────────────────────────────────────────┘
                                     │
                                     ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │         MONTE CARLO RACE SIMULATOR   (XGBoost engine + GP engine)            │
 │   10,000 races × 4 baseline pit strategies × 2 engines = 80,000 races        │
 │   + pit loss 22 s + rival model + safety-car hazard (15% per 10-lap block)   │
 └───────────────────────────────────┬──────────────────────────────────────────┘
                                     │
                                     ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │        FULL-WINDOW STRATEGY OPTIMIZATION  (Differential Evolution)           │
 │   p ∈ {31..56} · dual-engine objective · common random numbers · 3 seeds     │
 │   50,000 races per candidate · benchmarked vs Newton and golden-section      │
 └───────────────────────────────────┬──────────────────────────────────────────┘
                                     │
                                     ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │     DECISION OUTPUT + VALIDATION + INTERACTIVE DASHBOARD (Streamlit)         │
 │   Optimal pit lap 34 · win% · P10–P90 gap bands · reality-check vs real pits │
 └──────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Data Processing — step by step

### 5.1 Building the "gap to car ahead" signal (clean vs dirty air)

FastF1 gives lap times, not *gaps*. We reconstruct gaps:

1. Sort all laps by driver and lap number, compute each driver's **cumulative race time**
   (running sum of lap times).
2. For each lap number, rank drivers by cumulative time; `GapToAhead` = difference to the
   driver directly in front.
3. Flag `CleanAir = True` if the gap is **> 2.0 s** (or the car is the leader).

Why 2 s? Aero research and the data itself: within 2 s, downforce collapses and the
following car slides, overheating its tires. The EDA boxplot confirms dirty-air laps are
systematically slower **and noisier** — keeping them in would poison the tire-degradation
model.

```
  Gap to car ahead          Clean air  (used for modeling)
 |----------------|-----------------------|--------------------------------->
 0s              2s                       time
  └──── dirty air: aero loss, overheated tires ────┘
```

### 5.2 The cleaning funnel

```
  All laps (~1,100+)
        │  IsAccurate == True
        ▼
  Accurate laps
        │  remove SC/VSC (TrackStatus 4,5,6,7)
        ▼
  Green-flag laps
        │  remove PitInTime / PitOutTime laps
        ▼
  Green-flag, on-track laps
        │  remove GapToAhead < 2 s  (dirty air)
        ▼
  Clean-air green laps
        │  remove |lap − stint mean| > 2.5σ  (driver errors)
        ▼
  Outlier-free laps
        │  keep stints with ≥ 3 laps
        ▼
  FINAL: 564 laps · 52 stints · 19 drivers
```

### 5.3 Physics-based fuel correction (the key preprocessing idea)

A 2023 F1 car starts a race with ≈ **110 kg** of fuel and burns ≈ **1.93 kg/lap**
(110 kg ÷ 57 laps). Burning fuel makes the car lighter, so lap times improve linearly
through a stint — roughly **0.055 s per kg** (a standard F1 rule-of-thumb; also
consistent with our data). Left uncorrected, this ~0.1 s/lap trend *cancels out* the
tire-wear trend the model is supposed to learn.

We remove it analytically:

```
FuelCorrectedLapTime(lap) = LapTime(lap) − (RACE_LAPS − LapNumber) × Δ_fuel

where  Δ_fuel = K × FuelPerLap = 0.055 × (110/57) ≈ 0.1061 s/lap
       (RACE_LAPS − LapNumber) = kg of fuel remaining relative to race start
```

```
 Lap time (s)
   ▲
   │\                          raw laps: fuels burns → faster, tires wear → slower
   │ \                         (two effects superimposed)
   │  \________
   │            \___/‾‾\___
   │
   └──────────────────────────────────────► lap
   ▲
   │                          after fuel correction:
   │                         the downward fuel slope is gone;
   │              ___/‾‾‾‾‾‾   what remains is the tire degradation curve
   │        __/‾‾
   │  _____/
   └──────────────────────────────────────► lap
```

### 5.4 Target variable: `LapTimeDelta`

Even after fuel correction, every car/driver has a different *baseline* pace
(Verstappen's "slow" lap may be faster than another car's "fast" lap). We therefore
predict a **relative** quantity:

```
LapTimeDelta(lap) = FuelCorrectedLapTime(lap) − min(FuelCorrectedLapTime over the stint)
```

This is the **pace drop-off relative to that stint's own best lap** — zero at the
freshest point, growing as tires age. It is 100% tire wear + track evolution, and is
comparable across all cars and drivers. It also makes the assumption of a per-stint
baseline explicit: not modeling absolute lap time isolates the degradation phenomenon
we care about.

### 5.5 Engineered features

| Feature | Meaning | Rationale |
|---|---|---|
| `TyreLife` | laps on current tire | the core wear signal |
| `TyreAgeSq` | TyreLife² | degradation accelerates — curvature |
| `TyreAgeCubed` | TyreLife³ | allows a "cliff" (sharp late-life fall-off) |
| `CompoundEncoded` | SOFT=0, MEDIUM=1, HARD=2 | different wear rates per compound |
| `TrackTemp` | °C | hot asphalt overheats tires → faster wear |
| `AirTemp` | °C | ambient thermal load |
| `LapNumber` | race lap | track evolution (rubber laid down = faster) |

Interactions (temperature × age, compound × age) are learned by tree models and by the
GP kernel; the polynomial model handles tyre-age curvature explicitly.

### 5.6 Leakage-safe train/test split

Laps within one stint are **highly correlated** (same tire, same fuel trend, same
driver). A random lap-wise split would put later laps of a stint in train and earlier
laps of the *same* stint in test — the model would effectively see its own future.

We use `GroupShuffleSplit(groups=StintID, test_size=0.2)`:
**all laps of a stint stay on one side.** Result: 452 train laps / 112 test laps,
and every reported metric is an honest *unseen-stint* generalization score.

```
 WRONG (random split):  stint A laps [1..10] → train, laps [11..15] → test  ✗ leak
 RIGHT (group split):   stint A [all laps] → train,  stint B [all laps] → test  ✓
```

---

## 6. Exploratory Data Analysis — what we found

1. **Compound degradation curves (LOWESS-smoothed `LapTimeDelta` vs tire age).**
   - SOFT degrades fastest and accelerates around **tire age ≈ 16** — the "soft cliff".
   - HARD is flatter but slower baseline; MEDIUM offers the flattest long-run slope,
     which is why a Medium→Hard one-stop is the reference strategy.
   - We smooth with LOWESS (locally-weighted regression) because raw lap deltas are
     noisy; LOWESS makes the trend and its derivative (degradation *rate* in s/lap)
     visible.

2. **Clean air vs dirty air boxplot.** Dirty-air laps: higher median lap time, much
   larger variance → justifies the clean-air filter.

3. **Track temperature × tire age heatmap.** The mean pace penalty is largest in the
   high-temperature × old-tire corner and grows **multiplicatively** — thermal load and
   tire age compound, they are not independent additive effects. This motivates giving
   temperature features to the models (and why the GP learns a long length-scale on
   temperature… it matters mostly in interaction with age).

---

## 7. The Five Models — what each one is and what it does here

All five predict the same target (`LapTimeDelta`) on the same leakage-safe test set so
the comparison is apples-to-apples. They were chosen to represent five distinct
modeling families: **linear → polynomial → bagging → boosting → kernel/Bayesian**.

### 7.1 Model 1 — Linear Regression (baseline)

- **What it is:** ordinary least squares, one coefficient per feature.
- **Role:** the sanity floor. If a sophisticated model cannot beat a straight line, the
  features or the problem framing are wrong.
- **Result:** RMSE 0.680 s, R² 0.460 — a surprisingly strong baseline because the
  nonlinear age features (age², age³) are already in the design matrix.

### 7.2 Model 2 — Polynomial Regression, degree 2

- **What it is:** linear regression on polynomial expansions of **tire-age features
  only** (`TyreLife`, `TyreLife²`, `TyreLife³`, plus their cross terms); the other
  features enter linearly. Degree chosen by evidence, not by eye.
- **Degree selection (AIC / BIC / 5-fold CV):**

  | degree | AIC | BIC | CV-RMSE |
  |---|---|---|---|
  | 1 | 624.9 | 632.7 | 0.5917 |
  | 2 | 620.7 | **632.4** ← lowest BIC | 0.6065 |
  | 3 | 616.9 | 632.4 | 0.5907 |
  | 4 | **616.4** ← lowest AIC | 635.9 | 0.5854 |
  | 5 | 618.0 | 641.4 | **0.5839** ← lowest CV |
  | 6 | 650.3 | 677.5 | 0.6318 |

  BIC (which penalizes complexity harder) is minimized at **degree 2**; AIC/CV improve
  only marginally beyond it. We choose **degree 2** — the quadratic curvature captures
  the wear ramp, and higher degrees chase noise (degree 6 collapses).
- **Result:** RMSE 0.822 s — the weakest. With the interaction terms restricted and fit
  globally, the polynomial cannot represent the cliff, compounds, and thermal effects
  flexibly. (Its high SIS confirms: a rigid global curve is dangerous around the cliff.)
- **Why keep it:** it is the interpretable, closed-form "textbook" model that shows F1
  degradation is fundamentally non-linear, and it motivates the flexible learners.

### 7.3 Model 3 — XGBoost with a custom **asymmetric** loss (primary model)

- **What it is:** gradient-boosted decision trees. Each tree fits the gradient of a
  loss; 196 shallow trees (depth 3) with learning rate 0.011 add up to a flexible
  non-linear regression.
- **The twist — asymmetric loss.** Standard MSE penalizes under- and over-prediction
  equally. In strategy, they are **not** equally costly:
  - If the model *under-predicts* degradation (says the tire is faster than it really
    is: `pred < true`), the strategist leaves the driver out too long → the **cliff**
    hits mid-stint → seconds lost and track position surrendered.
  - If the model *over-predicts* degradation (`pred > true`), the team pits slightly
    early → a small, affordable cost.
  
  We implement gradient and Hessian of an asymmetric squared loss with
  **α_under = 2.5, α_over = 1.0**:

  ```
                      ⎧ 2.5 · (pred − true)   if pred < true  (optimism → 2.5× punishment)
  loss' = ⎨
                      ⎩ 1.0 · (pred − true)   if pred > true

                      ⎧ 2.5   if pred < true
  loss'' = ⎨
                      ⎩ 1.0   if pred > true
  ```

  ```
  Penalty
     ▲
     │          /  slope 2.5  (under-prediction of degradation = optimistic = BAD)
     │         /
     │________/__________► residual (pred − true)
     │        \
     │         \  slope 1.0  (over-prediction = conservative = cheap)
  ```

- **Optimization:** Optuna (TPE Bayesian search), **100 trials**, each scored by 5-fold
  CV RMSE on the training split, then early stopping (20 rounds) on a held-out
  validation slice. Best found: `n_estimators=196, max_depth=3, lr=0.011,
  subsample=0.52, colsample=0.64, reg_alpha≈1.6e-8, reg_lambda≈3.5e-6`.
- **Result:** RMSE 0.663 s, SIS **2.58 s** (best among non-GP models) — it trades a
  negligible amount of raw accuracy for fewer strategy-critical optimistic errors.
- **Also:** residual standard deviation **σ = 0.662 s** is exported and used as the
  fixed lap-noise in the XGBoost Monte Carlo engine.

### 7.4 Model 4 — Random Forest (bagging ensemble)

- **What it is:** 224 deep-ish trees (max_depth 5) trained on bootstrap samples with
  random feature subsets, predictions averaged. "Bagging" reduces variance; recursive
  splits capture interactions (age × compound, age × temperature) with zero manual
  feature design.
- **Optimization:** a lighter Optuna sweep, **30 trials**, same 5-fold CV protocol.
- **Result:** **best raw accuracy** — RMSE 0.648 s, MAE 0.541, R² 0.511. It is robust
  to residual outlier laps. Its SIS (2.72 s) is worse than XGBoost's, which neatly
  demonstrates the project's point: **lowest RMSE ≠ best strategy model**, because RF
  optimizes symmetric squared error.

### 7.5 Model 5 — Gaussian Process Regression (uncertainty engine)

- **What it is:** a Bayesian non-parametric regressor that treats the degradation curve
  as a draw from a distribution over smooth functions, defined by a kernel. Trained by
  maximizing the marginal likelihood; predictions come with a **per-point standard
  deviation σ(x)**.
- **Kernel (learned):**

  ```
  k(x,x') = 1.24² · Matérn₅/₂(length_scales=[1.09, 1.44, 1000, 3.71, 1000, 1000, 0.42])
            + WhiteKernel(noise = 0.171)
  ```

  - **Matérn 5/2**: twice-differentiable functions — a realistic prior for physical
    degradation curves.
  - Length scales tell a story: short scales (≈1.1, 1.4, 0.4) on age, age², LapNumber →
    those features drive variation; huge scales (1000) on TrackTemp/AirTemp and age³ →
    those features are (after GP feature standardization) nearly irrelevant *on their
    own*.
  - **WhiteKernel** absorbs lap-to-lap measurement noise.
- **Why it is in the project:** it is the only model that natively answers
  *"how sure am I?"*. Its per-lap σ grows on old tires (sparse data, cliff region) —
  exactly where strategy risk lives.
- **Result:** RMSE 0.649 s, MAE **0.531 s (best)**, and **SIS 2.15 s (best overall)**.
  It is also 25.7 s to train (vs 0.002 s for linear) — the honest cost of uncertainty.
- **Use in simulation:** its per-lap mean and σ seed the uncertainty-aware Monte Carlo
  engine (Section 9.3).

### 7.6 Head-to-head results (test set, 112 laps, unseen stints)

| Model | RMSE (s) | MAE (s) | R² | TW-RMSE (s) | **SIS (s)** | Train time (s) |
|---|---|---|---|---|---|---|
| Baseline (Linear) | 0.680 | 0.545 | 0.460 | 0.974 | 3.02 | 0.002 |
| Polynomial (Deg 2) | 0.822 | 0.651 | 0.213 | 1.149 | 3.65 | 0.002 |
| XGBoost (Asymmetric) | 0.663 | 0.541 | 0.487 | 0.939 | 2.58 | 1.02 |
| Random Forest | **0.648** | 0.541 | **0.511** | **0.919** | 2.72 | 0.32 |
| Gaussian Process | 0.649 | **0.531** | 0.508 | 0.922 | **2.15** | 25.7 |

### 7.7 Strategy-aware metrics (why we invented/used them)

Standard ML metrics don't know what a mistake *costs* in a race. We define two:

**1. TW-RMSE — Tire-age Weighted RMSE**

```
TW-RMSE = sqrt( mean( w(age) · (true − pred)² )),   w(age) = 1 + 3 · (age/max_age)²
```

Errors on old tires (near/at the cliff, where the pit decision happens) count up to
**4×** more than errors on fresh tires.

**2. SIS — Strategy Impact Score (seconds per stint)**

For each stint: sum the positive residuals only (`true > pred`, i.e. the model was
*optimistic* and thought the tire was faster than reality). Average per stint.

```
SIS = mean over stints [ Σ_laps max(0, true − pred) ]
```

This is the **total race time a strategist would lose by trusting the model's
optimism.** It is the single most important number in the project: the asymmetric
XGBoost and the GP win here — GP best at 2.15 s/stint vs 3.65 s for the polynomial.
A model can be worse at RMSE yet better at SIS (or vice versa) — that asymmetry is the
project's core insight.

---

## 8. Optimization — Everything Being Optimized, and How

There are **six distinct optimization problems** in the project. It is worth being
precise about each, because they operate at different levels.

### 8.1 Level 1 — Hyperparameter optimization (model capacity)

| Model | Search method | Trials | Objective | Result |
|---|---|---|---|---|
| XGBoost | Optuna TPE (Bayesian) | 100 | maximize −5-fold CV RMSE | 196 trees, depth 3, lr 0.011, subsample 0.52 |
| Random Forest | Optuna TPE | 30 | maximize −5-fold CV RMSE | 224 trees, depth 5, max_features 0.37 |

**How TPE works (professor-friendly explanation):** instead of grid search (exponential
cost) or random search (wasteful), Tree-structured Parzen Estimation models the
distribution of "good" hyperparameter settings from the trials so far and samples new
candidates likely to beat them. 100 trials explores far more of the space than a grid.

**Cross-validation:** each trial is scored by 5-fold CV RMSE — the model is trained 5×
on different folds and scored on the held-out fold, so hyperparameters cannot cheat by
memorizing.

### 8.2 Level 2 — Model/feature-structure optimization (polynomial degree)

Degree 1–6 compared by **AIC** (fits + light complexity penalty), **BIC** (heavier
complexity penalty), and **5-fold CV-RMSE**. Optimization objective: minimize
information criteria + out-of-fold error. Chosen: **degree 2** (BIC minimum; the
parsimonious elbow).

### 8.3 Level 3 — Kernel hyperparameter optimization (Gaussian Process)

The GP does not use random search: it maximizes the **log marginal likelihood**
`log p(y | X, θ)` with respect to kernel parameters (constant scale, seven length
scales, white-noise level) using 2 restarts of gradient-based optimization. This is
Bayesian model selection — automatically balancing fit against complexity, which is why
nonsense features get near-infinite length scales (effectively switched off).

### 8.4 Level 4 — Loss-function optimization (decision-theoretic risk tuning)

The asymmetric gradient/Hessian given to XGBoost is not just a metric — it *changes
what the model optimizes during training*. By setting α_under = 2.5 > α_over = 1.0, the
boosting procedure spends its capacity reducing optimistic errors. This is
**cost-sensitive learning**: encoding the real-world cost matrix of the pit-wall
decision into the training objective. Small change in code, fundamental change in what
"best model" means.

### 8.5 Level 5 — Baseline strategy grid (the original race decision)

The original top-level optimization was:

```
maximize    P( win | p )                     [over pit lap p]
subject to  p ∈ {42, 45, 48, 51}
            race dynamics (fuel, degradation, pit loss, rival, safety cars)
```

`P(win | p)` cannot be computed in closed form (the degradation curve is non-linear,
noise is stochastic, safety cars are discrete random events), so we estimate it by
Monte Carlo: 10,000 independent race simulations per candidate p, then compare the
empirical win rates. This is **simulation-based optimization / sample average
approximation**.

The grid is a useful baseline, but it is a hand-designed sample of only **4 of the 26
feasible pit laps**. Its best point, lap 42, is therefore not the true optimum. The final
layer in Section 8.6 removes that limitation.

### 8.6 Level 6 — Full-window pit-lap optimization (Differential Evolution)

This is the final optimization layer and the answer to the faculty's "actual optimization
algorithm" requirement. Instead of comparing four hand-picked laps, we solve

```
maximize    J(p) = 1/2 [ P_XGB(win | p) + P_GP(win | p) ] + 1e-6 * mean_median_gap(p)
subject to  p ∈ {31, 32, ..., 56}          (integer pit lap, current lap 30)
            one-stop strategy, rival pits lap 45, 22 s pit loss, 15% SC hazard
```

The tiny second term only breaks ties inside the win-probability plateau. The primary
objective is the **average win probability across the fixed-noise XGBoost engine and the
heteroscedastic Gaussian Process engine**. This is a deliberately model-robust objective:
the project's two simulation engines disagree about confidence, so optimizing one alone
would bake that engine's assumptions into the recommendation.

#### Why Differential Evolution?

`J(p)` is a **black-box, integer-valued, non-smooth, noisy, and potentially multi-modal**
function:

- the empirical win rate is a step function of `p`;
- pit lap is an integer, not a continuous variable;
- each evaluation is a Monte Carlo estimate;
- under the GP engine there is a second local maximum near lap 54, so the landscape is not
  guaranteed unimodal.

| Method | Derivative needed | Handles integer `p` | Global | Handles noisy black-box `J` | Verdict |
|---|---|---|---|---|---|
| Newton–Raphson | Yes (1st/2nd) | No | No | No | Not suitable |
| Golden-section | No | No (continuous relaxation) | No (unimodal only) | Weak | Benchmark only |
| **Differential Evolution** | No | **Yes** (`integrality=True`) | **Yes** | **Yes** | **Selected** |

DE is a population-based evolutionary algorithm from the genetic/GA family. SciPy's
`differential_evolution` supports bounds, integer variables, a deterministic seed, and a
callback, which is exactly what this problem needs.

#### Removing Monte Carlo noise: common random numbers

Every candidate pit lap is evaluated on the **same** safety-car draws and the same
standard-normal lap-noise draws. This is a sample-average approximation with **common
random numbers (CRN)**. Without CRN, a lucky seed can make a worse pit lap look better;
with CRN, the ranking reflects the strategy. Independent seeds are used only for final
validation, so the optimizer cannot overfit the shared sample.

The optimizer uses the same vectorized state simulator as the dashboard's live what-if
(Section 11), extended to both engines, so both engines share the safety-car and noise
draws. The baseline grid tables in Section 9 use the notebook's original per-race loop;
the small numerical differences between the two are Monte Carlo sampling, not a modelling
change.

The implementation is `optimizer.py`; the notebook calls it at the end of the pipeline and
the dashboard reads its exported artifacts.

#### Benchmark — all three candidate algorithms

| Method | Search range | Best lap | XGB win | GP win | Ensemble win | Evaluations | Verdict |
|---|---|---|---|---|---|---|---|
| Grid search (baseline) | 42–51 (4 points) | 42 | 99.64% | 56.35% | 78.00% | 4 | Misses the full-window peak |
| Golden-section (full window) | 31–56 | 34 | 99.99% | 99.98% | 99.99% | 9 | Finds the peak here, but cannot certify it |
| Golden-section (old grid bracket) | 42–51 | 42 | 99.64% | 56.35% | 78.00% | 7 | Trapped at the bracket edge |
| Newton–Raphson (quadratic surrogate) | local, from lap 42 | 55 | 0.04% | 59.07% | 29.56% | 31 | Start-dependent; roots ranged 5–148 |
| **Differential Evolution** | **31–56** | **34** | **99.99%** | **99.98%** | **99.99%** | **19** | **Selected** |

The Newton row is especially instructive: started from the old grid's lap 42, it lands on
lap 55 and collapses. The roots from different starts ranged from 5 to 148, i.e. outside
the race. Golden-section only matches DE when it is given the full bracket; seeded with the
old grid window it returns the bracket edge, which is exactly the grid bias we are trying
to remove.

*Evaluation counts are not directly comparable across methods: DE and golden-section
consume full Monte Carlo evaluations, while Newton consumes local quadratic-surrogate
points. The verdict column, not the raw count, is the comparison.*

#### Independent validation (5 fresh seeds × 20,000 races)

| Scenario | Pit lap | XGB win | GP win | Ensemble win | XGB median gap | GP median gap |
|---|---|---|---|---|---|---|
| Grid best (baseline) | 42 | 99.59% | 56.50% | 78.05% | +12.94 s | +1.07 s |
| **DE optimum** | **34** | **99.99%** | **99.97%** | **99.98%** | **+18.00 s** | **+19.73 s** |
| XGB-only optimum | 36 | 99.99% | 99.42% | 99.71% | +18.93 s | +14.57 s |
| GP-only optimum | 33 | 99.97% | 99.98% | 99.98% | +16.92 s | +20.37 s |

The gain from the baseline is **+21.99 percentage points** of ensemble win probability,
**+5.06 s** of XGB median margin, and **+18.66 s** of GP median margin. DE reaches the
answer in **19 unique pit-lap evaluations**, **3 generations**, and under **1.3 s** with
50,000 races per candidate; all three DE seeds (42, 7, 2026) return lap 34.

Because the paired validation uses common random numbers, the XGB median-gap gain is
deterministic (the fixed residual noise cancels between the two strategies); the GP gain
varies across seeds because its per-lap σ depends on the tire state. The win-probability
comparison still varies across seeds and is reported with 95% confidence intervals.

The three objective variants — XGB-only (lap 36), GP-only (lap 33), and dual-engine (lap
34) — all select the same **33–36 early-stop neighborhood**. The recommendation is not a
single-engine artifact, a boundary artifact (lap 34 is interior to 31–56), or Monte Carlo
noise (the gain survives independent seeds).

#### Why this matters

For a one-dimensional problem, an exhaustive 26-point scan is also cheap. The value of DE
is that it treats `J` as a black box, handles integer/non-smooth/noisy functions, and
extends naturally to multi-stop or compound-choice decisions where exhaustive search
explodes. The proof is not "DE is faster than a 26-point scan"; it is "DE finds the true
full-window optimum without assuming differentiability, unimodality, or a lucky seed."

### 8.7 Bonus — Computational optimization

Naïvely, the GP simulation needs a GP query for every lap of every simulation:
10,000 × 27 laps × 4 strategies × 2 cars ≈ 2.16 million GP queries (~hours). Because
the car's state sequence for a *given strategy is deterministic* (pit lap is fixed, so
tire age/compound/lap at every future lap are known), we:

1. Precompute the state matrix for each strategy once (216 unique lap-states total),
2. Query the GP **once in a single batch**,
3. Vectorize all 10,000 simulations as NumPy arrays (sample noise from the per-lap σ
   broadcast across runs).

This turns hours into seconds and is why the interactive "what-if" simulator in the
dashboard can run 10,000 races live on a button click.

---

## 9. Monte Carlo Race Simulation — the core decision engine

### 9.1 Race model

At lap 30, both cars have 27 laps left. Each simulated "race" walks lap by lap:

```
STATE per car: (tire_age, compound, cumulative_time)
                                                                 
  For each future lap t = 31 … 57:
      1. tire_age += 1
      2. if t == our_pit_lap:  total += 22.0 s ; tire_age=1 ; compound=HARD
      3. lap_delta = model.predict(age, age², age³, compound, TrackTemp, AirTemp, t)
      4. total += lap_delta + N(0, σ_lap)
      5. safety-car hazard: every 10 laps, P(SC) = 0.15
         → if triggered: add 3 neutralized laps at 45 s to BOTH cars
  Final gap = (rival_total + 2.1 s head start) − our_total
  Win if gap > 0
```

The rival is modeled identically (starts on age-18 Mediums, pits at lap 45 for Hards).
The 2.1 s term encodes that the rival is physically ahead; positive final gap means we
overcame it.

### 9.2 Engine A — XGBoost with fixed residual noise

`N(0, σ_lap)` uses one constant **σ = 0.662 s** (the model's test residual std) on every
lap. Simple, but it assumes the same uncertainty on a fresh tire and on a 25-lap-old
tire — false, and dangerous exactly at the cliff.

**Results (10,000 sims per strategy):**

| Our pit lap | Win % | Median gap | P10 | P90 |
|---|---|---|---|---|
| **42** | **99.65%** | +12.92 s | +6.77 | +19.09 |
| 45 | 92.46% | +6.88 s | +0.72 | +13.21 |
| 48 | 51.89% | +0.27 s | −5.99 | +6.28 |
| 51 | 15.18% | −5.87 s | −12.39 | +1.65 |

### 9.3 Engine B — Gaussian Process with per-lap, heteroscedastic σ

Instead of one σ, each lap state (age, compound, lap…) gets its **own** predictive
σ(x) from the GP. On fresh tires σ is small; as the tire ages into sparse data near
the cliff, σ grows. The simulation then draws `N(0, σ(x_lap))` per lap.

**Results:**

| Our pit lap | Win % | Median gap | P10 | P90 |
|---|---|---|---|---|
| 42 | 57.0% | +1.15 s | −7.20 | +9.74 |
| 45 | 45.4% | −0.78 s | −9.87 | +8.22 |
| 48 | 50.6% | +0.10 s | −9.43 | +9.69 |
| 51 | 57.7% | +1.59 s | −8.50 | +11.32 |

> **These four-lap tables are the baseline grid.** The full-window Differential Evolution
> optimizer in Section 8.6 searches every feasible lap 31–56 and finds the dual-engine
> optimum at **lap 34**: 99.99% XGB win, 99.98% GP win, and median margins of +18.0 s /
> +19.7 s. The grid's lap 42 is dominated. The baseline tables are retained because they
> expose exactly why the grid was insufficient.

### 9.4 Interpreting the two engines (the intellectually interesting bit)

The engines **agree on structure but disagree on confidence**:

```
     Win probability
100% ┤  XGB ●
     │       ●
 75% ┤          ●
     │              ○────────○──○   GP: flat, ~45–58% everywhere
 50% ┤                  ●
     │  GP ○       ○
 25% ┤                      ●
     │  ○                      ○
  0% ┤                          ●  XGB
     └────┬────┬────┬────┬────►
         L42  L45  L48  L51     our pit lap
```

- XGBoost's fixed, *small* noise creates razor-sharp win probabilities (99.7% at lap
  42) — it mistakes low residual variance for high certainty about the cliff.
- The GP's age-growing σ is **honest**: it says "we have less data about 20+-lap-old
  tires, so the cliff's exact lap is uncertain", compressing all probabilities toward
  50% and widening P10–P90 to ~±9–10 s.
- Direction of the answer is the same (early-ish stop is defensible; pitting last is
  bad), but risk is correctly priced. **The GP engine is the "expected-loss" view; the
  XGB engine is the "point-estimate" view.** Reporting both is more honest than
  reporting either alone.
- The full-window optimizer confirms the direction and sharpens it: XGB-only prefers
  lap 36, GP-only prefers lap 33, and the dual-engine objective prefers lap 34. All are
  in the same early-stop neighborhood; the old grid's earliest point, lap 42, is 6–9
  laps too late.

Both engines are exposed side-by-side in the dashboard, plus a live what-if simulator
where the user moves the pit lap, current tire age, and rival gap sliders and re-runs
10,000 GP simulations in a click.

---

## 10. Validation — "Proofs" the result is trustworthy

### 10.1 Model tracks reality lap-by-lap (PER, stint 3)
For Perez's final 20-lap stint (laps 36–57), we plot cumulative predicted vs actual
fuel-corrected time. **Total drift = 0.58 s over 20 laps** — far below the ~2 s
decision margins F1 strategists work with. A model that accumulated 10 s of drift
would be useless; 0.58 s means pit-window calls are credible.

```
 Cumulative seconds in stint
   ▲                       actual
   │                    ●●●
   │                ●●●
   │            ●●●          model (XGBoost)
   │        ●●● ●●●
   │    ●●● ●●●
   │●●● ●●●
   └──────────────────────────────► lap 36 … 57
 Drift (s)
  + │    ▂▁▁▂▁▁▁▂▁▁▁
  0 ┤──────────────────────────────
  − │
```
(final |drift| = 0.58 s)

### 10.2 Recommendations match real pit decisions
Real teams made 11 final-window pit stops between laps **35 and 50** (median lap 41).
The original grid recommended lap 42, which sits inside that window. The full-window
optimizer pushes the recommendation to **lap 34** — one lap before the earliest real stop
and seven laps before the grid's choice. That is a deliberately aggressive undercut, but
it is data-consistent: the real strategy envelope starts at lap 35, and the optimizer is
responding to a specific rival pitting on lap 45, not to the field average.

```
 # real stops
 4 │          █
 3 │          █     █
 2 │    █     █     █
 1 │    █  █  █  █  █     █
 0 └──────────────────────────────► race lap
      35 37 39 41 43 45 47 50
      ▲ optimizer: lap 34 (aggressive edge)
                  ▲ old grid: lap 42
                  ▲ rival model pit: lap 45
```

### 10.3 Optimization provenance
Every design choice is evidence-backed, not eyeballed: polynomial degree by AIC/BIC/CV,
XGBoost by 100-trial Optuna, RF by 30-trial Optuna, GP kernel by marginal likelihood, the
baseline pit grid by Monte Carlo, and the final pit lap by **full-window Differential
Evolution** with common random numbers and independent-seed validation (`optimizer.py`).

---

## 11. The Streamlit Dashboard (presentation layer)

`app.py` is a **read-only viewer** of notebook-exported artifacts (`artifacts/`):
trained models (`joblib`), cleaned laps (`parquet`), and result tables (`csv`). It never
retrains — the notebook is the single source of truth.

| Tab | Contents |
|---|---|
| **Overview** | Key metrics + "key findings" cards (optimal undercut, soft cliff, uncertainty) |
| **Data & Method** | Fuel-correction equations, cleaning funnel, leakage-safe split, compound mix |
| **EDA** | LOWESS degradation curves, clean vs dirty air boxplot, temperature × age heatmap |
| **Models** | The 5-model metric table + RMSE/SIS bars + degree selection + PER stint trace with GP ±1σ |
| **Monte Carlo** | Baseline 4-point grid: win% by strategy per engine, gap distributions, **live what-if simulator** |
| **Optimizer** | Problem formulation, Newton/golden/DE benchmark, DE convergence, full-window landscape, independent validation, and "how the algorithm helps" |
| **Proof & Inference** | GP kernel/params, cumulative drift chart, real pit-stop comparison, uncertainty σ, final optimization proof |

The app demonstrates the whole story in seven clicks — ideal for a live demo/viva. The
Optimizer tab is the direct evidence for the faculty requirement: it shows the old grid's
failure, the chosen algorithm's convergence, and the validated gain.

---

## 12. Novelty — what makes this more than a "predict lap time" project

1. **Physics-informed target design.** Fuel-burn correction with a thermodynamic-style
   linear burn model + per-stint baseline subtraction isolates a physically meaningful
   quantity (pure tire wear) from several confounds. Most student projects regress raw
   lap times and inherit the fuel trend as a fake signal.

2. **Decision-theoretic loss function.** The custom asymmetric objective (2.5× penalty
   on optimism) encodes the *cost asymmetry of race strategy* directly into gradient
   boosting. Combined with SIS, the evaluation itself is strategy-aware. This is a
   genuine transfer from operations research (newsvendor/asymmetric costs) into sports
   analytics.

3. **Dual-engine uncertainty propagation.** Systematically comparing a fixed-noise
   simulation against a heteroscedastic GP simulation quantifies how model uncertainty
   reshapes a strategy ranking — a first-class result (99.7% vs 57%!), not a footnote.
   The GP's per-lap σ is a native Bayesian output used as simulation input, closing the
   loop model → uncertainty → decision.

4. **Stint-grouped leakage-safe protocol** plus clean/dirty-air separation — prevents
   the two most common hidden leaks in motorsport ML.

5. **Closed loop with reality.** The final recommendation is validated against actual
   pit-stop timing and a lap-by-lap cumulative drift trace (0.58 s), rather than only
   against a held-out split.

6. **Delivery as a decision product.** Notebook → serialized artifacts → interactive
   pit-wall dashboard with a live 10,000-race what-if simulator (vectorized GP
   evaluation).

7. **Full-window stochastic strategy optimization.** The original 4-point grid is
   replaced by Differential Evolution over every feasible pit lap, with a dual-engine
   objective, common random numbers, and independent-seed validation. The optimizer
   finds an interior optimum (lap 34) that the grid could not see, improves the
   model-averaged win probability by ~22 percentage points, and is benchmarked against
   Newton–Raphson and golden-section so the algorithm choice itself is evidence-backed.

---

## 13. How Is This Useful? (real-world value)

- **Race strategy:** with live lap data mid-race, this pipeline can re-run the
  simulation every few laps and output "pit now / pit in n laps" with a win
  probability. The full-window optimizer makes that recommendation a true search
  result rather than a four-point grid pick. Direct pit-wall decision support.
- **Undercut/overcut planning:** quantifies the trade-off between track position and
  tire life probabilistically, including safety-car risk.
- **Risk communication:** P10/median/P90 gap bands let strategists distinguish "we
  win 90% of the time" from "coin flip" — the language teams actually use.
- **Transferable methodology:** fuel correction, asymmetric losses, and uncertainty
  propagation apply to any motorsport series (MotoGP, WEC, karting), to vehicle fleet
  maintenance (predict wear of components and schedule replacement optimally), and to
  other sequential decision problems with expensive asymmetric errors (finance,
  medicine, predictive maintenance).
- **Teaching value:** a compact end-to-end case study of the full data-science loop —
  raw data → domain physics → feature/target design → model selection → uncertainty →
  stochastic optimization → decision UI → real-world validation.

---

## 14. Limitations & Honest Caveats

- **Single race (Bahrain 2023)** → conclusions are circuit-specific (high degradation,
  warm track). Generalization to other circuits requires re-fitting; the pipeline is
  reusable, the constants (fuel effect, cliff lap) are not.
- **Compound imbalance:** MEDIUM has only 8 laps in the modeling set, so medium-specific
  conclusions are weak; SOFT/HARD dominate. (The GP's long length scales show the
  temperature/compound features carry limited marginal signal.)
- **Rival modeled by rule**, not learned (pit lap 45 fixed) — a game-theoretic
  extension would model the rival's response to our stop.
- **MC safety car model** is a coarse hazard (P=0.15 per 10-lap block, synchronised
  for both cars); real SCs interact with pit timing strategically.
- **GP is O(n³)** (25.7 s train, 2.16M queries avoided via vectorization) — scaling to
  a full season would need sparse/inducing-point GPs or deep kernels.
- **The optimization is one-dimensional:** the full-window DE removes the old grid
  limitation, but the decision is still "which lap for a single stop" with a fixed rival
  plan. Two-stop strategies, compound choice, and a reacting rival are higher-dimensional
  extensions (where DE's advantage over exhaustive search becomes larger).
- **The optimizer uses 50,000 races per candidate** to keep the dual-engine objective
  stable. That is cheap here because the state sequences are vectorized, but a live
  in-race version would need a smaller simulation budget or a warm-started surrogate.

## 15. Future Work

- Sequential decision-making: re-plan every lap with updating data (MDP / dynamic
  programming).
- Multi-stop and compound-choice optimization with a higher-dimensional DE.
- Game-theoretic rival model (both teams using our simulator).
- Season-wide multi-circuit model with circuit embeddings.
- Deep GP / Bayesian neural nets for better-calibrated σ.
- Live telemetry streaming into the dashboard for in-race updates.

---

## 16. Likely Professor Questions & Model Answers

**Q: Why not just regress lap time directly?**
A: Absolute lap time mixes fuel load, driver, car, track evolution. We remove fuel
analytically and subtract the stint's own minimum, so the target *is* tire wear. This
makes diverse cars/drivers comparable and massively improves signal-to-noise.

**Q: Why is a custom loss legal/valid for XGBoost?**
A: Gradient boosting only needs the first two derivatives of any twice-differentiable
loss. Our piecewise-quadratic asymmetric loss has valid gradient/Hessian (piecewise
linear/constant), so we supply them via the custom-objective interface. Softplus is a
smooth alternative; ours is the hinge-style asymmetric quadratic analogue.

**Q: You report RMSE ~0.65 s — isn't that poor?**
A: Lap-to-lap noise in F1 is itself ~0.3–0.7 s (traffic, crosswinds, driver
variation). R²≈0.5 on unseen *stints* (harder than unseen laps) with 0.53 s MAE is
competitive. More importantly, the *decision-relevant* metrics (SIS, drift) show the
errors don't accumulate into wrong strategies: 0.58 s drift over a 20-lap stint.

**Q: What exactly is being optimized?**
A: Six levels: hyperparameters (Optuna/CV), polynomial degree (AIC/BIC/CV), GP kernel
(marginal likelihood), training loss (asymmetric cost-sensitive objective), the baseline
race decision (Monte Carlo over a 4-point grid), and the final race decision (full-window
Differential Evolution over 31–56 with a dual-engine objective).

**Q: Why Differential Evolution? Why not Newton–Raphson or golden-section?**
A: The objective is a black-box Monte Carlo win probability over an integer pit lap. It is
not differentiable (Newton is out), not guaranteed unimodal (golden-section can be trapped
by the GP engine's second local peak), and is noisy (DE is evaluated with common random
numbers and validated on fresh seeds). DE is derivative-free, population-based, handles
integer variables and bounds, and is robust to non-convexity. We still implemented Newton
and golden-section as benchmarks: Newton from lap 42 lands on lap 55 with 29.6% ensemble
win and roots ranging 5–148; golden-section finds lap 34 only when given the full bracket
and returns the old grid's lap 42 when seeded with the old window. DE finds lap 34 and is
stable across all three seeds.

**Q: If you also plotted the full 31–56 landscape, isn't that just a 26-point grid?**
A: The landscape is a post-hoc validation plot, not the optimizer. DE itself was run on a
fresh cache and found the optimum in 19 unique evaluations without assuming unimodality or
differentiability. For one dimension an exhaustive scan is cheap; DE's value is that the
same algorithm extends to higher-dimensional strategy spaces (two stops, compound choice,
rival response) where exhaustive search explodes.

**Q: Why do the two simulation engines disagree so much?**
A: They disagree about *confidence*, not direction. XGB forces one small noise σ on all
tires; the GP knows σ grows with tire age. The GP therefore produces honest, wider
uncertainty bands and probabilities nearer 50%. Reporting both exposes model risk
instead of hiding it.

**Q: How do you know the model didn't cheat?**
A: GroupShuffleSplit by stint: entire stints (not random laps) are held out, so no
lap of a test stint was ever seen in training. Fuel correction and filtering use no
future information from the test set.

**Q: What is the single biggest takeaway?**
A: In strategy analytics, the model that minimizes RMSE is not necessarily the model
that wins races. Objective design (asymmetric loss, strategy-aware metrics) and honest
uncertainty (GP σ) change the answer more than squeezing the third decimal of accuracy.

---

## 17. Repository Map

```
F1 Dataset/
├── get_data.py                          # downloads Bahrain 2023 via FastF1 → Excel
├── optimizer.py                         # full-window Differential Evolution optimizer + benchmarks
├── f1-tire-degradation-performance-decay-analysis.ipynb   # full pipeline (source of truth)
├── app.py                               # Streamlit pit-wall dashboard (read-only)
├── F1_Bahrain_2023_Dataset.xlsx         # raw exported lap table
├── cache/                               # FastF1 HTTP cache (raw sessions)
├── artifacts/                           # notebook exports consumed by app.py
│   ├── config.json                      # constants, MC config, best params, GP kernel, OPTIMIZATION
│   ├── linear_model.joblib, poly_model.joblib, poly_transformer.joblib
│   ├── xgb_model.joblib, rf_model.joblib, gp_model.joblib, gp_scaler.joblib
│   ├── laps_df.parquet                  # cleaned modeling dataset (564 laps)
│   ├── metrics_table.csv                # 5-model × 6-metric comparison
│   ├── poly_model_selection.csv         # AIC/BIC/CV per polynomial degree
│   ├── sim_xgb.parquet, sim_gp.parquet  # 40,000 MC races per engine
│   ├── win_tables.csv, win_probability_comparison.csv
│   ├── lowess_curves.csv, temp_heatmap.csv, lap_conditions.parquet
│   ├── per_stint_trace.csv              # PER stint 3: actual vs all 5 models
│   ├── real_pit_stops.csv               # real pit laps for validation
│   ├── optimization_landscape.csv       # win% and gap stats for laps 31–56
│   ├── optimization_comparison.csv      # grid vs golden vs Newton vs DE
│   ├── optimization_convergence.csv     # DE best objective per generation
│   ├── optimization_de_seeds.csv        # DE stability across seeds
│   ├── optimization_validation.csv      # independent-seed validation summary
│   ├── optimization_validation_seeds.csv# per-seed validation rows
│   └── optimization_newton_roots.csv    # Newton start-dependence evidence
└── .streamlit/config.toml               # dark F1-style theme
```

## 18. Reproduce It

```bash
python get_data.py                        # fetch + cache raw data (or rely on cache/)
jupyter nbconvert --to notebook --execute --inplace \
    f1-tire-degradation-performance-decay-analysis.ipynb   # retrains + exports base artifacts/
python optimizer.py                       # full-window DE optimization + validation artifacts
streamlit run app.py                      # opens the dashboard
```

The notebook's final cells call `optimizer.run_full_optimization(...)` as well, so a full
notebook execution also produces the optimization artifacts. `python optimizer.py` is the
standalone path when you do not want to rerun the 100-trial Optuna and model-training
sections.

Key run configuration (from `artifacts/config.json`): fuel K = 0.055 s/kg, race 57 laps,
current lap 30, pit-loss 22 s, SC hazard 15% per 10-lap block, 10,000 sims/strategy for
the baseline grid at laps 42/45/48/51, rival 2.1 s ahead on 18-lap-old mediums. The final
optimizer uses 50,000 common-random-number races per candidate over laps 31–56, three DE
seeds (42, 7, 2026), and five independent validation seeds × 20,000 races.

---

## 19. One-Paragraph Summary for the Professor

> We built an end-to-end pit-strategy decision system on 2023 Bahrain GP data. Raw
> FastF1 laps are cleaned (safety car, pit, dirty-air, outlier removal) and
> physics-corrected for fuel burn (~0.106 s/lap), then converted into a pure
> tire-degradation target by subtracting each stint's best fuel-corrected lap. Five
> models — linear, degree-2 polynomial (chosen by AIC/BIC/CV), an asymmetric-loss
> XGBoost tuned by 100-trial Optuna, a Random Forest (30-trial Optuna), and a Gaussian
> Process with Matérn 5/2 + WhiteKernel — predict degradation on a stint-grouped
> leakage-safe split, evaluated not only by RMSE but by strategy-aware metrics
> (tire-age-weighted RMSE and the Strategy Impact Score of optimistic errors). The
> predictions drive a Monte Carlo race simulator including pit loss, a rival car, and
> safety-car risk. The baseline four-point grid picks lap 42 (99.7% XGB win, 57% GP win),
> but the final layer replaces that grid with a **full-window Differential Evolution**
> optimizer over laps 31–56. The optimizer maximizes the average win probability across
> both engines using common random numbers, and finds **lap 34** (99.99% XGB, 99.98% GP):
> +22 percentage points of model-averaged win probability and +5.1 s / +18.7 s of median
> margin, stable across three DE seeds and five independent validation seeds. Newton's
> method and golden-section are implemented as benchmarks and fail or become
> bracket-dependent, which is why DE is the defensible choice. The result is validated
> lap-by-lap (0.58 s drift over a 20-lap stint) against real pit-stop timing, and shipped
> as an interactive Streamlit pit-wall dashboard with a dedicated Optimizer tab. The
> novelty is the combination of physics-informed target design, a cost-sensitive
> asymmetric training objective, native uncertainty propagation from a Gaussian Process
> into the race simulation, strategy-aware evaluation, and full-window stochastic
> strategy optimization — turning a regression exercise into an honest, probabilistic
> race decision tool.

---

## 20. What Was Added in the Final Version — Full-Window Pit-Lap Optimizer

This section is the short "what changed" companion to the detailed methodology in
Section 8.6. It is written for the final review/viva: what the new component is, why it
is a real optimization algorithm, what it found, and why the result is better.

### 20.1 The gap it closes

The original project ended with a **four-point strategy grid**:

```text
p ∈ {42, 45, 48, 51}
```

That is a hand-designed sample, not an optimizer. It evaluates only **4 of the 26
feasible pit laps** (31–56), so it cannot see the true optimum. The final version adds a
genuine full-window optimization layer instead of extrapolating from four points.

### 20.2 The new component

`optimizer.py` implements the final strategy layer:

```text
maximize    J(p) = 1/2 [ P_XGB(win | p) + P_GP(win | p) ]
                   + 1e-6 * mean_median_gap(p)
subject to  p ∈ {31, 32, ..., 56}      (integer pit lap, current lap 30)
            one-stop strategy, rival pits lap 45, 22 s pit loss, 15% SC hazard
```

The tiny second term only breaks ties inside the win-probability plateau. The main
objective is deliberately **model-robust**: it averages the fixed-noise XGBoost engine and
the uncertainty-aware GP engine instead of optimizing to one model's confidence
assumptions.

Every candidate is evaluated with **common random numbers (CRN)** — the same safety-car
and lap-noise draws — so the comparison reflects the pit lap, not a lucky seed. Final
validation uses five fresh, independent seeds.

### 20.3 Why Differential Evolution

The chosen algorithm is **Differential Evolution** from
`scipy.optimize.differential_evolution`, with `integrality=True` over `[31, 56]`, three
seeds (42, 7, 2026), and a callback for convergence.

The objective is a black-box, integer, non-smooth, noisy Monte Carlo estimate, and under
the GP engine the landscape has a second local maximum near lap 54. That rules out the
simpler alternatives:

| Method | What it requires | What happened here | Verdict |
|---|---|---|---|
| Newton–Raphson | Differentiable objective, local derivative | Started from the old grid's lap 42, it landed on **lap 55** with **29.56%** ensemble win; roots across starts ranged from **5 to 148** | Rejected |
| Golden-section | Continuous, unimodal objective | Found lap 34 only when given the full bracket; seeded with the old grid window it returned **lap 42** | Not robust |
| **Differential Evolution** | Only bounds + a scalar objective | Found **lap 34** in **19 unique evaluations**, **3 generations**, ~**1.2 s**; all three seeds returned lap 34 | **Selected** |

The benchmark table in the dashboard reads directly from
`artifacts/optimization_comparison.csv`; the Newton root evidence is in
`artifacts/optimization_newton_roots.csv`.

### 20.4 Results — before vs after

| Metric | Old grid best (lap 42) | DE optimum (lap 34) | Change |
|---|---|---|---|
| XGBoost win probability | 99.64% | **99.99%** | +0.35 pp |
| Gaussian Process win probability | 56.35% | **99.98%** | +43.63 pp |
| Dual-engine win probability | 78.00% | **99.99%** | **+21.99 pp** |
| XGBoost median final gap | +12.91 s | **+17.97 s** | **+5.06 s** |
| GP median final gap | +1.06 s | **+19.71 s** | **+18.66 s** |

**Independent validation** (five fresh seeds × 20,000 races per scenario):

| Scenario | Pit lap | XGB win | GP win | Dual-engine win | XGB median gap | GP median gap |
|---|---|---|---|---|---|---|
| Old grid best | 42 | 99.59% | 56.50% | 78.05% | +12.94 s | +1.07 s |
| **DE optimum** | **34** | **99.99%** | **99.97%** | **99.98%** | **+18.00 s** | **+19.73 s** |

The gain is not a boundary artifact: lap 34 is interior to 31–56. It is not a
single-engine artifact: XGB-only prefers lap 36, GP-only prefers lap 33, and the
dual-engine objective prefers lap 34 — all in the same early-stop neighborhood. It is not
Monte Carlo noise: the gain survives the independent seeds.

### 20.5 Why this is better

1. **It replaces hand-design with search.** The old grid was a human choice of four laps;
   DE searches every feasible integer lap and finds an interior optimum.
2. **It optimizes the actual dual-engine decision.** The objective is the average win
   probability across both simulation engines, which matches the project's central
   uncertainty story.
3. **It is validated, not just optimized.** The old grid's best point is dominated on both
   engines in the optimization run and in five independent validation runs.
4. **It is benchmarked against the alternatives.** Newton and golden-section are not just
   dismissed theoretically; they are implemented and their failure modes are recorded.
5. **It is cheap enough to re-run.** 19 unique evaluations, 3 generations, ~1.2 s with
   50,000 races per candidate. A pit wall can re-run this as the race evolves.
6. **It extends naturally.** A one-dimensional exhaustive scan is also cheap; DE's value
   is that the same algorithm handles higher-dimensional decisions (two stops, compound
   choice, rival response) where exhaustive search explodes.

### 20.6 What changed in the repository

- `optimizer.py` — the Differential Evolution optimizer, Newton/golden benchmarks,
  common-random-number simulator, independent validation, artifact export, and CLI.
- `f1-tire-degradation-performance-decay-analysis.ipynb` — new final section with the
  algorithm explanation, benchmark table, convergence plot, landscape plot, and
  independent validation.
- `app.py` — new **Optimizer** tab with the algorithm rationale, benchmark, convergence,
  landscape proof, validation table, and Newton evidence; the sidebar/overview now
  recommend the optimized lap, and the what-if simulator default/ranges were aligned.
- `README.md` — Section 8.6 documents the full methodology; this Section 20 is the
  before/after summary for the final review.
- `artifacts/optimization_*.csv` and `artifacts/config.json` — generated proof artifacts:
  landscape, comparison, convergence, DE seeds, independent validation, Newton roots, and
  the `OPTIMIZATION` config block.

The old four-point grid is intentionally retained in the Monte Carlo tab and README
Section 9 as the **baseline**; the new optimizer is the final decision layer.

### 20.7 Reproduce

```bash
python optimizer.py          # regenerate optimization artifacts + update config.json
streamlit run app.py         # open the dashboard → Optimizer tab
```

The notebook's final cells call the same function, so a full notebook execution also
produces these artifacts.
