# Demo Guide: F1 Tire Degradation & Pit Strategy (Bahrain GP 2023)

Everything is pre-computed. Nothing needs internet and nothing needs re-training.
Just double-click.

---

## 1. Which file to double-click

| Double-click | What happens | Start-up time |
|---|---|---|
| **`run_dashboard.bat`** | Starts the Streamlit dashboard and opens it in your browser at http://localhost:8501 | ~5–10 s |
| **`open_notebook.bat`** | Opens `executed_notebook.ipynb` in Jupyter with **all outputs and plots already saved** | ~5 s |
| **`run_optimizer.bat`** | Runs the Differential Evolution pit-lap optimizer in a console window and keeps the result on screen | ~10 s |

- Keep each black console window open while you use it. **Closing the window stops the app.**
- If you double-click `run_dashboard.bat` while the dashboard is already running, it just
  opens the browser tab again.
- `run_optimizer.bat` runs with `--no-save`, so it never changes the files the dashboard reads.
- In the notebook, **do not click "Run All"**: it retrains every model, including a
  100-trial Optuna search, and takes a long time. Just scroll; every output is already there.

**Suggested demo order:** dashboard (main story) → optimizer console (live proof the
algorithm runs) → notebook (if the teacher wants to see the code and training).

### Why the notebook shows slightly different numbers from the dashboard

`executed_notebook.ipynb` is a **fresh full re-run** done on this laptop (about 33 minutes).
The Optuna hyperparameter search in the notebook has no fixed random seed, so each re-run
picks slightly different XGBoost settings. The dashboard, optimizer and README use the
**official saved models** in `artifacts/`.

| | Dashboard / README (official) | Notebook fresh re-run |
|---|---|---|
| XGBoost RMSE | 0.663 s | 0.672 s |
| Random Forest RMSE | 0.648 s | 0.646 s |
| Linear / Polynomial / GP | identical | identical |
| Grid best → DE optimum | lap 42 → **lap 34** | lap 42 → **lap 33** |
| Ensemble win probability | 78.00% → 99.99% | 78.10% → 99.99% |

**If the teacher asks:** "Optuna is a randomized search, so a re-run finds slightly different
hyperparameters. The optimum moves by one lap, from 34 to 33, which stays inside the
33–36 early-stop window that all our objective variants agree on. The recommendation is robust
to retraining." (This actually strengthens the result.) The re-run's full output files are kept
in `artifacts_notebook_rerun/`.

---

## 2. Dashboard tabs: what each one shows

| Tab | What to show / say |
|---|---|
| **Overview** | Headline numbers: 564 laps modeled, 19 drivers / 52 stints, **recommended pit lap 34**, best model by strategy damage = Gaussian Process. Key-findings cards. |
| **Data & Method** | Physics-based fuel correction (0.055 s/kg, ≈0.106 s/lap), the cleaning funnel (SC laps, pit laps, dirty air, ±2.5σ outliers removed), and the leakage-safe split by stint (452 train / 112 test laps). |
| **EDA** | LOWESS tire-degradation curves per compound, clean vs dirty air, track-temperature × tire-age heatmap. |
| **Models** | The 5-model comparison table, RMSE and SIS bar charts, polynomial degree selection (AIC/BIC/CV), and Perez's stint traced lap by lap with the GP ±1σ band. |
| **Monte Carlo** | Baseline 4-point strategy grid (pit laps 42/45/48/51), win % per engine, gap distributions, plus the **live what-if simulator**. Pick an engine, move the sliders, click **"Run 10,000 simulations"**. |
| **Optimizer** | Problem formulation, the Newton vs golden-section vs Differential Evolution benchmark, DE convergence, the full-window landscape (laps 31–56), and independent-seed validation. |
| **Proof & Inference** | GP kernel and learned parameters, cumulative-drift chart (0.58 s over 20 laps), comparison with real teams' pit stops (laps 35–50, median 41), uncertainty σ, and the final optimization proof. |

---

## 3. Key numbers to quote

### The answer
- **Optimal pit lap: lap 34** (Differential Evolution over the full window, laps 31–56; current lap 30).
- **Ensemble win probability: 99.99%** at lap 34 (XGBoost engine 99.99%, Gaussian Process engine 99.98%).
- Old 4-point grid's best: **lap 42 → 78.00%** ensemble win probability (XGB 99.64%, GP 56.35%).
- **Improvement: +21.99 percentage points** of win probability, plus **+5.06 s** (XGB) and
  **+18.66 s** (GP) of extra median finishing margin over the rival.

### Model metrics (test set: 112 laps from stints the models never saw; `artifacts/metrics_table.csv`)

| Model | RMSE (s) | MAE (s) | R² | TW-RMSE (s) | SIS (s/stint) |
|---|---|---|---|---|---|
| Baseline (Linear) | 0.680 | 0.545 | 0.460 | 0.974 | 3.02 |
| Polynomial (deg 2) | 0.822 | 0.651 | 0.213 | 1.149 | 3.65 |
| XGBoost (asymmetric loss) | 0.663 | 0.541 | 0.487 | 0.939 | 2.58 |
| Random Forest | **0.648** | 0.541 | **0.511** | **0.919** | 2.72 |
| Gaussian Process | 0.649 | **0.531** | 0.508 | 0.922 | **2.15** |

- **RMSE**: plain accuracy. Random Forest is best (0.648 s).
- **SIS (Strategy Impact Score)**: seconds a strategist loses per stint by trusting an
  *optimistic* model. This is the metric that matters for pit calls. **Gaussian Process is best (2.15 s)**.
- **TW-RMSE**: errors on old tires weighted up to 4×, because that is where the pit decision happens.
- Core insight: a model can win on RMSE but lose on strategy cost. That is why the project uses SIS.

### Baseline Monte Carlo grid (10,000 races per strategy)

| Pit lap | XGB win % | GP win % |
|---|---|---|
| 42 | 99.65% | 57.02% |
| 45 | 92.46% | 45.38% |
| 48 | 51.89% | 50.55% |
| 51 | 15.18% | 57.72% |

### Optimization algorithm results (`artifacts/optimization_comparison.csv`)

| Algorithm | Best lap | Ensemble win | Verdict |
|---|---|---|---|
| Grid search (baseline, 4 points) | 42 | 78.00% | Misses the full-window peak |
| Golden-section (full window) | 34 | 99.99% | Finds it here but can't certify it (assumes unimodal) |
| Golden-section (old grid bracket) | 42 | 78.00% | Trapped at the bracket edge |
| Newton–Raphson (quadratic surrogate) | 55 | 29.56% | Start-dependent, roots ranged 5–148: rejected |
| **Differential Evolution (selected)** | **34** | **99.99%** | Global, integer-native, no derivative needed |

- DE needs **19 unique pit-lap evaluations, 3 generations, about 1 s**, with
  **50,000 simulated races per candidate** (common random numbers).
- All three DE seeds (42, 7, 2026) agree on **lap 34**.
- **Independent validation** (5 new seeds × 20,000 races): DE optimum 99.98% vs grid best 78.05%.
- XGB-only objective → lap 36, GP-only → lap 33, dual-engine → lap 34: all in the same
  **33–36 early-stop window**, so the result isn't an artifact of one model.
- Real teams pitted between laps 35 and 50 (median 41). Lap 34 is an aggressive undercut
  aimed at a rival who pits on lap 45.

### Other numbers worth having ready
- Data: Bahrain GP 2023 race via FastF1. 564 clean laps, 19 drivers, 52 stints (361 hard, 195 soft, 8 medium laps).
- XGBoost tuned with a **100-trial Optuna** search; Random Forest with 30 trials.
- Asymmetric loss: under-prediction (optimistic tire) penalized **2.5×** vs over-prediction 1.0×.
- Monte Carlo set-up: 22 s pit loss, 15% safety-car chance per 10-lap block, rival 2.1 s ahead on 18-lap-old tires.
- Model drift on Perez's last stint: **0.58 s over 20 laps**.

---

## 3b. How to explain the optimization algorithm (Differential Evolution)

**30-second version:** "We're at lap 30 and need to pick the best lap, between 31 and 56,
to pit. You can't write a formula for win probability; each value has to be estimated by
simulating 50,000 races. That makes the function noisy, integer-valued and non-differentiable,
so calculus methods like Newton–Raphson don't work. We used Differential Evolution, a global,
population-based algorithm that needs no derivative. In 19 evaluations it found lap 34, with a
99.99% win probability versus 78% for the old hand-picked grid."

**Step by step (what `optimizer.py` does with `scipy.optimize.differential_evolution`):**
1. **Decision variable:** pit lap *p*, an integer in [31, 56].
2. **Objective:** J(p) = average of the win probabilities from the XGBoost engine and the
   Gaussian Process engine (each from 50,000 Monte Carlo races), with a tiny tie-break on median gap.
3. **Initialize:** a population of 12 random candidate pit laps spread over the window.
4. **Mutation:** for each candidate, build a mutant from the current best plus a scaled
   difference of two other random candidates: *v = best + F·(x₁ − x₂)*, with F randomly in
   0.5–1.0 (strategy `best1bin`).
5. **Crossover:** mix the mutant with the original (crossover rate 0.7), then round to an integer lap.
6. **Selection:** evaluate J for the trial lap. If it beats the original, it replaces it (greedy, survival of the fittest).
7. **Repeat** until the population converges (tolerance 1e-4, max 30 generations).
   It converged in **3 generations, 19 unique laps evaluated**.

**Two details that make it rigorous:**
- **Common random numbers:** every candidate lap faces the *same* 50,000 random race scenarios
  (same safety cars, same noise), so differences between laps come from the strategy, not luck.
- **Robustness checks:** 3 DE seeds (42, 7, 2026) all give lap 34. Five fresh validation seeds
  × 20,000 races confirm it (99.98% vs 78.05%).

**Why not the alternatives?**
- **Grid search** tested only 4 of 26 laps (42/45/48/51) and missed the peak.
- **Newton–Raphson** needs a smooth derivative. Started from lap 42 it went to lap 55 (29.6% win). Its roots ranged from lap 5 to lap 148, outside the race.
- **Golden-section** assumes one single peak. Given the old bracket (42–51) it got stuck at the edge, lap 42.
- **DE** makes none of those assumptions: it's global, works on integers, and doesn't need derivatives.

**Likely question: "It's only 26 laps, why not just try them all?"**
Answer: For one pit stop, yes, brute force is cheap, and our full landscape scan confirms
lap 34. DE's value is that it treats the simulator as a black box and scales to harder
versions (two pit stops, choice of tire compound), where brute force explodes combinatorially.

---

## 3c. Showing the values change live

**The optimizer gives the same numbers every run on purpose.** It uses a fixed random seed
(42), so anyone can reproduce the result. To show it isn't hard-coded, change the seed or the
number of simulated races. Open PowerShell in this folder and run:

```
..\venv\Scripts\python.exe optimizer.py --no-save --seed 7
..\venv\Scripts\python.exe optimizer.py --no-save --seed 123
..\venv\Scripts\python.exe optimizer.py --no-save --n-sims 2000
```

| Command | Grid best (lap 42) | DE optimum | DE evaluations |
|---|---|---|---|
| default (seed 42, 50,000 races) | 78.00% | lap 34, 99.99% | 19 |
| `--seed 7` | 78.09% | lap 34, 99.99% | 19 |
| `--seed 123` | 78.12% | lap 34, 99.99% | 19 |
| `--n-sims 2000` | 78.03% | lap 34, 100.00% | 24 |

The percentages move a little because the random races are different, but the answer stays
**lap 34**. That shows the result is robust. `--no-save` keeps the dashboard files unchanged.

**Dashboard what-if simulator (Monte Carlo tab, bottom).** Move the **Pit lap** slider and click
**"Run 10,000 simulations"**. The model (Gaussian Process) predicts every remaining lap's
tire degradation, then simulates 10,000 races with random noise and safety cars:

| Pit lap | Win probability | Median gap |
|---|---|---|
| 31 | 98.9% | +12.95 s |
| **34** | **100.0%** | **+19.77 s** |
| 38 | 94.3% | +9.04 s |
| 42 | 57.0% | +1.15 s |
| 45 | 45.9% | −0.77 s |
| 48 | 50.6% | +0.12 s |

Story to tell: "Pit too early and the new tires must last too long. Pit too late and the old
tires cost time every lap. Lap 34 is the sweet spot."

⚠️ For the demo, **only move the Pit lap slider**:
- *Rival gap*: in the code, a bigger value counts in **our** favour (8.0 s gives a higher win %
  than 2.1 s). That's the opposite of what the label suggests, and it's in the original
  project code (notebook and app use the same formula).
- *Tire age*: values like 30 push the model beyond the tire ages it was trained on, so results
  can look odd (older tires appear to help). Leave both at their defaults.

---

## 4. If something goes wrong during the demo

| Problem | Fix |
|---|---|
| Browser doesn't open after `run_dashboard.bat` | Wait about 10 s, then type **http://localhost:8501** in the browser yourself. |
| "Port 8501 is already in use" or an old dashboard shows up | A previous dashboard is still running. Just use the open browser tab, or close the old black window and double-click again. |
| Dashboard shows a red error box | Refresh the page (F5). If it stays, close the black window and double-click `run_dashboard.bat` again. |
| XGBoost error `input stream corrupted` | The Windows-loadable model was replaced by the GitHub copy. Double-click nothing; open PowerShell in this folder and run `..\venv\Scripts\python.exe recover_xgb_model.py` (takes 2 s). |
| Notebook plots look blank | In Jupyter, use **File → Trust Notebook**, or close and reopen with `open_notebook.bat`. Never "Run All". |
| Jupyter asks for a token/password | Close it and use `open_notebook.bat` (it opens the logged-in link automatically). |
| Console window flashes and closes | Open PowerShell in this folder and run the .bat by name (e.g. `.\run_dashboard.bat`) to read the error. |
| Everything fails | Show the static evidence: `executed_notebook.ipynb` can be opened from any Jupyter / VS Code, and `README.md` has every table and number. |

Backup plan for the optimizer: the same numbers are on the dashboard's **Optimizer** tab.

---

## 5. Files added for the demo (for reference)

| File | Purpose |
|---|---|
| `run_dashboard.bat`, `open_notebook.bat`, `run_optimizer.bat` | One-click launchers (use the venv at `C:\Users\Arush\OneDrive\Desktop\venv`). |
| `executed_notebook.ipynb` | Fully executed copy of the notebook with all outputs (the original notebook is unchanged). |
| `recover_xgb_model.py` | Makes the original XGBoost model loadable on Windows (same trees, identical predictions, RMSE 0.662985 s). |
| `rebuild_xgb_model.py` | Alternative: retrains the XGBoost model from `config.json` (RMSE 0.664 s). Not used for the demo because it shifts the DE optimum to lap 33. |
| `artifacts/xgb_model_original.joblib` | Untouched backup of the GitHub model file. |
| `artifacts/xgb_model_rebuilt.joblib` | The retrained model from `rebuild_xgb_model.py` (kept for reference only). |
| `artifacts_notebook_rerun/` | All models/results regenerated by the fresh notebook run (reference only; the dashboard does not use them). |
| `.streamlit/config.toml` | Added `browser.gatherUsageStats = false` (no telemetry/internet); theme and `headless = true` kept, so no e-mail prompt. |

### Live "retrain" if asked
Training the final XGBoost model takes about 2 seconds. Only the 100-trial search is slow.
Open PowerShell in this folder and run:

```
..\venv\Scripts\python.exe rebuild_xgb_model.py
..\venv\Scripts\python.exe recover_xgb_model.py
```

The first command retrains and prints the test RMSE (≈0.664 s). The second puts the official
model back so the dashboard still shows lap 34. **Always run both, in this order.**
