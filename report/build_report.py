"""
Build the final project report (Word .docx) for the F1 tire-strategy project.

The report reuses the styles, fonts and page border of the course's sample report
(HCI_DA3.docx) as a template and writes new content, figures and tables into it.
Figures and prediction tables come from report/make_figures.py; every number is
read from artifacts/ so the report always matches the dashboard.

Run from the project folder:
    python report/make_figures.py
    python report/build_report.py [path/to/template.docx]
"""

import json
import re
import shutil
import struct
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
FIG = ROOT / "report" / "figures"
TAB = ROOT / "report" / "tables"
TEMPLATE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads" / "HCI_DA3.docx"
OUT = ROOT / "F1_Tire_Strategy_Final_Report.docx"
LOGO = ROOT / "report" / "figures" / "vit_logo.png"

GITHUB = "https://github.com/Sudharshan348/F1-tire-analysis"
TEXT_WIDTH = 9026  # A4 width minus 1" margins, in DXA

cfg = json.loads((ART / "config.json").read_text())
opt = cfg["OPTIMIZATION"]
metrics = pd.read_csv(ART / "metrics_table.csv")
win = pd.read_csv(ART / "win_probability_comparison.csv")
comparison = pd.read_csv(ART / "optimization_comparison.csv")
validation = pd.read_csv(ART / "optimization_validation.csv").set_index("Scenario")
seeds = pd.read_csv(ART / "optimization_de_seeds.csv")
poly_sel = pd.read_csv(ART / "poly_model_selection.csv")
laps = pd.read_parquet(ART / "laps_df.parquet")
sample_preds = json.loads((TAB / "sample_predictions.json").read_text())
scenario_preds = json.loads((TAB / "scenario_predictions.json").read_text())


# =============================================================== XML helpers
class Doc:
    def __init__(self):
        self.body = []
        self.images = []      # (rid, target name, source path)
        self.links = []       # (rid, url)
        self.num_lists = 0    # numbered-list instances (each restarts at 1)
        self.fig_no = 0
        self.tab_no = 0
        self._rid = 100
        self._docpr = 1

    def rid(self):
        self._rid += 1
        return f"rId{self._rid}"


D = Doc()
FONT = "Times New Roman"
INLINE = re.compile(r"(\*\*.+?\*\*|\*[^*\s][^*]*?\*|_\{.+?\}|\^\{.+?\}|\[\[.+?\]\])")


def rpr(size=24, bold=False, italic=False, color=None, vert=None, highlight=None, font=FONT):
    x = f'<w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:cs="{font}" w:eastAsia="{font}"/>'
    if bold:
        x += "<w:b/><w:bCs/>"
    if italic:
        x += "<w:i/><w:iCs/>"
    if color:
        x += f'<w:color w:val="{color}"/>'
    x += f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
    if highlight:
        x += f'<w:highlight w:val="{highlight}"/>'
    if vert:
        x += f'<w:vertAlign w:val="{vert}"/>'
    return f"<w:rPr>{x}</w:rPr>"


def run(text, **kw):
    return f'<w:r>{rpr(**kw)}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def runs(text, size=24, bold=False, italic=False, color=None):
    """Inline markup: **bold**, *italic*, _{sub}, ^{sup}, [[placeholder]] (yellow)."""
    out = []
    for tok in INLINE.split(text):
        if not tok:
            continue
        if tok.startswith("**"):
            out.append(run(tok[2:-2], size=size, bold=True, italic=italic, color=color))
        elif tok.startswith("_{"):
            out.append(run(tok[2:-1], size=size, bold=bold, italic=italic, color=color, vert="subscript"))
        elif tok.startswith("^{"):
            out.append(run(tok[2:-1], size=size, bold=bold, italic=italic, color=color, vert="superscript"))
        elif tok.startswith("[["):
            out.append(run(tok[2:-2], size=size, bold=bold, italic=italic, color=color, highlight="yellow"))
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            out.append(run(tok[1:-1], size=size, bold=bold, italic=True, color=color))
        else:
            out.append(run(tok, size=size, bold=bold, italic=italic, color=color))
    return "".join(out)


def ppr(align="both", before=0, after=120, line=276, ind_left=None, hanging=None, first=None,
        style=None, keep_next=False, num=None, shade=None, page_break_before=False):
    x = ""
    if style:
        x += f'<w:pStyle w:val="{style}"/>'
    if keep_next:
        x += "<w:keepNext/>"
    if page_break_before:
        x += "<w:pageBreakBefore/>"
    if num:
        x += f'<w:numPr><w:ilvl w:val="0"/><w:numId w:val="{num}"/></w:numPr>'
    if shade:
        x += f'<w:shd w:val="clear" w:color="auto" w:fill="{shade}"/>'
    x += f'<w:spacing w:before="{before}" w:after="{after}" w:line="{line}" w:lineRule="auto"/>'
    ind = ""
    if ind_left is not None:
        ind += f' w:left="{ind_left}"'
    if hanging is not None:
        ind += f' w:hanging="{hanging}"'
    if first is not None:
        ind += f' w:firstLine="{first}"'
    if ind:
        x += f"<w:ind{ind}/>"
    x += f'<w:jc w:val="{align}"/>'
    return f"<w:pPr>{x}</w:pPr>"


def P(text, **kw):
    size = kw.pop("size", 24)
    bold = kw.pop("bold", False)
    italic = kw.pop("italic", False)
    D.body.append(f"<w:p>{ppr(**kw)}{runs(text, size=size, bold=bold, italic=italic)}</w:p>")


def H1(text, page_break=True):
    D.body.append(f'<w:p>{ppr(align="left", before=240, after=160, style="Heading1", keep_next=True, page_break_before=page_break)}'
                  f'{run(text, size=32, bold=True, color="000000")}</w:p>')


def H2(text):
    D.body.append(f'<w:p>{ppr(align="left", before=240, after=100, style="Heading2", keep_next=True)}'
                  f'{run(text, size=28, bold=True, color="000000")}</w:p>')


def H3(text):
    D.body.append(f'<w:p>{ppr(align="left", before=160, after=80, keep_next=True)}{run(text, size=24, bold=True)}</w:p>')


def bullets(items):
    for it in items:
        D.body.append(f'<w:p>{ppr(after=60, num=90, ind_left=720, hanging=360)}{runs(it)}</w:p>')
    D.body[-1] = D.body[-1].replace('w:after="60"', 'w:after="140"', 1)


def numbered(items):
    D.num_lists += 1
    num_id = 200 + D.num_lists
    for it in items:
        D.body.append(f'<w:p>{ppr(after=60, num=num_id, ind_left=720, hanging=360)}{runs(it)}</w:p>')
    D.body[-1] = D.body[-1].replace('w:after="60"', 'w:after="140"', 1)


def EQ(text):
    D.body.append(f'<w:p>{ppr(align="center", before=60, after=140)}{runs(text, italic=True)}</w:p>')


def CODE(lines):
    for i, ln in enumerate(lines):
        after = 120 if i == len(lines) - 1 else 0
        D.body.append(f'<w:p>{ppr(align="left", after=after, line=240, shade="F2F2F2", ind_left=144, keep_next=i < len(lines) - 1)}'
                      f'{run(ln if ln else " ", size=18, font="Consolas")}</w:p>')


def page_break():
    D.body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')


def png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    return struct.unpack(">II", head[16:24])


def image_xml(path, width_in):
    w, h = png_size(path)
    cx = int(width_in * 914400)
    cy = int(cx * h / w)
    rid = D.rid()
    name = f"image{len(D.images) + 1}.png"
    D.images.append((rid, name, path))
    D._docpr += 1
    n = D._docpr
    return (f'<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
            f'<wp:effectExtent l="0" t="0" r="0" b="0"/><wp:docPr id="{n}" name="Picture {n}"/>'
            '<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
            '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:pic><pic:nvPicPr><pic:cNvPr id="{n}" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic>'
            "</wp:inline></w:drawing></w:r>")


def FIGURE(fname, caption, width_in=6.2):
    D.fig_no += 1
    D.body.append(f'<w:p>{ppr(align="center", before=120, after=60, keep_next=True)}{image_xml(FIG / fname, width_in)}</w:p>')
    D.body.append(f'<w:p>{ppr(align="center", after=200)}{runs(f"**Figure {D.fig_no}.** {caption}", size=21)}</w:p>')
    return D.fig_no


def TABLE(caption, headers, rows, widths, size=19, align_first_left=True):
    D.tab_no += 1
    assert abs(sum(widths) - TEXT_WIDTH) < 5, (caption, sum(widths))
    D.body.append(f'<w:p>{ppr(align="left", before=160, after=80, keep_next=True)}'
                  f'{runs(f"**Table {D.tab_no}.** {caption}", size=21)}</w:p>')
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)

    def cell(text, w, header=False, left=False):
        shade = '<w:shd w:val="clear" w:color="auto" w:fill="D9E2F3"/>' if header else ""
        jc = "left" if left else "center"
        return (f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{shade}<w:vAlign w:val="center"/></w:tcPr>'
                f'<w:p>{ppr(align=jc, after=0, line=240)}{runs(str(text), size=size, bold=header)}</w:p></w:tc>')

    trs = ['<w:tr><w:trPr><w:tblHeader/></w:trPr>' + "".join(cell(h, w, True) for h, w in zip(headers, widths)) + "</w:tr>"]
    for r in rows:
        trs.append("<w:tr><w:trPr><w:cantSplit/></w:trPr>" + "".join(
            cell(v, w, False, left=(i == 0 and align_first_left) or (isinstance(v, str) and len(v) > 28))
            for i, (v, w) in enumerate(zip(r, widths))) + "</w:tr>")
    D.body.append(
        f'<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="{TEXT_WIDTH}" w:type="dxa"/>'
        '<w:jc w:val="center"/><w:tblLayout w:type="fixed"/>'
        '<w:tblCellMar><w:top w:w="40" w:type="dxa"/><w:left w:w="80" w:type="dxa"/>'
        '<w:bottom w:w="40" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tblCellMar></w:tblPr>'
        f"<w:tblGrid>{grid}</w:tblGrid>{''.join(trs)}</w:tbl>")
    D.body.append(f'<w:p>{ppr(after=120)}</w:p>')
    return D.tab_no


def LINK(prefix, url, suffix=""):
    rid = D.rid()
    D.links.append((rid, url))
    link = (f'<w:hyperlink r:id="{rid}"><w:r>{rpr(color="0563C1")[:-8]}<w:u w:val="single"/></w:rPr>'
            f'<w:t xml:space="preserve">{escape(url)}</w:t></w:r></w:hyperlink>')
    D.body.append(f"<w:p>{ppr(align='left')}{runs(prefix)}{link}{runs(suffix)}</w:p>")


def pct(x, d=2):
    return f"{x * 100:.{d}f}%"


# =============================================================== content
m = metrics.set_index("Model")
KERNEL = cfg["gp_kernel"].replace("**2", "²").replace(" * ", " × ")
best = opt["best_pit_lap"]

# ---------------------------------------------------------------- cover page
for _ in range(1):
    D.body.append(f'<w:p>{ppr(align="center", after=0)}</w:p>')
D.body.append(f'<w:p>{ppr(align="center", after=240)}{image_xml(LOGO, 5.2)}</w:p>')
cover = [
    ("PREDICTIVE ANALYTICS", True, 36, 360),
    ("Project Final Review", True, 36, 0),
    ("Final Report", True, 36, 360),
    ("PROJECT TITLE", True, 36, 60),
    ("Data-Driven Formula 1 Pit-Stop Strategy: Tyre Degradation Modelling, Monte Carlo Race "
     "Simulation and Differential Evolution Optimization (Bahrain Grand Prix 2023)", False, 32, 360),
    ("COURSE CODE", True, 36, 60),
    ("[[Course Code]]", False, 32, 240),
    ("FACULTY", True, 36, 60),
    ("[[Faculty Name]]", False, 32, 240),
    ("TEAM MEMBERS", True, 36, 60),
    ("B Sudharshan – [[Reg. No.]]", False, 32, 0),
    ("Kunwar Arush Srivastava – 23BCE5024", False, 32, 0),
    ("Dharanitharan D – [[Reg. No.]]", False, 32, 0),
]
for text, bold, size, after in cover:
    D.body.append(f'<w:p>{ppr(align="center", after=after, line=240)}{runs(text, size=size, bold=bold)}</w:p>')

# ---------------------------------------------------------------- 1. abstract
H1("1. Abstract")
P("In Formula 1, the lap on which a driver stops to change tyres often decides the race. Pitting too early "
  "means the new tyres must survive too many laps; pitting too late means losing time on worn tyres every lap. "
  "Teams make this call under uncertainty about how fast tyres degrade, about the rival's plan and about safety "
  "cars. This project builds a complete, data-driven decision system for that call using real timing data from "
  "the 2023 Bahrain Grand Prix, obtained through the FastF1 library.")
P(f"Raw laps are cleaned (safety-car, pit, dirty-air and outlier laps removed, leaving {cfg['n_laps_modeled']} laps "
  f"from {cfg['n_drivers']} drivers and {cfg['n_stints']} stints) and corrected for fuel burn with a physics-based "
  "model (0.055 s per kg), so the target variable isolates pure tyre degradation. Five regression models are "
  "compared on unseen stints: Linear Regression, Polynomial Regression, XGBoost with a custom asymmetric "
  "(cost-sensitive) loss, Random Forest and a Gaussian Process. Besides RMSE, two strategy-aware metrics are "
  "used: a tyre-age-weighted RMSE and a Strategy Impact Score (SIS) that measures the race time lost by trusting "
  f"an optimistic model. Random Forest gives the lowest RMSE ({m.loc['Random Forest', 'RMSE (s)']:.3f} s), while "
  f"the Gaussian Process gives the lowest SIS ({m.loc['Gaussian Process', 'SIS (s)']:.2f} s per stint).")
P("The models drive a Monte Carlo race simulator with two engines: XGBoost with fixed noise, and a Gaussian "
  "Process whose uncertainty grows with tyre age. A baseline grid of four pit laps (42, 45, 48, 51) picked lap 42 "
  f"with a {pct(opt['grid_best_ensemble_win_pct'])} average win probability. Replacing the grid with a "
  "Differential Evolution optimizer over every feasible lap (31–56), using common random numbers and 50,000 "
  f"simulated races per candidate, finds **lap {best}** with a **{pct(opt['ensemble_win_pct'])}** win "
  f"probability, a gain of **+{opt['improvement_pp']:.2f} percentage points**, in only {opt['n_evaluations']} "
  "evaluations. Grid search, golden-section search and Newton–Raphson were benchmarked and fail for clear reasons. "
  "The result is validated on five independent seeds and delivered through an interactive Streamlit dashboard.")
P("**Keywords:** Formula 1, tyre degradation, race strategy, XGBoost, Gaussian Process, cost-sensitive learning, "
  "Monte Carlo simulation, Differential Evolution, stochastic optimization, predictive analytics.")

# ---------------------------------------------------------------- 2. introduction
H1("2. Introduction")
H2("2.1 Background")
P("A modern Formula 1 race lasts about 57 laps, and every driver must use at least two different dry tyre "
  "compounds, which forces at least one pit stop. A pit stop costs roughly 20–25 seconds, so its timing is one of "
  "the few levers a team controls during the race. Tyre performance is not constant: as a tyre ages, grip drops "
  "and lap times rise, sometimes gently and sometimes with a sudden \"cliff\". At the same time the car gets "
  "lighter as fuel burns, which makes it faster and hides the tyre effect in raw lap times.")
P("Strategy engineers combine tyre models, race simulation and experience to decide when to pit. Published "
  "research has shown the value of race simulation [1]–[3], machine learning for strategy decisions [4], [5] "
  "and formal optimization of pit stops [6], [7]. However, a student-accessible, end-to-end pipeline that goes "
  "from public timing data to a validated, optimized pit lap, while handling model uncertainty honestly, is still "
  "rare. This project builds such a pipeline.")
H2("2.2 Problem Statement")
P("At lap 30 of the 2023 Bahrain Grand Prix, our car is on 15-lap-old medium tyres and the rival is 2.1 s ahead "
  "on 18-lap-old mediums, planning to pit on lap 45. **On which lap should we pit for hard tyres to maximize the "
  "probability of finishing ahead of the rival?** Answering this requires (a) predicting how much lap time is "
  "lost as tyres age, (b) turning those predictions into race outcomes under uncertainty, and (c) searching for "
  "the best pit lap.")
H2("2.3 Scope")
bullets([
    "**Data:** one race, the 2023 Bahrain Grand Prix (race session), loaded with FastF1 [23] from the local cache.",
    "**Prediction:** lap-time loss due to tyre age (*LapTimeDelta*) for soft, medium and hard compounds.",
    "**Decision:** a single pit stop between laps 31 and 56 against one rival with a fixed strategy, including "
    "pit-loss time and random safety cars.",
    "**Delivery:** a Jupyter notebook (training), a standalone optimizer script and a Streamlit dashboard "
    "(read-only demo with a live what-if simulator).",
    "**Out of scope:** multi-stop strategies, choice of compound, a rival that reacts to our decisions, and "
    "other circuits. These are discussed as future work.",
])
H2("2.4 Objectives")
numbered([
    "Build a clean, leakage-free dataset that isolates tyre degradation from fuel effect, traffic and safety cars.",
    "Train and compare five regression models, including a cost-sensitive XGBoost and an uncertainty-aware "
    "Gaussian Process.",
    "Evaluate models with both standard metrics (RMSE, MAE, R²) and strategy-aware metrics (TW-RMSE, SIS).",
    "Develop a Monte Carlo race simulator that propagates model uncertainty and safety-car risk into a win "
    "probability for any pit lap.",
    "Find the optimal pit lap with a proper optimization algorithm (Differential Evolution) and justify the choice "
    "against grid search, golden-section search and Newton–Raphson.",
    "Validate the recommendation with independent seeds and against the real pit stops made in the race.",
    "Deliver a working, interactive dashboard that demonstrates predictions and decisions live.",
])
H2("2.5 Methods / Models (Overview)")
P("The pipeline (Figure 1) has six stages: data collection with FastF1; cleaning and physics-based fuel "
  "correction; feature engineering; training of five models (Linear, Polynomial degree 2, XGBoost with asymmetric "
  "loss, Random Forest, Gaussian Process) with hyperparameters tuned by Optuna; a two-engine Monte Carlo race "
  "simulator; and a Differential Evolution optimizer over the full pit window. A Streamlit dashboard presents the "
  "results. Section 4 explains each stage in detail.")
FIGURE("fig01_pipeline.png", "End-to-end pipeline from raw timing data to the optimized pit-lap decision.")
H2("2.6 Expected Results / Outcome")
bullets([
    "A tyre-degradation model accurate to well under one second per lap on unseen stints.",
    "Evidence that the model with the lowest RMSE is not necessarily the best model for strategy decisions.",
    "A win-probability estimate for every feasible pit lap, with uncertainty bands.",
    "An optimal pit lap that clearly beats the hand-picked baseline grid, found efficiently and reproducibly.",
    "A working dashboard and optimizer that can be demonstrated live.",
])
P(f"All expected outcomes were achieved: test RMSE ranges from {m['RMSE (s)'].min():.3f} s to "
  f"{m['RMSE (s)'].max():.3f} s, the optimizer finds lap {best} ({pct(opt['ensemble_win_pct'])} win probability "
  f"versus {pct(opt['grid_best_ensemble_win_pct'])} for the grid), and the dashboard runs locally in seconds.")
H2("2.7 Conclusion of the Introduction")
P("Pit-stop timing needs both a good predictive model and a sound optimization method. Sections 3–8 cover the "
  "literature, methods, results with prediction outputs, conclusions, references and the code appendix.")

# ---------------------------------------------------------------- 3. literature review
H1("3. Literature Review")
P("Twenty studies were reviewed in three groups: (A) motorsport race simulation and strategy, (B) machine-learning "
  "models used for prediction, and (C) optimization and simulation methods used for the decision. Table 1 at the "
  "end of the section summarizes them and the research gap this project addresses.")
H2("3.1 Race Simulation and Strategy in Motorsport")
P("**Bekker and Lotz (2009) [1]** built one of the first published discrete-event simulations of a Formula 1 "
  "race, modelling on-track events such as car failures, overtaking manoeuvres and pit stops. The model was "
  "developed for a single team to test race strategies and was validated against three races of the 2005 season. "
  "It showed that simulation is a practical tool for comparing strategies, but its inputs were set by hand rather "
  "than learned from data, which motivates our data-driven degradation models.")
P("**Heilmeier, Graf and Lienkamp (2018) [2]** presented a lap-wise race simulation for strategy decisions in "
  "circuit motorsport, which models tyre degradation, fuel-mass effects, pit stops and overtaking. It allows "
  "engineers to compare strategies such as an early versus a late stop. Our simulator follows the same lap-wise "
  "principle, but replaces fixed parametric tyre curves with machine-learning models trained on the race itself.")
P("**Heilmeier et al. (2020) [3]** extended the race simulation with Monte Carlo methods to capture probabilistic "
  "events, arguing that real races often depart from plans because of random events such as accidents and safety "
  "cars. Running many randomized races turns a single deterministic outcome into a distribution. This directly "
  "supports our Monte Carlo approach; we add a second source of randomness, the model's own uncertainty, through "
  "the Gaussian Process engine.")
P("**Heilmeier, Thomaser, Graf and Betz (2020) [4]** proposed a \"Virtual Strategy Engineer\" based on two "
  "artificial neural networks that decide whether a driver should pit and which compound to fit, trained on "
  "Formula 1 timing data from the 2014–2019 seasons. The networks made reasonable decisions and reacted to the "
  "race situation. Such models learn to imitate past team decisions; our work instead optimizes win probability "
  "explicitly, so the recommendation is not limited to what teams did before.")
P("**Liu and Fotouhi (2020) [5]** developed race strategies for Formula E by replacing slow lap-time simulation "
  "with neural-network prediction models and searching strategies with Monte Carlo Tree Search, giving a fast "
  "tool for both pre-race planning and in-race decisions. Their \"learned surrogate + search algorithm\" design is "
  "the same idea as our \"degradation models + Differential Evolution\" design, applied to energy management "
  "instead of tyres.")
P("**Carrasco Heine and Thraves (2023) [6]** formulated pit-stop strategy optimization as a sequential decision "
  "problem solved with dynamic programming, deciding when to stop and which tyres to fit. Dynamic programming "
  "gives exact optimal policies but requires a discretized, fully specified model of the race, which becomes "
  "large quickly. Our black-box approach avoids writing the race as explicit state transitions.")
P("**Aguad and Thraves (2024) [7]** modelled two drivers optimizing their pit stops as a zero-sum feedback "
  "Stackelberg game solved with dynamic programming over hundreds of millions of states. They showed that a "
  "strategic driver facing a rival who ignores competition raises the winning odds by more than 15%, and that "
  "yellow flags tend to help the slower driver. This highlights a limitation of our work: the rival follows a "
  "fixed plan (pit on lap 45) and does not react.")
P("**Thomas et al. (2025) [8]**, a collaboration between Imperial College London and the Mercedes-AMG PETRONAS "
  "F1 team, applied reinforcement learning to Formula 1 race strategy and focused on making the agent's pit-stop "
  "decisions explainable to strategists. It shows that industry is actively exploring learned strategy agents, "
  "and that interpretability matters, which is why our dashboard shows the full win-probability landscape rather "
  "than a single answer.")
P("**Pacejka and Bakker (1992) [9]** introduced the \"Magic Formula\" tyre model, an empirical formula for tyre "
  "forces that is still the industry standard in vehicle dynamics. It requires detailed rig or test data that is "
  "not publicly available for Formula 1 tyres. This is why we model degradation directly from lap times with "
  "data-driven regression instead of physical tyre-force models.")
H2("3.2 Machine-Learning Models for Prediction")
P("**Chen and Guestrin (2016) [10]** introduced XGBoost, a scalable gradient-boosted tree system with a "
  "regularized objective and a second-order (gradient and Hessian) approximation of the loss. Because only the "
  "gradient and Hessian are needed, any twice-differentiable custom loss can be plugged in. We use this to train "
  "XGBoost with an asymmetric loss that penalizes optimistic tyre predictions 2.5 times more than pessimistic ones.")
P("**Breiman (2001) [11]** proposed Random Forests, which average many decision trees trained on bootstrap "
  "samples with random feature subsets. The method is accurate, robust to noise and captures interactions without "
  "manual feature design. In our study Random Forest achieves the lowest RMSE, confirming its strength as a "
  "general-purpose regressor.")
P("**Rasmussen and Williams (2006) [12]** provide the standard reference on Gaussian Process regression, in which "
  "a kernel defines a prior over smooth functions, kernel hyperparameters are learned by maximizing the marginal "
  "likelihood, and every prediction comes with a variance. We use a Matérn 5/2 kernel with one length scale per "
  "feature; the predictive standard deviation feeds the uncertainty-aware simulation engine.")
P("**Elkan (2001) [13]** laid out the foundations of cost-sensitive learning: when different errors have different "
  "costs, a model should be trained and judged by expected cost, not by accuracy alone. In pit strategy, "
  "under-predicting tyre degradation (staying out too long) costs far more than over-predicting it, which is the "
  "basis for our asymmetric XGBoost loss and our Strategy Impact Score.")
P("**Bergstra, Bardenet, Bengio and Kégl (2011) [14]** introduced sequential model-based hyperparameter "
  "optimization, including the Tree-structured Parzen Estimator (TPE), and showed it finds better settings than "
  "random or grid search for the same budget. TPE is the sampler we use to tune XGBoost (100 trials) and Random "
  "Forest (30 trials).")
P("**Akiba et al. (2019) [15]** presented Optuna, a define-by-run hyperparameter optimization framework with "
  "efficient sampling and pruning. We use Optuna with 5-fold cross-validated RMSE as the objective of each trial.")
H2("3.3 Optimization and Simulation Methods for the Decision")
P("**Storn and Price (1997) [16]** introduced Differential Evolution (DE), a population-based global optimizer that "
  "creates new candidates from scaled differences between existing ones, mixes them by crossover and keeps a "
  "candidate only if it improves. DE needs no derivatives and has few control parameters, which makes it suitable "
  "for black-box objectives such as our simulated win probability.")
P("**Das and Suganthan (2011) [17]** surveyed DE variants, parameter settings and applications, including "
  "constrained, discrete and noisy problems, and reported DE's robustness on multimodal functions. This supports "
  "our use of DE on an integer-valued, noisy and possibly multimodal objective (our Gaussian Process engine has a "
  "second local peak near lap 54).")
P("**Kleywegt, Shapiro and Homem-de-Mello (2002) [18]** analysed the Sample Average Approximation (SAA) method for "
  "stochastic discrete optimization: the expected value is replaced by an average over sampled scenarios, and the "
  "solution converges to the true optimum as the sample grows. Our win probability, averaged over 50,000 simulated "
  "races per candidate lap, is exactly such a sample average.")
P("**Glasserman and Yao (1992) [19]** gave guidelines and guarantees for common random numbers (CRN), showing when "
  "evaluating competing alternatives on the same random scenarios reduces the variance of their difference. Our "
  "optimizer evaluates every candidate pit lap on the same safety-car and lap-noise draws, so differences between "
  "laps reflect the strategy rather than luck.")
P("**Cleveland (1979) [20]** proposed LOWESS, a robust locally weighted regression for smoothing scatter plots. We "
  "use LOWESS in exploratory analysis to reveal the tyre-degradation trend of each compound from noisy lap data.")

lit_rows = [
    ["[1] Bekker & Lotz, 2009", "Discrete-event F1 race simulation", "Simulation is practical for strategy tests", "Hand-set inputs; no learned tyre model"],
    ["[2] Heilmeier et al., 2018", "Lap-wise race simulation", "Compares early vs late stops", "Fixed parametric tyre curves"],
    ["[3] Heilmeier et al., 2020", "Monte Carlo race simulation", "Captures random race events", "Model uncertainty not propagated"],
    ["[4] Heilmeier et al., 2020", "Neural networks (VSE)", "Learns pit and compound decisions", "Imitates past decisions; no optimization"],
    ["[5] Liu & Fotouhi, 2020", "NN surrogate + MCTS (Formula E)", "Fast strategy search", "Energy, not tyre degradation"],
    ["[6] Carrasco Heine & Thraves, 2023", "Dynamic programming", "Optimal stop timing and tyres", "Needs explicit discretized model"],
    ["[7] Aguad & Thraves, 2024", "DP + Stackelberg game", "Strategic rival gains >15% win odds", "Large state space"],
    ["[8] Thomas et al., 2025", "Explainable reinforcement learning", "Interpretable strategy agent", "Needs large simulator training"],
    ["[9] Pacejka & Bakker, 1992", "Magic Formula tyre model", "Industry-standard tyre physics", "Needs private test data"],
    ["[10] Chen & Guestrin, 2016", "XGBoost", "Supports custom 2nd-order losses", "Used here with asymmetric loss"],
    ["[11] Breiman, 2001", "Random Forest", "Robust, accurate ensemble", "Symmetric loss only"],
    ["[12] Rasmussen & Williams, 2006", "Gaussian Process regression", "Predictions with uncertainty", "O(n³) training cost"],
    ["[13] Elkan, 2001", "Cost-sensitive learning", "Train/judge by expected cost", "Basis for asymmetric loss and SIS"],
    ["[14] Bergstra et al., 2011", "TPE / Bayesian HPO", "Beats grid/random search", "Used for model tuning"],
    ["[15] Akiba et al., 2019", "Optuna framework", "Efficient HPO with pruning", "Used for model tuning"],
    ["[16] Storn & Price, 1997", "Differential Evolution", "Derivative-free global search", "Chosen optimizer"],
    ["[17] Das & Suganthan, 2011", "DE survey", "Robust on multimodal/noisy problems", "Supports DE choice"],
    ["[18] Kleywegt et al., 2002", "Sample Average Approximation", "Sampled objective converges", "Basis for 50,000-race estimates"],
    ["[19] Glasserman & Yao, 1992", "Common random numbers", "Reduces comparison variance", "Used in optimizer"],
    ["[20] Cleveland, 1979", "LOWESS smoothing", "Robust trend from noisy data", "Used in EDA"],
]
TABLE("Summary of the reviewed literature", ["Study", "Method", "Key contribution", "Limitation / use in this project"],
      lit_rows, [2156, 2290, 2290, 2290], size=18)
H2("3.4 Research Gap")
P("The reviewed work either simulates races with hand-set or parametric tyre models [1]–[3], learns strategy "
  "decisions by imitation [4], [8], or optimizes with exact methods that need an explicit discretized race model "
  "[6], [7]. Few studies (i) learn tyre degradation from public data with a physics-based fuel correction, (ii) "
  "train models with a loss that reflects the cost asymmetry of strategy errors [13], (iii) propagate model "
  "uncertainty into the race simulation, and (iv) search the full pit window with a derivative-free global optimizer "
  "[16], validated with independent seeds. This project combines all four in one reproducible pipeline.")

# ---------------------------------------------------------------- 4. materials and methods
H1("4. Materials and Methods / Models")
H2("4.1 Datasets")
P("All data comes from the official Formula 1 timing feed, accessed through the open-source FastF1 Python "
  "library [23] with fastf1.get_session(2023, 'Bahrain', 'R'). Responses are cached locally (the *cache/* "
  "folder), so the project runs offline. FastF1 provides, for every lap of every driver: lap time, tyre compound, "
  "tyre age, stint number, pit-in and pit-out times, track status (green, yellow, safety car, virtual safety car), "
  "position and an accuracy flag, plus weather samples (air and track temperature, humidity).")
TABLE("Dataset overview", ["Property", "Value"], [
    ["Source", "FastF1 API, 2023 Bahrain Grand Prix, race session"],
    ["Race length", f"{cfg['RACE_LAPS']} laps"],
    ["Laps after cleaning", f"{cfg['n_laps_modeled']} laps, {cfg['n_drivers']} drivers, {cfg['n_stints']} stints"],
    ["Compound mix (cleaned laps)", f"Hard {cfg['compound_counts']['HARD']}, Soft {cfg['compound_counts']['SOFT']}, Medium {cfg['compound_counts']['MEDIUM']}"],
    ["Train / test split", f"{cfg['n_train']} / {cfg['n_test']} laps (whole stints held out)"],
    ["Mean track / air temperature", f"{cfg['MEAN_TRACK_TEMP']:.1f} °C / {cfg['MEAN_AIR_TEMP']:.1f} °C"],
    ["Columns in processed table", f"{laps.shape[1]}"],
], [3000, 6026])
TABLE("Main variables used in the model", ["Variable", "Meaning", "Role"], [
    ["TyreLife", "Laps completed on the current tyre set", "Feature"],
    ["TyreAgeSq, TyreAgeCubed", "TyreLife² and TyreLife³", "Features (curvature, cliff)"],
    ["CompoundEncoded", "Soft = 0, Medium = 1, Hard = 2", "Feature"],
    ["TrackTemp, AirTemp", "Track and air temperature (°C)", "Features"],
    ["LapNumber", "Race lap (track evolution)", "Feature"],
    ["FuelCorrectedLapTime", "Lap time with fuel effect removed (s)", "Intermediate"],
    ["LapTimeDelta", "Fuel-corrected lap time minus the stint's best lap (s)", "Target"],
    ["Driver, Stint, StintID", "Identifiers", "Grouping for split"],
], [2600, 4426, 2000])
H2("4.2 Sample Dataset")
_counts = laps.groupby("StintID").size()
_sid = sorted(_counts[_counts >= 8].index)[0]
samp = laps[laps["StintID"] == _sid].sort_values("LapNumber").head(8)
P(f"Table 4 shows a sample of the processed dataset: the first eight clean laps of one stint ({samp['Driver'].iloc[0]}, "
  f"stint {int(samp['Stint'].iloc[0])}, {samp['Compound'].iloc[0].lower()} tyres). The target *LapTimeDelta* is zero "
  "on the stint's fastest lap and generally rises as the tyre ages.")
TABLE("Sample of the processed dataset", ["Driver", "Lap", "Compound", "Tyre age", "Lap time (s)", "Fuel-corr. (s)",
                                           "Track °C", "Air °C", "LapTimeDelta (s)"],
      [[r.Driver, int(r.LapNumber), r.Compound.title(), int(r.TyreLife), f"{r.LapTimeSeconds:.3f}",
        f"{r.FuelCorrectedLapTime:.3f}", f"{r.TrackTemp:.1f}", f"{r.AirTemp:.1f}", f"{r.LapTimeDelta:.3f}"]
       for r in samp.itertuples()],
      [800, 600, 1200, 900, 1150, 1150, 900, 800, 1526], size=18)
H2("4.3 Data Cleaning")
P("Only laps that represent pure tyre performance are kept:")
numbered([
    "Keep laps flagged accurate by the timing system (*IsAccurate*).",
    "Remove safety-car and virtual-safety-car laps (not representative pace).",
    "Remove pit-in and pit-out laps (they include pit-lane time).",
    "Remove dirty-air laps, where the car is less than 2 s behind another car (aerodynamic loss and tyre "
    "overheating). The gap is reconstructed from cumulative race times.",
    "Remove outliers more than 2.5 standard deviations from the stint mean (lock-ups, mistakes).",
    "Keep only stints with at least 3 laps, so a trend can be estimated.",
])
P(f"The result is {cfg['n_laps_modeled']} laps from {cfg['n_drivers']} drivers and {cfg['n_stints']} stints.")
H2("4.4 Physics-Based Fuel Correction and Target Variable")
P(f"A car starts with about 110 kg of fuel and burns about {cfg['FUEL_PER_LAP']:.2f} kg per lap. Each kilogram "
  f"costs about {cfg['FUEL_EFFECT_K']} s per lap, so the car gains about {cfg['DELTA_PER_LAP']:.4f} s per lap as "
  "fuel burns. This trend works against tyre wear and would hide it, so it is removed analytically:")
EQ(f"FuelCorrectedLapTime = LapTime − (57 − LapNumber) × {cfg['DELTA_PER_LAP']:.4f}")
P("Each car and driver also has a different baseline pace, so the target is measured relative to the stint's "
  "own best lap:")
EQ("LapTimeDelta = FuelCorrectedLapTime − min(FuelCorrectedLapTime over the same stint)")
P("LapTimeDelta is zero on the stint's best lap and grows with tyre wear, and it is comparable across cars.")
H2("4.5 Feature Engineering and Leakage-Safe Split")
P("Seven features are used: TyreLife, TyreLife², TyreLife³ (allowing curvature and a late \"cliff\"), the "
  "encoded compound, track temperature, air temperature and lap number (track evolution). Laps within a stint "
  "are strongly correlated, so a random lap-wise split would leak information. The data is split with "
  "*GroupShuffleSplit* (test size 20%, random state 42) grouped by stint, so every lap of a stint is either in "
  f"training or in testing: {cfg['n_train']} training laps and {cfg['n_test']} test laps.")
H2("4.6 Prediction Models")
H3("Model 1: Linear Regression (baseline)")
P("Ordinary least squares with one coefficient per feature. Because the age² and age³ terms are included, it can "
  "already represent a curved degradation trend. It serves as the baseline every other model must beat.")
H3("Model 2: Polynomial Regression (degree 2)")
P("Linear regression on a degree-2 polynomial expansion of the three tyre-age features, with the other features "
  "entering linearly. The degree was chosen among 1–6 using AIC [25], BIC [26] and 5-fold cross-validation "
  "(Table 5); BIC, which penalizes complexity most, is lowest at degree 2.")
TABLE("Polynomial degree selection (fitted on the hard compound, the largest group)", ["Degree", "AIC", "BIC", "5-fold CV RMSE (s)"],
      [[int(r.degree), f"{r.AIC:.1f}", f"{r.BIC:.2f}", f"{r.CV_RMSE:.4f}"] for r in poly_sel.itertuples()],
      [1800, 2400, 2400, 2426], align_first_left=False)
H3("Model 3: XGBoost with an Asymmetric (Cost-Sensitive) Loss")
P("Gradient-boosted decision trees [10]. In strategy, an optimistic prediction (the model says the tyre is faster "
  "than it really is) keeps the driver out too long and costs the race, while a pessimistic prediction only causes "
  "a slightly early stop. Following cost-sensitive learning [13], the standard squared loss is replaced by an "
  "asymmetric one whose gradient (g) and Hessian (h) are supplied to XGBoost, with residual r = ŷ − y:")
EQ("g = 2.5·r,  h = 2.5   if r < 0 (optimistic error)          g = 1.0·r,  h = 1.0   otherwise")
xp = cfg["best_xgb_params"]
P(f"Hyperparameters were tuned with Optuna's TPE sampler [14], [15] over 100 trials, each scored by 5-fold "
  f"cross-validated RMSE. The best setting is {xp['n_estimators']} trees, depth {xp['max_depth']}, learning rate "
  f"{xp['learning_rate']:.4f}, subsample {xp['subsample']:.2f} and column sample {xp['colsample_bytree']:.2f}. The "
  "final model is trained with early stopping (20 rounds) on the last 10% of the training data. Its test residual "
  f"standard deviation (σ = {cfg['residual_std']:.3f} s) is used as lap noise in the XGBoost simulation engine.")
H3("Model 4: Random Forest")
rp = cfg["best_rf_params"]
P(f"An ensemble of decision trees on bootstrap samples with random feature subsets [11]. Tuned with Optuna over 30 "
  f"trials; the best setting has {rp['n_estimators']} trees, maximum depth {rp['max_depth']}, minimum samples per "
  f"leaf {rp['min_samples_leaf']} and {rp['max_features']:.2f} of features per split.")
H3("Model 5: Gaussian Process Regression")
P("A Bayesian non-parametric model [12] with kernel k(x, x′) = C · Matérn_{5/2}(x, x′; ℓ_{1..7}) + White noise, "
  "fitted on standardized features. The constant, seven length scales and noise level are learned by maximizing "
  "the log marginal likelihood (2 optimizer restarts). The learned kernel is "
  f"{KERNEL}. Very long length scales (1,000) mean a feature has almost no effect on its own. The "
  "Gaussian Process returns a standard deviation for every prediction, which grows for old tyres where data is "
  "sparse; this is used by the uncertainty-aware simulation engine.")
H2("4.7 Evaluation Metrics")
P("Models are compared on the 112 unseen test laps using RMSE, MAE and R², plus two strategy-aware metrics:")
bullets([
    "**Tyre-age-weighted RMSE (TW-RMSE):** errors on old tyres, where the pit decision is made, count up to 4 "
    "times more: TW-RMSE = √( mean( w·(y − ŷ)² ) ), with w = 1 + 3·(age / max age)².",
    "**Strategy Impact Score (SIS):** for each stint, the sum of the positive residuals (y − ŷ > 0, i.e. the model "
    "was optimistic), averaged over stints. It is the race time a strategist would lose per stint by trusting the "
    "model's optimism. Lower is better.",
])
H2("4.8 Monte Carlo Race Simulator")
mc = cfg["MC"]
P(f"From lap {mc['CURRENT_LAP']}, both cars have {mc['RACE_TOTAL_LAPS'] - mc['CURRENT_LAP']} laps left. Each "
  "simulated race walks forward lap by lap:")
numbered([
    "Tyre age increases by one lap for both cars.",
    f"If it is our pit lap, add {mc['PIT_STOP_DELTA']:.0f} s and switch to new hard tyres; the rival pits on "
    "lap 45.",
    "Add the predicted lap-time loss for each car's tyre state plus random noise.",
    f"Every 10 laps, a safety car is triggered with probability {mc['SAFETY_CAR_PROBABILITY']:.2f}; it neutralizes "
    "3 laps for both cars.",
    f"At the end, final gap = (rival total + {mc['RIVAL_GAP_SECONDS']} s head start) − our total; we win if the gap "
    "is positive.",
])
P(f"**Engine A (XGBoost)** uses one fixed noise level σ = {cfg['residual_std']:.3f} s on every lap. "
  "**Engine B (Gaussian Process)** draws noise from the Gaussian Process's own per-lap standard deviation, which "
  "grows with tyre age. Because a strategy's tyre states are known in advance, all unique lap states are predicted "
  "once in a batch and the races are simulated as NumPy arrays, which turns hours of computation into seconds. "
  f"The Review 2 baseline evaluated four pit laps ({', '.join(str(mc['CURRENT_LAP'] + s) for s in mc['PIT_STRATEGIES'])}) "
  f"with {mc['N_SIMULATIONS']:,} races each.")
H2("4.9 Pit-Lap Optimization with Differential Evolution")
P("The final decision is formulated as an integer stochastic optimization problem over the pit lap p:")
EQ("maximize  J(p) = ½ [ P_{XGB}(win | p) + P_{GP}(win | p) ] + 10^{−6} · G(p),   p ∈ {31, …, 56}")
P("Here G(p) is the mean of the two engines' median finishing gaps. Averaging the two engines makes the decision "
  "robust to either model's assumptions; the tiny second term only breaks ties. J(p) has no formula: each value is a Monte Carlo estimate (Sample Average Approximation [18]), so "
  "it is noisy, integer-valued, not differentiable and possibly multimodal. Differential Evolution [16], [17] "
  "handles all of these. Using SciPy's implementation [22], each generation works as follows:")
numbered([
    "**Initialization:** a population of 12 candidate pit laps spread over laps 31–56.",
    "**Mutation:** for each candidate, a mutant is built from the current best plus a scaled difference of two "
    "other random candidates: v = x_{best} + F·(x_{r1} − x_{r2}), with F drawn from [0.5, 1.0] (strategy "
    "*best1bin*).",
    "**Crossover:** the mutant is mixed with the original candidate (crossover rate 0.7) and rounded to an integer lap.",
    "**Selection:** the trial lap is simulated (50,000 races per engine) and replaces the original only if J improves.",
    "**Termination:** repeat until the population converges (tolerance 10^{−4}) or 30 generations pass.",
])
P("**Common random numbers [19]:** every candidate lap is evaluated on the same safety-car and lap-noise draws, so "
  "differences between laps come from the strategy, not from luck. Independent seeds are used only for the final "
  "validation (5 seeds × 20,000 races). The optimizer was run with three DE seeds (42, 7 and 2026).")
TABLE("Differential Evolution settings", ["Parameter", "Value"], [
    ["Decision variable", "Pit lap p, integer, bounds 31–56"],
    ["Strategy", "best1bin (SciPy differential_evolution)"],
    ["Population size", f"{opt['popsize']} candidates"],
    ["Mutation factor F", "Random in [0.5, 1.0] (dithering)"],
    ["Crossover rate", "0.7"],
    ["Maximum generations / tolerance", f"{opt['maxiter']} / 1e-4"],
    ["Races per candidate", f"{opt['n_simulations']:,} per engine, common random numbers"],
    ["DE seeds", ", ".join(str(s) for s in opt["de_seeds"])],
    ["Validation", f"{len(opt['validation_seeds'])} fresh seeds × {opt['validation_simulations']:,} races"],
], [3500, 5526])
P("For comparison, three classical methods were also implemented: the baseline grid search (4 points), "
  "golden-section search [24] (on the full window and on the old grid's bracket 42–51), and Newton–Raphson on a "
  "local quadratic surrogate started from lap 42.")
H2("4.10 Implementation and Tools")
TABLE("Software and hardware used", ["Component", "Details"], [
    ["Language / environment", "Python 3.12 (virtual environment), Jupyter Notebook"],
    ["Data", "FastF1 3.9.0, pandas 2.3.3, NumPy 2.5.4"],
    ["Machine learning", "scikit-learn 1.9.1 [21], XGBoost 3.4.1, Optuna 5.0.0, statsmodels 0.15.0 (LOWESS)"],
    ["Optimization", "SciPy 1.18.1 differential_evolution, golden-section and Newton benchmarks [22]"],
    ["Visualization / app", "Matplotlib 3.11.2, Plotly 7.1.0, Streamlit 1.66.1 [27]"],
    ["Model storage", "joblib 1.6.0 (trained models), Parquet / CSV (results)"],
    ["Hardware", "Windows 11 laptop, 22 logical CPU cores, 15 GB RAM; runs fully offline"],
], [2800, 6226])
P("The code is organized in three layers. The notebook trains all models and exports them to *artifacts/*. "
  "The script *optimizer.py* runs the Differential Evolution search and benchmarks and saves its results. The "
  "Streamlit app *app.py* is a read-only viewer of these artifacts with a live what-if simulator, so the demo "
  "never retrains models and starts in seconds.")

# ---------------------------------------------------------------- 5. results and discussion
H1("5. Results and Discussion")
H2("5.1 Exploratory Analysis")
P("Figure 2 shows lap-time loss against tyre age for each compound with a LOWESS trend [20]. Loss grows steadily "
  "with age for both main compounds, at roughly 0.1–0.15 s per lap, and the soft tyre pulls away from the hard "
  "after about 18 laps. Medium has only 8 clean laps, so its curve is unreliable and is not interpreted.")
FIGURE("fig02_degradation.png", "Lap-time loss (LapTimeDelta) versus tyre age with LOWESS trends per compound.", 5.8)
H2("5.2 Model Performance")
TABLE("Test-set performance of the five models (112 laps from unseen stints)",
      ["Model", "RMSE (s)", "MAE (s)", "R²", "TW-RMSE (s)", "SIS (s/stint)", "Train time (s)"],
      [[r.Index, f"{r[1]:.3f}", f"{r[2]:.3f}", f"{r[3]:.3f}", f"{r[4]:.3f}", f"{r[5]:.2f}", f"{r[6]:.3f}"]
       for r in metrics.set_index("Model").itertuples()],
      [2226, 1100, 1100, 1000, 1200, 1200, 1200])
FIGURE("fig04_model_metrics.png", "RMSE and Strategy Impact Score for each model (best model highlighted).")
P(f"Random Forest has the lowest RMSE ({m.loc['Random Forest', 'RMSE (s)']:.3f} s) and highest R² "
  f"({m.loc['Random Forest', 'R²']:.3f}), but the Gaussian Process has the lowest MAE "
  f"({m.loc['Gaussian Process', 'MAE (s)']:.3f} s) and by far the lowest SIS "
  f"({m.loc['Gaussian Process', 'SIS (s)']:.2f} s per stint). The asymmetric XGBoost has a lower SIS "
  f"({m.loc['XGBoost (Asymmetric)', 'SIS (s)']:.2f} s) than Random Forest "
  f"({m.loc['Random Forest', 'SIS (s)']:.2f} s) despite a slightly higher RMSE, because its loss specifically "
  "suppresses optimistic errors. **The model with the lowest RMSE is not the model that loses the least race time.** "
  f"The polynomial model is the weakest (RMSE {m.loc['Polynomial (Deg 2)', 'RMSE (s)']:.3f} s), showing that a "
  "single rigid curve cannot capture compound and temperature effects. An RMSE of about 0.65 s is reasonable, since "
  "normal lap-to-lap variation in Formula 1 is of a similar size and the test stints were never seen in training.")
H2("5.3 Prediction Output")
P("Table 9 shows the actual output of all five trained models on eight laps from the unseen test set, chosen "
  "across both compounds and a range of tyre ages. The last column is the Gaussian Process's own uncertainty (±1σ).")
TABLE("Prediction output on unseen test laps: LapTimeDelta in seconds",
      ["Driver", "Lap", "Compound", "Tyre age", "Actual", "Linear", "Poly", "XGB", "RF", "GP", "GP ±σ"],
      [[r["Driver"], r["Lap"], r["Compound"], r["TyreAge"], f"{r['Actual']:.3f}", f"{r['Linear']:.3f}",
        f"{r['Poly']:.3f}", f"{r['XGB']:.3f}", f"{r['RF']:.3f}", f"{r['GP']:.3f}", f"{r['GPstd']:.3f}"]
       for r in sample_preds],
      [760, 560, 1180, 800, 800, 800, 800, 800, 800, 800, 926], size=18)
FIGURE("fig03_pred_vs_actual.png", "Predicted versus actual LapTimeDelta on the test set for the two models used "
       "in the race simulator (dashed line = perfect prediction).")
P("The models follow the actual degradation well on average, while individual laps scatter by a few tenths of a "
  "second because of driver and traffic variation. Table 10 shows the forecast for a new scenario: the predicted "
  "time loss after a given number of laps on soft and hard tyres at the race's mean temperatures. The Gaussian "
  "Process's uncertainty grows from about ±0.5 s on fresh tyres to more than ±1.2 s at 25 laps, which is exactly "
  "where strategy risk lies.")
TABLE("Scenario forecast: predicted time loss (s) by tyre age at mean race temperatures, lap 40",
      ["Compound", "Tyre age (laps)", "XGBoost prediction", "GP prediction", "GP uncertainty (±σ)"],
      [[r["Compound"], r["TyreAge"], f"{r['XGB']:.3f}", f"{r['GP']:.3f}", f"{r['GPstd']:.3f}"] for r in scenario_preds],
      [1500, 1700, 1900, 1900, 2026])
P("Figure 5 shows the models tracking Sergio Pérez's final stint lap by lap. The cumulative drift of the predicted "
  "stint time is only 0.58 s over 20 laps, well below the margins on which strategy calls are made.")
FIGURE("fig05_per_stint.png", "Actual versus predicted fuel-corrected lap times for Pérez's final stint (laps 36–57).", 5.8)
H2("5.4 Baseline Monte Carlo Strategy Grid (Review 2)")
TABLE("Baseline grid: 10,000 simulated races per pit lap",
      ["Pit lap", "XGB win %", "XGB median gap (s)", "XGB P10–P90 (s)", "GP win %", "GP median gap (s)", "GP P10–P90 (s)"],
      [[int(r.ActualPitLap), pct(r.WinPct_XGB), f"{r.MedianGap_XGB:+.2f}", f"{r.P10Gap_XGB:+.1f} to {r.P90Gap_XGB:+.1f}",
        pct(r.WinPct_GP), f"{r.MedianGap_GP:+.2f}", f"{r.P10Gap_GP:+.1f} to {r.P90Gap_GP:+.1f}"] for r in win.itertuples()],
      [900, 1150, 1400, 1500, 1150, 1400, 1526])
FIGURE("fig06_mc_grid.png", "Win probability of the four baseline pit laps for each simulation engine.", 5.8)
P("Both engines agree that pitting earlier is better, but they disagree strongly about confidence: at lap 42 the "
  "XGBoost engine gives 99.65% while the Gaussian Process engine gives 57.0%. XGBoost uses one small, fixed noise "
  "level on every lap and so becomes over-confident, while the Gaussian Process knows that uncertainty grows on "
  "old tyres. The main weakness of the grid is that it only tests 4 of the 26 possible pit laps.")
H2("5.5 Differential Evolution Optimization Results")
P(f"Differential Evolution searched every lap from 31 to 56 and converged to **lap {best}** in "
  f"{opt['n_iterations']} generations and {opt['n_evaluations']} unique lap evaluations "
  f"({opt['runtime_s']:.2f} s). At lap {best} the XGBoost engine gives {pct(opt['xgb_win_pct'])}, the Gaussian Process "
  f"engine {pct(opt['gp_win_pct'])} and the ensemble {pct(opt['ensemble_win_pct'])}, against "
  f"{pct(opt['grid_best_ensemble_win_pct'])} for the grid's lap 42: a gain of **+{opt['improvement_pp']:.2f} "
  f"percentage points**, with median finishing margins up by {opt['xgb_median_gap_gain']:+.2f} s (XGBoost) and "
  f"{opt['gp_median_gap_gain']:+.2f} s (Gaussian Process). Figure 7 shows the full landscape; the peak lies "
  "outside the window the grid examined.")
FIGURE("fig07_landscape.png", "Win probability for every feasible pit lap (post-hoc scan), with the grid's "
       "candidates and the DE optimum.")
TABLE("Comparison of optimization algorithms", ["Algorithm", "Search range", "Best lap", "Ensemble win", "Evaluations", "Verdict"],
      [[r.Algorithm, r.SearchRange, int(r.BestPitLap), pct(r.Ensemble_WinPct), int(r.Evaluations), r.Verdict]
       for r in comparison.itertuples()],
      [1900, 1300, 800, 1100, 1100, 2826], size=17)
FIGURE("fig08_benchmark.png", "Best pit lap and win probability found by each optimization method.", 5.8)
P("Grid search misses the peak because it never looks there. Golden-section search finds lap 34 only when it is "
  "given the full window; started on the old bracket (42–51) it stops at the bracket edge, repeating the grid's "
  "bias, and it assumes a single peak, which the Gaussian Process engine violates. Newton–Raphson needs a smooth "
  "derivative: started from lap 42 it jumps to lap 55 (29.56% win), and its roots from different starts ranged "
  "from lap 5 to lap 148, outside the race. Differential Evolution is the only method that is global, works "
  "directly on integer laps and needs no derivative. Table 13 shows that all three DE seeds return the same answer.")
TABLE("Differential Evolution runs with three seeds", ["Seed", "Best lap", "Ensemble win", "Unique evaluations", "Generations", "Runtime (s)"],
      [[int(r.Seed), int(r.BestPitLap), pct(r.Ensemble_WinPct), int(r.UniqueEvaluations), int(r.NIter), f"{r.Runtime_s:.2f}"]
       for r in seeds.itertuples()],
      [1200, 1300, 1600, 1800, 1500, 1626], align_first_left=False)
H2("5.6 Independent Validation")
vrows = []
for name in ["DE optimum", "GP-only optimum", "XGB-only optimum", "Grid best (baseline)"]:
    r = validation.loc[name]
    vrows.append([name, int(r.PitLap), pct(r.XGB_WinPct), pct(r.GP_WinPct),
                  f"{pct(r.Ensemble_WinPct)} ± {r.Ensemble_WinPct_CI * 100:.2f}", f"{r.XGB_MedianGap:+.2f}", f"{r.GP_MedianGap:+.2f}"])
TABLE("Validation on 5 fresh seeds × 20,000 races (95% confidence interval)",
      ["Scenario", "Pit lap", "XGB win", "GP win", "Ensemble win", "XGB median gap (s)", "GP median gap (s)"],
      vrows, [2000, 900, 1100, 1100, 1626, 1150, 1150])
FIGURE("fig09_validation.png", "Ensemble win probability on independent validation seeds.", 5.8)
P(f"On completely new random scenarios the DE optimum keeps a {pct(validation.loc['DE optimum', 'Ensemble_WinPct'])} "
  f"ensemble win probability against {pct(validation.loc['Grid best (baseline)', 'Ensemble_WinPct'])} for the grid's "
  "lap 42, so the gain is not an artifact of the optimization sample. Optimizing one engine alone moves the answer "
  "only slightly (XGBoost-only: lap 36; Gaussian-Process-only: lap 33), so all variants agree on an early stop "
  "between laps 33 and 36.")
H2("5.7 Working Model Demonstration")
P("The system runs locally with three one-click launchers: *run_dashboard.bat* (dashboard), *run_optimizer.bat* "
  "(optimizer) and *open_notebook.bat* (the executed notebook). Figure 10 shows the dashboard's overview tab with "
  "the recommended pit lap. Figure 11 shows the live what-if simulator: choosing pit lap 34 and running 10,000 "
  "Gaussian Process races returns a 100.0% win probability with a median margin of +19.77 s. Figure 12 shows the "
  "optimizer tab.")
FIGURE("fig10_dashboard_overview.png", "Streamlit dashboard: overview tab with key metrics and findings.", 5.3)
FIGURE("fig11_dashboard_whatif.png", "Live what-if simulator: 10,000 races for pit lap 34 run on a button click.", 5.3)
FIGURE("fig12_dashboard_optimizer.png", "Optimizer tab: problem formulation and algorithm comparison.", 5.3)
P("Running the optimizer from the command line prints the decision directly:")
CODE([
    "> python optimizer.py",
    "==============================================================================",
    "FULL-WINDOW PIT-STRATEGY OPTIMIZATION",
    "==============================================================================",
    "Baseline grid best : lap 42 | ensemble 78.00% | XGB 99.64% | GP 56.35%",
    "DE optimum         : lap 34 | ensemble 99.99% | XGB 99.99% | GP 99.98%",
    "Gain               : +21.99 pp ensemble win | XGB margin +5.06 s | GP margin +18.66 s",
    f"DE effort          : {opt['n_evaluations']} unique lap evaluations, {opt['runtime_s']:.2f} s",
    "DE seeds           : 42, 7, 2026",
    "==============================================================================",
])
H2("5.8 Discussion")
bullets([
    "**Strategy-aware evaluation changes the choice of model.** Ranking by RMSE favours Random Forest; ranking by "
    "strategy cost (SIS) favours the Gaussian Process and the asymmetric XGBoost. A strategy system should be judged "
    "by the cost of its errors, as cost-sensitive learning suggests [13].",
    "**Honest uncertainty matters.** The fixed-noise XGBoost engine produces near-certain win probabilities, while "
    "the Gaussian Process engine gives realistic ones. Averaging both protects the decision from either model's "
    "assumptions.",
    "**The algorithm matters as much as the model.** With the same models, switching from a 4-point grid to Differential "
    "Evolution raised the win probability by about 22 percentage points.",
    "**Comparison with reality.** Real teams made their final-window stops between laps 35 and 50 (median lap 41). "
    "The grid's lap 42 sits in the middle of that range; the optimizer's lap 34 is an aggressive undercut one lap "
    "before the earliest real stop, aimed specifically at a rival who pits on lap 45.",
    "**Robustness to retraining.** The notebook's Optuna search is randomized; a full re-run produced slightly "
    "different XGBoost settings (test RMSE 0.672 s) and an optimum of lap 33, inside the same 33–36 window. The "
    "recommendation is therefore stable.",
])
H2("5.9 Limitations")
bullets([
    "Only one race (Bahrain 2023) is modelled; conclusions are specific to this circuit and its conditions.",
    "Only 8 clean medium-compound laps are available, so medium-tyre behaviour is weakly supported.",
    "The rival follows a fixed rule (pits on lap 45) and does not react to our decision.",
    "The safety-car model is simple (15% per 10-lap block, applied to both cars equally).",
    "Only single-stop strategies with a fixed compound change are optimized.",
    "Gaussian Process training scales as O(n³), which limits scaling to a full season without approximations.",
])

# ---------------------------------------------------------------- 6. conclusions
H1("6. Conclusions and Future Work")
H2("6.1 Conclusions")
P("This project built a complete, reproducible decision system for Formula 1 pit-stop timing from public timing "
  "data. A physics-based fuel correction and careful cleaning produced a target that isolates tyre degradation. "
  "Five models were compared on unseen stints; the Random Forest was most accurate, but the Gaussian Process and "
  "the cost-sensitive XGBoost caused the least strategy damage, showing that strategy models must be judged by the "
  "cost of their errors. A two-engine Monte Carlo simulator turned predictions into win probabilities while "
  "propagating model uncertainty and safety-car risk.")
P(f"The Differential Evolution optimizer found the optimal pit lap, **lap {best}**, with a "
  f"{pct(opt['ensemble_win_pct'])} win probability, versus {pct(opt['grid_best_ensemble_win_pct'])} for the Review 2 "
  f"baseline grid (+{opt['improvement_pp']:.2f} percentage points), in only {opt['n_evaluations']} evaluations. "
  "Grid search, golden-section search and Newton–Raphson were shown to fail for specific, explainable reasons. "
  "The result was confirmed on independent seeds and on three DE seeds, and the full system runs as an "
  "interactive dashboard.")
H2("6.2 Future Work")
bullets([
    "Extend the optimization to two-stop strategies and the choice of compound, where Differential Evolution's "
    "advantage over exhaustive search grows.",
    "Model a reacting rival with game theory, building on Aguad and Thraves [7].",
    "Re-plan every lap with live data (sequential decision-making or reinforcement learning [8]).",
    "Train on multiple races and seasons with circuit-specific features.",
    "Use sparse Gaussian Processes or Bayesian neural networks for better-calibrated uncertainty at scale.",
    "Stream live timing into the dashboard for in-race decision support.",
])

# ---------------------------------------------------------------- 7. references
H1("7. References")
refs = [
    "J. Bekker and W. Lotz, \"Planning Formula One race strategies using discrete-event simulation,\" *Journal of the Operational Research Society*, vol. 60, no. 7, pp. 952–961, 2009.",
    "A. Heilmeier, M. Graf, and M. Lienkamp, \"A race simulation for strategy decisions in circuit motorsports,\" in *Proc. 21st Int. Conf. Intelligent Transportation Systems (ITSC)*, Maui, HI, USA, 2018, doi: 10.1109/ITSC.2018.8570012.",
    "A. Heilmeier, M. Graf, J. Betz, and M. Lienkamp, \"Application of Monte Carlo methods to consider probabilistic effects in a race simulation for circuit motorsport,\" *Applied Sciences*, vol. 10, no. 12, Art. no. 4229, 2020, doi: 10.3390/app10124229.",
    "A. Heilmeier, A. Thomaser, M. Graf, and J. Betz, \"Virtual strategy engineer: Using artificial neural networks for making race strategy decisions in circuit motorsport,\" *Applied Sciences*, vol. 10, no. 21, Art. no. 7805, 2020, doi: 10.3390/app10217805.",
    "X. Liu and A. Fotouhi, \"Formula-E race strategy development using artificial neural networks and Monte Carlo tree search,\" *Neural Computing and Applications*, 2020, doi: 10.1007/s00521-020-04871-1.",
    "O. F. Carrasco Heine and C. Thraves, \"On the optimization of pit stop strategies via dynamic programming,\" *Central European Journal of Operations Research*, vol. 31, no. 1, 2023, doi: 10.1007/s10100-022-00806-4.",
    "F. Aguad and C. Thraves, \"Optimizing pit stop strategies in Formula 1 with dynamic programming and game theory,\" *European Journal of Operational Research*, vol. 319, no. 3, pp. 908–919, 2024.",
    "D. Thomas, J. Jiang, A. Kori, A. Russo, S. Winkler, S. Sale, J. McMillan, F. Belardinelli, and A. Rago, \"Explainable reinforcement learning for Formula One race strategy,\" arXiv:2501.04068, 2025.",
    "H. B. Pacejka and E. Bakker, \"The magic formula tyre model,\" *Vehicle System Dynamics*, vol. 21, no. sup1, pp. 1–18, 1992.",
    "T. Chen and C. Guestrin, \"XGBoost: A scalable tree boosting system,\" in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, 2016, pp. 785–794.",
    "L. Breiman, \"Random forests,\" *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.",
    "C. E. Rasmussen and C. K. I. Williams, *Gaussian Processes for Machine Learning*. Cambridge, MA, USA: MIT Press, 2006.",
    "C. Elkan, \"The foundations of cost-sensitive learning,\" in *Proc. 17th Int. Joint Conf. Artificial Intelligence (IJCAI)*, 2001, pp. 973–978.",
    "J. Bergstra, R. Bardenet, Y. Bengio, and B. Kégl, \"Algorithms for hyper-parameter optimization,\" in *Advances in Neural Information Processing Systems 24*, 2011, pp. 2546–2554.",
    "T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, \"Optuna: A next-generation hyperparameter optimization framework,\" in *Proc. 25th ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, 2019, pp. 2623–2631.",
    "R. Storn and K. Price, \"Differential evolution – A simple and efficient heuristic for global optimization over continuous spaces,\" *Journal of Global Optimization*, vol. 11, no. 4, pp. 341–359, 1997.",
    "S. Das and P. N. Suganthan, \"Differential evolution: A survey of the state-of-the-art,\" *IEEE Transactions on Evolutionary Computation*, vol. 15, no. 1, pp. 4–31, 2011.",
    "A. J. Kleywegt, A. Shapiro, and T. Homem-de-Mello, \"The sample average approximation method for stochastic discrete optimization,\" *SIAM Journal on Optimization*, vol. 12, no. 2, pp. 479–502, 2002.",
    "P. Glasserman and D. D. Yao, \"Some guidelines and guarantees for common random numbers,\" *Management Science*, vol. 38, no. 6, pp. 884–908, 1992.",
    "W. S. Cleveland, \"Robust locally weighted regression and smoothing scatterplots,\" *Journal of the American Statistical Association*, vol. 74, no. 368, pp. 829–836, 1979.",
    "F. Pedregosa et al., \"Scikit-learn: Machine learning in Python,\" *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.",
    "P. Virtanen et al., \"SciPy 1.0: Fundamental algorithms for scientific computing in Python,\" *Nature Methods*, vol. 17, pp. 261–272, 2020.",
    "FastF1 contributors, \"FastF1: A Python package for accessing and analyzing Formula 1 results, schedules, timing data and telemetry,\" GitHub repository. [Online]. Available: https://github.com/theOehrly/Fast-F1",
    "J. Kiefer, \"Sequential minimax search for a maximum,\" *Proceedings of the American Mathematical Society*, vol. 4, no. 3, pp. 502–506, 1953.",
    "H. Akaike, \"A new look at the statistical model identification,\" *IEEE Transactions on Automatic Control*, vol. 19, no. 6, pp. 716–723, 1974.",
    "G. Schwarz, \"Estimating the dimension of a model,\" *The Annals of Statistics*, vol. 6, no. 2, pp. 461–464, 1978.",
    "Streamlit, \"Streamlit: A faster way to build and share data apps.\" [Online]. Available: https://streamlit.io",
]
for i, ref in enumerate(refs, 1):
    D.body.append(f'<w:p>{ppr(align="both", after=100, ind_left=567, hanging=567)}{runs(f"[{i}]")}'
                  f'<w:r><w:tab/></w:r>{runs(ref)}</w:p>')

# ---------------------------------------------------------------- 8. appendix
H1("8. Appendix")
H2("A. Source Code (GitHub)")
LINK("The complete source code, trained models and results are available at: ", GITHUB, ".")
H2("B. How to Run the Project")
CODE([
    "# 1. Dashboard (opens http://localhost:8501)",
    "streamlit run app.py                 # or double-click run_dashboard.bat",
    "",
    "# 2. Pit-lap optimizer (prints grid vs Differential Evolution result)",
    "python optimizer.py                  # or double-click run_optimizer.bat",
    "python optimizer.py --no-save --seed 7   # re-run with another random seed",
    "",
    "# 3. Full training pipeline (retrains all models, about 30 minutes)",
    "jupyter nbconvert --to notebook --execute f1-tire-degradation-performance-decay-analysis.ipynb",
])
H2("C. Final Hyperparameters")
TABLE("Tuned hyperparameters (from artifacts/config.json)", ["Model", "Hyperparameters"], [
    ["XGBoost (100 Optuna trials)", f"n_estimators={xp['n_estimators']}, max_depth={xp['max_depth']}, learning_rate={xp['learning_rate']:.4f}, "
     f"subsample={xp['subsample']:.3f}, colsample_bytree={xp['colsample_bytree']:.3f}, reg_alpha={xp['reg_alpha']:.1e}, reg_lambda={xp['reg_lambda']:.1e}"],
    ["Random Forest (30 Optuna trials)", f"n_estimators={rp['n_estimators']}, max_depth={rp['max_depth']}, min_samples_leaf={rp['min_samples_leaf']}, "
     f"min_samples_split={rp['min_samples_split']}, max_features={rp['max_features']:.3f}"],
    ["Gaussian Process", KERNEL],
    ["Polynomial Regression", f"degree={cfg['optimal_degree']} on tyre-age features"],
    ["Asymmetric loss", "alpha_under = 2.5, alpha_over = 1.0"],
], [2800, 6226], size=18)
H2("D. Repository Structure")
TABLE("Main files in the repository", ["File / folder", "Purpose"], [
    ["f1-tire-degradation-performance-decay-analysis.ipynb", "Data processing, EDA, model training, Monte Carlo grid, exports"],
    ["optimizer.py", "Differential Evolution pit-lap optimizer, benchmarks and validation"],
    ["app.py", "Streamlit dashboard (7 tabs, live what-if simulator)"],
    ["artifacts/", "Trained models (joblib), processed data (Parquet) and result tables (CSV)"],
    ["cache/", "FastF1 cache of the raw 2023 Bahrain Grand Prix data"],
    ["get_data.py", "Downloads the raw data into cache/ (not needed when the cache exists)"],
    ["README.md", "Full project documentation"],
], [4200, 4826], size=18)


# =============================================================== packaging
NS = ('xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
      'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
      'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
      'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
      'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
      'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"')

SECT = ('<w:sectPr><w:footerReference w:type="default" r:id="rIdFooter1"/>'
        '<w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>'
        '<w:pgBorders w:offsetFrom="page"><w:top w:val="single" w:sz="4" w:space="24" w:color="auto"/>'
        '<w:left w:val="single" w:sz="4" w:space="24" w:color="auto"/><w:bottom w:val="single" w:sz="4" w:space="24" w:color="auto"/>'
        '<w:right w:val="single" w:sz="4" w:space="24" w:color="auto"/></w:pgBorders>'
        '<w:cols w:space="708"/><w:titlePg/><w:docGrid w:linePitch="360"/></w:sectPr>')

FOOTER = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
          'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
          f'<w:p><w:pPr><w:jc w:val="center"/></w:pPr>'
          f'<w:r>{rpr(size=20)}<w:fldChar w:fldCharType="begin"/></w:r>'
          f'<w:r>{rpr(size=20)}<w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
          f'<w:r>{rpr(size=20)}<w:fldChar w:fldCharType="separate"/></w:r>'
          f'<w:r>{rpr(size=20)}<w:t>2</w:t></w:r>'
          f'<w:r>{rpr(size=20)}<w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>')

BULLET_ABS = ('<w:abstractNum w:abstractNumId="90"><w:multiLevelType w:val="hybridMultilevel"/>'
              '<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="•"/>'
              '<w:lvlJc w:val="left"/><w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr>'
              '<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:hint="default"/></w:rPr></w:lvl>'
              '</w:abstractNum>'
              '<w:abstractNum w:abstractNumId="91"><w:multiLevelType w:val="hybridMultilevel"/>'
              '<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/><w:lvlText w:val="%1."/>'
              '<w:lvlJc w:val="left"/><w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr>'
              '<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:hint="default"/></w:rPr></w:lvl>'
              '</w:abstractNum>')


def build():
    build_dir = ROOT / "report" / "_build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    with zipfile.ZipFile(TEMPLATE) as z:
        z.extractall(build_dir)
    shutil.rmtree(build_dir / "word" / "media", ignore_errors=True)
    shutil.rmtree(build_dir / "customXml", ignore_errors=True)
    (build_dir / "word" / "media").mkdir()

    # numbering: bullets (90) + one restartable decimal list per numbered block
    num_path = build_dir / "word" / "numbering.xml"
    num = num_path.read_text(encoding="utf-8")
    first_num = num.index("<w:num ")
    nums = '<w:num w:numId="90"><w:abstractNumId w:val="90"/></w:num>'
    for i in range(1, D.num_lists + 1):
        nums += (f'<w:num w:numId="{200 + i}"><w:abstractNumId w:val="91"/>'
                 '<w:lvlOverride w:ilvl="0"><w:startOverride w:val="1"/></w:lvlOverride></w:num>')
    num = num[:first_num] + BULLET_ABS + num[first_num:]
    num = num.replace("</w:numbering>", nums + "</w:numbering>")
    num_path.write_text(num, encoding="utf-8")

    # images
    for rid, name, src in D.images:
        shutil.copy(src, build_dir / "word" / "media" / name)

    rels = ['<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>',
            '<Relationship Id="rIdNumbering" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>',
            '<Relationship Id="rIdSettings" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>',
            '<Relationship Id="rIdWeb" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/webSettings" Target="webSettings.xml"/>',
            '<Relationship Id="rIdFonts" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/fontTable" Target="fontTable.xml"/>',
            '<Relationship Id="rIdTheme" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>',
            '<Relationship Id="rIdFooter1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/>']
    rels += [f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{name}"/>'
             for rid, name, _ in D.images]
    rels += [f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" Target="{escape(url)}" TargetMode="External"/>'
             for rid, url in D.links]
    (build_dir / "word" / "_rels" / "document.xml.rels").write_text(
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + "".join(rels) + "</Relationships>",
        encoding="utf-8")
    (build_dir / "word" / "footer1.xml").write_text(FOOTER, encoding="utf-8")

    ct_path = build_dir / "[Content_Types].xml"
    ct = ct_path.read_text(encoding="utf-8")
    ct = re.sub(r'<Override PartName="/customXml/[^>]+/>', "", ct)
    ct = ct.replace("</Types>", '<Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/></Types>')
    ct_path.write_text(ct, encoding="utf-8")

    (build_dir / "docProps" / "core.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        '<dc:title>Data-Driven Formula 1 Pit-Stop Strategy: Final Report</dc:title>'
        '<dc:creator>B Sudharshan, Kunwar Arush Srivastava, Dharanitharan D</dc:creator>'
        '<cp:lastModifiedBy>Kunwar Arush Srivastava</cp:lastModifiedBy>'
        '<dcterms:created xsi:type="dcterms:W3CDTF">2026-10-26T00:00:00Z</dcterms:created>'
        '<dcterms:modified xsi:type="dcterms:W3CDTF">2026-10-26T00:00:00Z</dcterms:modified>'
        '</cp:coreProperties>', encoding="utf-8")

    document = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document {NS}><w:body>'
                + "".join(D.body) + SECT + "</w:body></w:document>")
    (build_dir / "word" / "document.xml").write_text(document, encoding="utf-8")

    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        ct_file = build_dir / "[Content_Types].xml"
        z.write(ct_file, "[Content_Types].xml")
        for p in sorted(build_dir.rglob("*")):
            if p.is_file() and p != ct_file:
                z.write(p, p.relative_to(build_dir).as_posix())
    shutil.rmtree(build_dir)
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB): {D.fig_no} figures, {D.tab_no} tables, {len(refs)} references")


if __name__ == "__main__":
    build()
