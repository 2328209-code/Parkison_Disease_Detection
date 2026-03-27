"""
Parkinson's Disease Detection — Streamlit Dashboard
====================================================
Report pipeline: Yeo-Johnson → LASSO+RFECV → SVM-RBF / Deep MLP
               → Optuna → Platt Calibration → SHAP

Run:
    streamlit run app.py
"""

import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score)
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import VarianceThreshold, RFECV
from sklearn.linear_model import LassoCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, brier_score_loss,
                             roc_curve, confusion_matrix, classification_report)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import shap

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & DARK THEME
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Parkinson's Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] {
    background-color: #0a0a0a !important;
    color: #e0e0e0 !important;
    font-family: 'Courier New', monospace !important;
}
[data-testid="stSidebar"] {
    background-color: #111111 !important;
    border-right: 1px solid #2a2a2a !important;
}
[data-testid="stSidebar"] * { color: #cccccc !important; }
[data-testid="stMetric"] {
    background: #141414 !important;
    border: 1px solid #2a9d8f !important;
    border-radius: 8px !important;
    padding: 12px !important;
}
[data-testid="stMetricValue"] { color: #2a9d8f !important; font-size: 2rem !important; }
[data-testid="stMetricLabel"] { color: #888888 !important; }
.stButton > button {
    background: linear-gradient(135deg, #2a9d8f, #1a7a6e) !important;
    color: white !important; border: none !important;
    border-radius: 6px !important; font-weight: bold !important;
    letter-spacing: 1px !important; padding: 0.5rem 1.5rem !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #38c9b6, #2a9d8f) !important;
    box-shadow: 0 0 18px rgba(42,157,143,0.5) !important;
}
[data-testid="stFileUploader"] {
    border: 1px dashed #2a9d8f !important;
    border-radius: 8px !important; padding: 10px !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: #111111 !important; border-bottom: 1px solid #2a2a2a !important;
}
.stTabs [data-baseweb="tab"] {
    color: #888888 !important; font-family: 'Courier New', monospace !important;
}
.stTabs [aria-selected="true"] {
    color: #2a9d8f !important; border-bottom: 2px solid #2a9d8f !important;
}
h1, h2, h3 { color: #2a9d8f !important; letter-spacing: 2px !important; }
h1 { border-bottom: 1px solid #2a2a2a; padding-bottom: 10px; }
.streamlit-expanderHeader {
    background: #141414 !important;
    border: 1px solid #2a2a2a !important;
    color: #2a9d8f !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# THEME CONSTANTS & HELPERS
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG  = "#0a0a0a"
CARD_BG  = "#141414"
TEAL     = "#2a9d8f"
TEAL2    = "#38c9b6"
ORANGE   = "#e76f51"
YELLOW   = "#e9c46a"
GRID_COL = "#2a2a2a"
TEXT_COL = "#cccccc"


def _dark_rc():
    plt.rcParams.update({
        "figure.facecolor": DARK_BG, "axes.facecolor": CARD_BG,
        "axes.edgecolor": GRID_COL, "axes.labelcolor": TEXT_COL,
        "xtick.color": TEXT_COL, "ytick.color": TEXT_COL,
        "text.color": TEXT_COL, "grid.color": GRID_COL,
        "legend.facecolor": CARD_BG, "legend.edgecolor": GRID_COL,
        "font.family": "monospace", "axes.titlecolor": TEAL,
        "axes.titlesize": 12, "axes.labelsize": 10,
    })


def _show(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=DARK_BG, edgecolor="none")
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE FUNCTIONS (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_preprocess(file_bytes: bytes):
    """Load CSV and run Yeo-Johnson + LASSO+RFECV."""
    from io import BytesIO
    df = pd.read_csv(BytesIO(file_bytes), sep=",")
    TARGET = "status"
    y = df[TARGET].values.astype(int)
    X = df.drop(columns=[TARGET]).select_dtypes(include=[np.number])
    X = X.fillna(X.median())
    feature_names = list(X.columns)

    # Split first (preprocessing fitted on train only)
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X.values, y, test_size=0.20, stratify=y, random_state=42)

    # Yeo-Johnson
    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    X_tr_yj = pt.fit_transform(X_tr_raw)
    X_te_yj = pt.transform(X_te_raw)

    # Variance filter
    vt = VarianceThreshold(threshold=0.01)
    X_tr_vt = vt.fit_transform(X_tr_yj)
    X_te_vt = vt.transform(X_te_yj)

    # LassoCV screening
    lasso = LassoCV(cv=10, max_iter=5000, random_state=42, n_jobs=None)
    lasso.fit(X_tr_vt, y_tr)
    lmask = np.abs(lasso.coef_) > 0
    if lmask.sum() < 5:
        lmask = np.abs(lasso.coef_) > np.percentile(np.abs(lasso.coef_), 80)
    X_tr_l = X_tr_vt[:, lmask]
    X_te_l = X_te_vt[:, lmask]

    # RFECV
    svm_lin = SVC(kernel="linear", random_state=42)
    rfecv = RFECV(
        estimator=svm_lin,
        step=max(1, X_tr_l.shape[1] // 20),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="roc_auc",
        min_features_to_select=10,
        n_jobs=None,
    )
    rfecv.fit(X_tr_l, y_tr)
    X_tr_sel = rfecv.transform(X_tr_l)
    X_te_sel = rfecv.transform(X_te_l)

    # Rebuild selected feature names
    vt_names    = [feature_names[i] for i, s in enumerate(vt.get_support()) if s]
    lasso_names = [vt_names[i] for i, s in enumerate(lmask) if s]
    sel_names   = [lasso_names[i] for i, s in enumerate(rfecv.support_) if s]

    skew_before = pd.DataFrame(X_tr_raw).skew().abs().median()
    skew_after  = pd.DataFrame(X_tr_yj).skew().abs().median()

    return {
        "df": df, "y_tr": y_tr, "y_te": y_te,
        "X_tr_sel": X_tr_sel, "X_te_sel": X_te_sel,
        "sel_names": sel_names,
        "n_raw": X.shape[1], "n_sel": X_tr_sel.shape[1],
        "skew_before": skew_before, "skew_after": skew_after,
        "power_tf": pt, "vt": vt, "lasso_mask": lmask,
        "rfecv": rfecv,
    }


@st.cache_resource(show_spinner=False)
def run_optuna_and_train(cache_key: str, X_tr_json: str, y_tr_json: str,
                         model_type: str, n_trials: int):
    """Optuna search + final calibrated model."""
    X_tr = np.array(pd.read_json(X_tr_json))
    y_tr = np.array(pd.read_json(y_tr_json, typ="series"))

    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    trial_history = []

    def objective(trial):
        if model_type == "svm_rbf":
            C     = trial.suggest_float("C",     1e-3, 1e3, log=True)
            gamma = trial.suggest_float("gamma", 1e-4, 10,  log=True)
            m = SVC(C=C, gamma=gamma, kernel="rbf",
                    probability=True, random_state=42)
        else:
            lr = trial.suggest_float("learning_rate_init", 1e-5, 1e-2, log=True)
            alpha = trial.suggest_float("alpha", 1e-5, 1e-2, log=True)
            h = trial.suggest_categorical("hidden_layer_sizes",
                                          [(256,128,64),(128,64,32),(256,128),(512,256,128,64)])
            m = MLPClassifier(hidden_layer_sizes=h, activation="relu",
                              learning_rate_init=lr, alpha=alpha,
                              max_iter=500, early_stopping=True,
                              validation_fraction=0.1, random_state=42)
        sc = cross_val_score(m, X_tr, y_tr, cv=cv5, scoring="roc_auc", n_jobs=None).mean()
        trial_history.append(sc)
        return sc

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.HyperbandPruner(),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params

    if model_type == "svm_rbf":
        base = SVC(C=bp["C"], gamma=bp["gamma"], kernel="rbf",
                   probability=True, random_state=42)
    else:
        h = bp.get("hidden_layer_sizes", (256,128,64))
        if isinstance(h, str): h = eval(h)
        base = MLPClassifier(hidden_layer_sizes=h, activation="relu",
                             learning_rate_init=bp.get("learning_rate_init", 1e-3),
                             alpha=bp.get("alpha", 1e-4),
                             max_iter=500, early_stopping=True,
                             validation_fraction=0.1, random_state=42)

    cal_model = CalibratedClassifierCV(base, cv=5, method="sigmoid")
    cal_model.fit(X_tr, y_tr)
    return {"model": cal_model, "best_params": bp,
            "best_cv_auc": study.best_value, "trial_history": trial_history}


def compute_shap(model, X_train, X_test, sel_names):
    try:
        background  = shap.kmeans(X_train, min(50, X_train.shape[0]))
        explainer   = shap.KernelExplainer(model.predict_proba, background)
        n           = min(60, X_test.shape[0])
        sv          = explainer.shap_values(X_test[:n], nsamples=100)
        if isinstance(sv, list):
            sv1 = sv[1]
        elif len(sv.shape) == 3:
            sv1 = sv[:, :, 1]
        else:
            sv1 = sv
        bv          = explainer.expected_value
        if isinstance(bv, (list, np.ndarray)): bv = bv[1]
        return {"explainer": explainer, "sv": sv1, "bv": bv,
                "X_shap": X_test[:n], "names": sel_names}
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def _class_dist(y):
    _dark_rc()
    counts = pd.Series(y).value_counts()
    labels = ["Healthy (0)", "Parkinson's (1)"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].bar(labels, [counts.get(0,0), counts.get(1,0)],
                color=[TEAL, ORANGE], edgecolor=DARK_BG, width=0.5)
    axes[0].set_title("Class Distribution"); axes[0].grid(axis="y", alpha=0.3)
    axes[0].spines[["top","right"]].set_visible(False)
    for i, v in enumerate([counts.get(0,0), counts.get(1,0)]):
        axes[0].text(i, v+1, str(v), ha="center", fontweight="bold")
    wedges, texts, autotexts = axes[1].pie(
        [counts.get(0,0), counts.get(1,0)], labels=labels,
        colors=[TEAL, ORANGE], autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": DARK_BG, "linewidth": 2},
        textprops={"color": TEXT_COL})
    for at in autotexts: at.set_color(DARK_BG); at.set_fontweight("bold")
    axes[1].set_title("Class Proportions")
    fig.tight_layout(); return fig


def _skewness_bars(sk_before, sk_after):
    _dark_rc()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["Before\nYeo-Johnson", "After\nYeo-Johnson"],
           [sk_before, sk_after], color=[ORANGE, TEAL],
           edgecolor=DARK_BG, width=0.4)
    ax.set_title("Median |Skewness| of Features"); ax.set_ylabel("Median |Skewness|")
    ax.grid(axis="y", alpha=0.3); ax.spines[["top","right"]].set_visible(False)
    for i, v in enumerate([sk_before, sk_after]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")
    fig.tight_layout(); return fig


def _optuna_history(trial_history):
    _dark_rc()
    running_best = np.maximum.accumulate(trial_history)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(range(1, len(trial_history)+1), trial_history,
            color=GRID_COL, lw=1, alpha=0.6, label="Trial AUC")
    ax.plot(range(1, len(running_best)+1), running_best,
            color=TEAL, lw=2.5, marker="o", markersize=3,
            markerfacecolor=ORANGE, markeredgecolor=DARK_BG, label="Best AUC")
    ax.fill_between(range(1, len(running_best)+1), running_best, alpha=0.12, color=TEAL)
    ax.set_title("Optuna Bayesian Optimisation Convergence")
    ax.set_xlabel("Trial"); ax.set_ylabel("CV AUC-ROC")
    ax.legend(); ax.grid(alpha=0.3); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); return fig


def _confusion_matrix(cm):
    _dark_rc()
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "teal_dark", [CARD_BG, TEAL], N=256)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap=cmap)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Healthy","Parkinson's"]); ax.set_yticklabels(["Healthy","Parkinson's"])
    ax.set_xlabel("Predicted Label"); ax.set_ylabel("Actual Label")
    ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    fontsize=20, fontweight="bold",
                    color=DARK_BG if cm[i,j] > cm.max()/2 else TEXT_COL)
    fig.tight_layout(); return fig


def _roc(fpr, tpr, auc_sc, label="Model"):
    _dark_rc()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color=TEAL, lw=3, label=f"{label}  AUC = {auc_sc:.4f}")
    ax.fill_between(fpr, tpr, alpha=0.10, color=TEAL)
    ax.plot([0,1],[0,1], "--", color=ORANGE, lw=1.5, label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve"); ax.legend(loc="lower right")
    ax.grid(alpha=0.3); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); return fig


def _calibration(y_test, probs):
    _dark_rc()
    pt, pp = calibration_curve(y_test, probs, n_bins=10)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(pp, pt, "s-", color=TEAL, lw=2, label="Model")
    ax.plot([0,1],[0,1], "--", color=ORANGE, lw=1.5, label="Perfect")
    ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve (Brier score = lower is better)")
    ax.legend(); ax.grid(alpha=0.3); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); return fig


def _prob_distribution(probs, y_test):
    _dark_rc()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(probs[y_test==0], bins=20, color=TEAL,   alpha=0.7,
            label="Healthy", edgecolor=DARK_BG)
    ax.hist(probs[y_test==1], bins=20, color=ORANGE, alpha=0.7,
            label="Parkinson's", edgecolor=DARK_BG)
    ax.axvline(0.5, color=YELLOW, ls="--", lw=1.5, label="Threshold 0.5")
    ax.set_title("Prediction Probability Distribution")
    ax.set_xlabel("P(Parkinson's)"); ax.set_ylabel("Count")
    ax.legend(); ax.grid(alpha=0.3); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); return fig


def _shap_summary(shap_data):
    fig, _ = plt.subplots(figsize=(10, 5))
    shap.summary_plot(shap_data["sv"], shap_data["X_shap"],
                      feature_names=shap_data["names"], show=False, plot_type="dot")
    plt.tight_layout(); return fig


def _shap_bar(shap_data):
    fig, _ = plt.subplots(figsize=(9, 4))
    shap.summary_plot(shap_data["sv"], shap_data["X_shap"],
                      feature_names=shap_data["names"], show=False,
                      plot_type="bar", max_display=15)
    plt.tight_layout(); return fig


def _shap_waterfall(shap_data, idx=0):
    try:
        expl = shap.Explanation(
            values=shap_data["sv"][idx],
            base_values=shap_data["bv"],
            data=shap_data["X_shap"][idx],
            feature_names=shap_data["names"],
        )
        fig, _ = plt.subplots(figsize=(10, 5))
        shap.waterfall_plot(expl, show=False, max_display=15)
        plt.tight_layout(); return fig
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Parkinson's Detector")
    st.markdown("---")
    st.markdown("**Pipeline (Report Chapter 5)**")
    st.markdown("""
    1. Yeo-Johnson Normalisation
    2. LASSO + RFECV Feature Selection
    3. SVM-RBF **or** Deep MLP
    4. Bayesian Optimisation (Optuna)
    5. Platt Calibration
    6. SHAP Interpretability
    """)
    st.markdown("---")
    st.markdown("### Upload Dataset")
    uploaded = st.file_uploader(
        "parkinsons.csv", type=["csv"],
        help="UCI Parkinson's dataset"
    )
    st.markdown("---")
    model_choice   = st.selectbox("Model", ["svm_rbf", "deep_mlp"],
                                  format_func=lambda x: "SVM-RBF" if x=="svm_rbf" else "Deep MLP")
    n_trials       = st.slider("Optuna Trials", 10, 100, 30, step=10)
    show_shap      = st.checkbox("Enable SHAP (slower)", value=True)
    st.markdown("---")
    st.caption("© 2025 · KIIT University · Subham Barman")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 🧠 Parkinson's Disease Detection")
st.markdown(
    "<p style='color:#888;font-size:0.9rem;letter-spacing:1px'>"
    "Explainable Precision Diagnostics · Yeo-Johnson · LASSO-RFECV · "
    "SVM-RBF / Deep MLP · Optuna · SHAP</p>",
    unsafe_allow_html=True,
)

if uploaded is None:
    st.info("👈  Upload **parkinsons.csv** in the sidebar to begin.", icon="📂")
    st.stop()

# ── Preprocessing ──────────────────────────────────────────────────────────────
file_bytes = uploaded.read()
with st.spinner("🔄 Preprocessing: Yeo-Johnson → LASSO → RFECV …"):
    data = load_and_preprocess(file_bytes)

tabs = st.tabs([
    "📊 Dataset Overview",
    "⚙️ Train Model",
    "📈 Results & Evaluation",
    "📊 SHAP Interpretability",
    "🔮 Predict",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 – DATASET OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("## Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples",     data["df"].shape[0])
    c2.metric("Raw Features",      data["n_raw"])
    c3.metric("After LASSO+RFECV", data["n_sel"])
    c4.metric("Reduction",         f"{100*(1-data['n_sel']/data['n_raw']):.0f}%")

    with st.expander("📄 Raw Data (first 10 rows)"):
        st.dataframe(data["df"].head(10), use_container_width=True)

    st.markdown("### Class Distribution")
    _show(_class_dist(np.concatenate([data["y_tr"], data["y_te"]])))

    st.markdown("### Yeo-Johnson Effect on Skewness")
    _show(_skewness_bars(data["skew_before"], data["skew_after"]))

    st.markdown("### Selected Features")
    sel_df = pd.DataFrame({"Feature Name": data["sel_names"],
                            "Index": range(len(data["sel_names"]))})
    st.dataframe(sel_df, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 – TRAIN
# ═════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("## Train the Model")
    st.markdown(
        f"Press **TRAIN** to run Optuna ({n_trials} trials) and fit a "
        f"calibrated **{'SVM-RBF' if model_choice=='svm_rbf' else 'Deep MLP'}**."
    )

    col_btn, col_note = st.columns([1, 3])
    train_btn = col_btn.button("🚀 TRAIN", use_container_width=True)

    if train_btn or "train_results" in st.session_state:
        if train_btn or "train_results" not in st.session_state:
            cache_key = f"{model_choice}_{n_trials}_{data['n_sel']}"
            with st.spinner("🔄 Optuna + model training … (first run ~1–2 min)"):
                tr = run_optuna_and_train(
                    cache_key,
                    pd.DataFrame(data["X_tr_sel"]).to_json(),
                    pd.Series(data["y_tr"]).to_json(),
                    model_choice, n_trials,
                )
                # compute test metrics
                probs = tr["model"].predict_proba(data["X_te_sel"])
                preds = tr["model"].predict(data["X_te_sel"])
                tr["probs"]     = probs
                tr["preds"]     = preds
                tr["acc"]       = accuracy_score(data["y_te"], preds)
                tr["auc_roc"]   = roc_auc_score(data["y_te"], probs[:,1])
                tr["precision"] = precision_score(data["y_te"], preds, zero_division=0)
                tr["recall"]    = recall_score(data["y_te"], preds, zero_division=0)
                tr["f1"]        = f1_score(data["y_te"], preds, zero_division=0)
                tr["brier"]     = brier_score_loss(data["y_te"], probs[:,1])
                tr["cm"]        = confusion_matrix(data["y_te"], preds)
                fpr_, tpr_, _   = roc_curve(data["y_te"], probs[:,1])
                tr["fpr"]       = fpr_
                tr["tpr"]       = tpr_
                tr["report"]    = classification_report(
                    data["y_te"], preds, output_dict=True)
                st.session_state["train_results"] = tr
                st.session_state["data"] = data

        tr = st.session_state["train_results"]
        st.success("✅ Training Complete!", icon="🎉")

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Accuracy",       f"{tr['acc']*100:.2f}%")
        m2.metric("AUC-ROC",        f"{tr['auc_roc']:.4f}")
        m3.metric("Precision (PD)", f"{tr['precision']:.4f}")
        m4.metric("Recall (PD)",    f"{tr['recall']:.4f}")
        m5.metric("F1 (PD)",        f"{tr['f1']:.4f}")
        m6.metric("Brier Score",    f"{tr['brier']:.4f}")

        st.markdown("### Optuna Convergence")
        _show(_optuna_history(tr["trial_history"]))
        st.caption(f"Best params: `{tr['best_params']}`  |  "
                   f"Best CV AUC: **{tr['best_cv_auc']:.4f}**")

        with st.expander("📋 Full Classification Report"):
            st.dataframe(
                pd.DataFrame(tr["report"]).transpose()
                  .style.format("{:.4f}").background_gradient(cmap="Blues"),
                use_container_width=True,
            )
    else:
        col_note.info("Press **TRAIN** to start.", icon="ℹ️")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 – RESULTS
# ═════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("## Results & Evaluation")

    if "train_results" not in st.session_state:
        st.warning("⚠️ Please train the model first.", icon="⚠️"); st.stop()

    tr   = st.session_state["train_results"]
    data = st.session_state.get("data", data)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### Confusion Matrix")
        _show(_confusion_matrix(tr["cm"]))
    with col_r:
        st.markdown("### ROC Curve")
        _show(_roc(tr["fpr"], tr["tpr"], tr["auc_roc"],
                   "SVM-RBF" if model_choice=="svm_rbf" else "Deep MLP"))

    st.markdown("### Calibration Curve (Platt Scaling)")
    _show(_calibration(data["y_te"], tr["probs"][:,1]))

    st.markdown("### Prediction Probability Distribution")
    _show(_prob_distribution(tr["probs"][:,1], data["y_te"]))

    st.markdown("### Performance Summary")
    summary = {
        "Metric": ["Accuracy","AUC-ROC","Precision (PD)","Recall (PD)",
                   "F1-Score (PD)","Brier Score","RFECV Features"],
        "Value":  [f"{tr['acc']*100:.2f}%", f"{tr['auc_roc']:.4f}",
                   f"{tr['precision']:.4f}", f"{tr['recall']:.4f}",
                   f"{tr['f1']:.4f}", f"{tr['brier']:.4f}", str(data["n_sel"])],
    }
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 – SHAP
# ═════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("## SHAP Interpretability")

    if "train_results" not in st.session_state:
        st.warning("⚠️ Please train the model first.", icon="⚠️"); st.stop()

    if not show_shap:
        st.info("Enable **SHAP** in the sidebar to see explanations.", icon="ℹ️")
        st.stop()

    tr   = st.session_state["train_results"]
    data = st.session_state.get("data", data)

    if "shap_data" not in st.session_state:
        with st.spinner("🔄 Computing SHAP values (KernelExplainer) …"):
            shap_data = compute_shap(
                tr["model"], data["X_tr_sel"], data["X_te_sel"], data["sel_names"])
            st.session_state["shap_data"] = shap_data

    sd = st.session_state["shap_data"]

    if "error" in sd:
        st.error(f"SHAP error: {sd['error']}"); st.stop()

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("**Beeswarm Summary (global)**")
        _show(_shap_summary(sd))
    with col_s2:
        st.markdown("**Mean |SHAP| Bar Plot**")
        _show(_shap_bar(sd))

    st.markdown("### Patient-level Waterfall")
    pat_idx = st.slider("Select test patient index", 0, len(sd["X_shap"])-1, 0)
    wf_fig  = _shap_waterfall(sd, pat_idx)
    if wf_fig:
        _show(wf_fig)
    else:
        st.warning("Waterfall plot unavailable for this patient.")

    st.markdown("### SHAP Feature Importance Table")
    imp_df = pd.DataFrame({
        "Feature": sd["names"],
        "Mean |SHAP|": np.abs(sd["sv"]).mean(axis=0),
    }).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)
    imp_df.index += 1
    st.dataframe(imp_df.head(20), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 – PREDICT
# ═════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("## 🔮 Predict on New Data")

    if "train_results" not in st.session_state:
        st.warning("⚠️ Please train the model first.", icon="⚠️"); st.stop()

    tr   = st.session_state["train_results"]
    data = st.session_state.get("data", data)

    mode = st.radio("Prediction mode",
                    ["✏️ Manual feature entry", "📥 Upload CSV"],
                    horizontal=True)

    if "✏️" in mode:
        st.markdown("Enter values for the **RFECV-selected features** (others default to 0).")
        n_show = min(10, len(data["sel_names"]))
        cols_inp = st.columns(min(n_show, 5))
        vals = {}
        for i in range(n_show):
            vals[i] = cols_inp[i % 5].number_input(
                data["sel_names"][i][:12], value=0.0, format="%.4f", key=f"v_{i}")

        if st.button("🔮 PREDICT"):
            # Build vector of zeros then fill in entered values
            vec = np.zeros((1, data["n_sel"]))
            for i, v in vals.items():
                vec[0, i] = v

            # We need to un-standardise, re-apply pipeline manually:
            # Because the selected features are already Yeo-Johnson + standardised,
            # the user ideally enters raw feature values and we pass through pipeline.
            # For simplicity we treat entered values as already-scaled RFECV features.
            pred  = tr["model"].predict(vec)[0]
            prob  = tr["model"].predict_proba(vec)[0, 1]

            st.markdown("---")
            if pred == 1:
                st.error(f"🔴 **Parkinson's Detected** · Confidence: **{prob*100:.1f}%**")
            else:
                st.success(f"🟢 **Healthy** · Confidence: **{(1-prob)*100:.1f}%**")

            # Gauge bar
            _dark_rc()
            fig_g, ax_g = plt.subplots(figsize=(7, 1.5))
            ax_g.barh(0, prob, height=0.5,
                      color=ORANGE if pred==1 else TEAL, edgecolor=DARK_BG)
            ax_g.barh(0, 1-prob, left=prob, height=0.5,
                      color=GRID_COL, edgecolor=DARK_BG)
            ax_g.axvline(0.5, color=YELLOW, ls="--", lw=1.5)
            ax_g.set_xlim(0,1); ax_g.set_yticks([])
            ax_g.set_xlabel("P(Parkinson's)")
            ax_g.set_title(f"Gauge: {prob*100:.1f}% Parkinson's", pad=8)
            ax_g.spines[["top","right","left"]].set_visible(False)
            fig_g.tight_layout()
            _show(fig_g)

    else:  # CSV upload
        pred_file = st.file_uploader(
            "Upload CSV (must have same raw feature columns as training data)",
            type=["csv"], key="pred_csv",
        )
        if pred_file:
            pred_df  = pd.read_csv(pred_file)
            has_true = "status" in pred_df.columns
            if has_true:
                y_true_new = pred_df["status"].values
                pred_df    = pred_df.drop(columns=["status"])

            st.info(f"Loaded {len(pred_df)} rows × {pred_df.shape[1]} columns.")

            try:
                # Run through same pipeline as training
                X_new_raw = pred_df.select_dtypes(include=[np.number]).fillna(0).values
                pt   = data["power_tf"]
                vt   = data["vt"]
                lmsk = data["lasso_mask"]
                rfcv = data["rfecv"]

                X_new_yj  = pt.transform(X_new_raw)
                X_new_vt  = vt.transform(X_new_yj)
                X_new_l   = X_new_vt[:, lmsk]
                X_new_sel = rfcv.transform(X_new_l)

                new_preds = tr["model"].predict(X_new_sel)
                new_probs = tr["model"].predict_proba(X_new_sel)[:, 1]

                out = pd.DataFrame({
                    "Prediction": ["Parkinson's 🔴" if p==1 else "Healthy 🟢"
                                   for p in new_preds],
                    "P(Parkinson's)": new_probs.round(4),
                })
                if has_true:
                    out["True Label"] = ["Parkinson's" if t==1 else "Healthy"
                                         for t in y_true_new]
                    out["Correct"] = new_preds == y_true_new

                st.dataframe(out, use_container_width=True)
                if has_true:
                    st.metric("Accuracy on uploaded file",
                              f"{(new_preds == y_true_new).mean()*100:.2f}%")
            except Exception as e:
                st.error(f"Prediction error: {e}\n\n"
                         "Ensure the CSV has the same feature columns as the training data.")
