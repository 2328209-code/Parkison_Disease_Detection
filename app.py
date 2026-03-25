# =============================================================================
# Parkinson's Disease Detection - Streamlit App
# Dark Theme + PSO + Hybrid Model (RF + XGBoost)
# =============================================================================
# Requirements:
#   pip install streamlit numpy pandas scikit-learn xgboost imbalanced-learn seaborn matplotlib
# Run:
#   streamlit run parkinsons_app.py
# =============================================================================

import os
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.metrics import (classification_report, roc_auc_score,
                             roc_curve, auc, confusion_matrix, accuracy_score)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & DARK THEME
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Parkinson's Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS for black/dark theme
st.markdown("""
<style>
/* ── Global dark background ── */
html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] {
    background-color: #0a0a0a !important;
    color: #e0e0e0 !important;
    font-family: 'Courier New', monospace !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #111111 !important;
    border-right: 1px solid #2a2a2a !important;
}
[data-testid="stSidebar"] * { color: #cccccc !important; }

/* ── Main content blocks ── */
[data-testid="stVerticalBlock"] > div { background: transparent !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #141414 !important;
    border: 1px solid #2a9d8f !important;
    border-radius: 8px !important;
    padding: 12px !important;
}
[data-testid="stMetricValue"] { color: #2a9d8f !important; font-size: 2rem !important; }
[data-testid="stMetricLabel"] { color: #888888 !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #2a9d8f, #1a7a6e) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: bold !important;
    letter-spacing: 1px !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #38c9b6, #2a9d8f) !important;
    box-shadow: 0 0 18px rgba(42,157,143,0.5) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 1px dashed #2a9d8f !important;
    border-radius: 8px !important;
    padding: 10px !important;
}

/* ── Sliders, select boxes ── */
.stSlider > div > div { background: #2a9d8f !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #111111 !important;
    border-bottom: 1px solid #2a2a2a !important;
}
.stTabs [data-baseweb="tab"] {
    color: #888888 !important;
    font-family: 'Courier New', monospace !important;
    letter-spacing: 1px !important;
}
.stTabs [aria-selected="true"] {
    color: #2a9d8f !important;
    border-bottom: 2px solid #2a9d8f !important;
}

/* ── DataFrame ── */
[data-testid="stDataFrame"] { border: 1px solid #2a2a2a !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #141414 !important;
    border: 1px solid #2a2a2a !important;
    color: #2a9d8f !important;
}

/* ── Success / error alerts ── */
[data-testid="stAlert"] { border-radius: 8px !important; }

/* ── Headers ── */
h1, h2, h3 { color: #2a9d8f !important; letter-spacing: 2px !important; }
h1 { border-bottom: 1px solid #2a2a2a; padding-bottom: 10px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB DARK THEME HELPER
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG  = "#0a0a0a"
CARD_BG  = "#141414"
TEAL     = "#2a9d8f"
TEAL2    = "#38c9b6"
ORANGE   = "#e76f51"
YELLOW   = "#e9c46a"
GRID_COL = "#2a2a2a"
TEXT_COL = "#cccccc"

def apply_dark_style():
    plt.rcParams.update({
        "figure.facecolor":  DARK_BG,
        "axes.facecolor":    CARD_BG,
        "axes.edgecolor":    GRID_COL,
        "axes.labelcolor":   TEXT_COL,
        "xtick.color":       TEXT_COL,
        "ytick.color":       TEXT_COL,
        "text.color":        TEXT_COL,
        "grid.color":        GRID_COL,
        "grid.alpha":        0.6,
        "legend.facecolor":  CARD_BG,
        "legend.edgecolor":  GRID_COL,
        "font.family":       "monospace",
        "axes.titlecolor":   TEAL,
        "axes.titlesize":    12,
        "axes.labelsize":    10,
    })


def fig_to_streamlit(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=DARK_BG, edgecolor="none")
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CACHED PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_preprocess(file_bytes: bytes):
    from io import BytesIO
    df = pd.read_csv(BytesIO(file_bytes), sep=",", skiprows=1)
    X = df.drop("class", axis=1)
    y = df["class"]

    # Variance filter
    vf = VarianceThreshold(threshold=0.01)
    X_var = pd.DataFrame(vf.fit_transform(X))

    # Correlation filter
    corr = X_var.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > 0.90)]
    X_filt = X_var.drop(columns=to_drop)

    # Mutual information
    mi = mutual_info_classif(X_filt, y)
    mi_df = pd.DataFrame({"Feature": X_filt.columns, "MI Score": mi})
    mi_df = mi_df.sort_values("MI Score", ascending=False)

    top80 = mi_df.head(80)["Feature"].values
    X_mi = X_filt[top80]

    return df, X_mi, y, mi_df


@st.cache_resource(show_spinner=False)
def run_pso_and_train(X_mi_json: str, y_json: str):
    X_mi = pd.read_json(X_mi_json)
    y    = pd.read_json(y_json, typ="series")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_mi, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # ── PSO ──────────────────────────────────────────────────
    n_p, n_it = 20, 25
    w, c1, c2 = 0.7, 1.5, 1.5
    nf = X_tr_s.shape[1]

    pos = np.random.randint(0, 2, (n_p, nf))
    vel = np.random.uniform(-1, 1, (n_p, nf))
    pb_pos = pos.copy()
    pb_sc  = np.zeros(n_p)
    gb_pos = None
    gb_sc  = 0.0
    pso_hist = []

    def fitness(p):
        sel = np.where(p == 1)[0]
        if len(sel) == 0:
            return 0
        m = RandomForestClassifier(n_estimators=50, random_state=42)
        s = cross_val_score(m, X_tr_s[:, sel], y_tr, cv=3).mean()
        return 0.99 * s - 0.01 * len(sel) / nf

    for it in range(n_it):
        for i in range(n_p):
            sc = fitness(pos[i])
            if sc > pb_sc[i]:
                pb_sc[i]  = sc
                pb_pos[i] = pos[i].copy()
            if sc > gb_sc:
                gb_sc  = sc
                gb_pos = pos[i].copy()
        for i in range(n_p):
            r1 = np.random.rand(nf)
            r2 = np.random.rand(nf)
            vel[i] = (w * vel[i]
                      + c1 * r1 * (pb_pos[i] - pos[i])
                      + c2 * r2 * (gb_pos   - pos[i]))
            sig = 1 / (1 + np.exp(-vel[i]))
            pos[i] = np.where(np.random.rand(nf) < sig, 1, 0)
        pso_hist.append(gb_sc)

    best_idx = np.where(gb_pos == 1)[0]
    X_tr_pso = X_tr_s[:, best_idx]
    X_te_pso = X_te_s[:, best_idx]

    # ── Hybrid RF + XGBoost ────────────────────────────────────
    rf  = RandomForestClassifier(n_estimators=200, random_state=42)
    xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                        use_label_encoder=False, eval_metric="logloss",
                        random_state=42)
    hybrid = VotingClassifier([("rf", rf), ("xgb", xgb)], voting="soft")
    hybrid.fit(X_tr_pso, y_tr)

    y_pred = hybrid.predict(X_te_pso)
    y_prob = hybrid.predict_proba(X_te_pso)[:, 1]
    acc    = accuracy_score(y_te, y_pred)
    roc_sc = roc_auc_score(y_te, y_prob)
    cm     = confusion_matrix(y_te, y_pred)
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    report = classification_report(y_te, y_pred, output_dict=True)

    return {
        "hybrid":    hybrid,
        "scaler":    scaler,
        "best_idx":  best_idx,
        "acc":       acc,
        "roc":       roc_sc,
        "cm":        cm,
        "fpr":       fpr,
        "tpr":       tpr,
        "pso_hist":  pso_hist,
        "report":    report,
        "n_pso_feat": len(best_idx),
        "y_test":    y_te,
        "y_pred":    y_pred,
        "y_prob":    y_prob,
        "X_test_pso": X_te_pso,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CHART FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def plot_mi_scores(mi_df):
    apply_dark_style()
    top = mi_df.head(20)
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(range(len(top)), top["MI Score"],
                  color=[TEAL if i % 2 == 0 else TEAL2 for i in range(len(top))],
                  edgecolor=DARK_BG, linewidth=0.5)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels([f"F{f}" for f in top["Feature"]], rotation=45, ha="right", fontsize=8)
    ax.set_title("Top 20 Mutual Information Scores", fontsize=13, pad=12)
    ax.set_ylabel("MI Score")
    ax.set_xlabel("Feature Index")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_class_distribution(y):
    apply_dark_style()
    counts = y.value_counts()
    labels = ["Healthy (0)", "Parkinson's (1)"]
    colors = [TEAL, ORANGE]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # Bar
    axes[0].bar(labels, counts.values, color=colors, edgecolor=DARK_BG, linewidth=1.5, width=0.5)
    axes[0].set_title("Class Distribution", pad=10)
    axes[0].set_ylabel("Count")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].spines[["top","right"]].set_visible(False)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 2, str(v), ha="center", color=TEXT_COL, fontsize=11, fontweight="bold")

    # Pie
    wedges, texts, autotexts = axes[1].pie(
        counts.values, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": DARK_BG, "linewidth": 2},
        textprops={"color": TEXT_COL}
    )
    for at in autotexts:
        at.set_color(DARK_BG); at.set_fontweight("bold")
    axes[1].set_title("Class Proportions", pad=10)

    fig.tight_layout()
    return fig


def plot_pso_convergence(pso_hist):
    apply_dark_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(pso_hist)+1), pso_hist,
            color=TEAL, linewidth=2.5, marker="o", markersize=4,
            markerfacecolor=ORANGE, markeredgecolor=DARK_BG)
    ax.fill_between(range(1, len(pso_hist)+1), pso_hist,
                    alpha=0.15, color=TEAL)
    ax.set_title("PSO Convergence – Global Best Fitness", pad=12)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness Score")
    ax.grid(linestyle="--", alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm):
    apply_dark_style()
    fig, ax = plt.subplots(figsize=(5, 4.5))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "teal_dark", [CARD_BG, TEAL], N=256)
    im = ax.imshow(cm, cmap=cmap)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Healthy","Parkinson's"], fontsize=10)
    ax.set_yticklabels(["Healthy","Parkinson's"], fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("Actual Label", fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=13, pad=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=20, fontweight="bold",
                    color=DARK_BG if cm[i,j] > cm.max()/2 else TEXT_COL)
    fig.tight_layout()
    return fig


def plot_roc_curve(fpr, tpr, roc_score):
    apply_dark_style()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color=TEAL, linewidth=3,
            label=f"Hybrid Model  AUC = {roc_score:.4f}")
    ax.fill_between(fpr, tpr, alpha=0.10, color=TEAL)
    ax.plot([0,1],[0,1], linestyle="--", color=ORANGE, linewidth=1.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve", fontsize=13, pad=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_feature_importance(hybrid_model, n_pso_feat):
    apply_dark_style()
    rf_imp = hybrid_model.estimators_[0].feature_importances_
    top_n  = min(20, len(rf_imp))
    idx    = np.argsort(rf_imp)[-top_n:][::-1]
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [TEAL if i % 2 == 0 else TEAL2 for i in range(top_n)]
    ax.barh(range(top_n), rf_imp[idx][::-1], color=colors[::-1],
            edgecolor=DARK_BG, linewidth=0.5)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([f"PSO-Feature {i}" for i in idx[::-1]], fontsize=8)
    ax.set_title(f"Top {top_n} RF Feature Importances (PSO-selected)", pad=12)
    ax.set_xlabel("Importance")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_probability_distribution(y_prob, y_test):
    apply_dark_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    healthy    = y_prob[y_test == 0]
    parkinsons = y_prob[y_test == 1]
    ax.hist(healthy,    bins=20, color=TEAL,   alpha=0.7, label="Healthy",       edgecolor=DARK_BG)
    ax.hist(parkinsons, bins=20, color=ORANGE, alpha=0.7, label="Parkinson's",   edgecolor=DARK_BG)
    ax.axvline(0.5, color=YELLOW, linestyle="--", linewidth=1.5, label="Threshold 0.5")
    ax.set_title("Prediction Probability Distribution", pad=12)
    ax.set_xlabel("Predicted Probability (Parkinson's)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Parkinson's Detector")
    st.markdown("---")
    st.markdown("**Model:** RF + XGBoost (Soft Voting)")
    st.markdown("**Feature Selection:** PSO + MI")
    st.markdown("---")
    st.markdown("### Upload Dataset")
    uploaded = st.file_uploader(
        "pd_speech_features.csv", type=["csv"],
        help="Upload the Parkinson's speech features CSV."
    )
    st.markdown("---")
    st.markdown("### Manual Prediction")
    st.caption("After training, enter values below:")
    n_manual = st.number_input("Number of features to enter", min_value=1, max_value=10, value=5)
    st.markdown("---")
    st.caption("© 2025 Parkinson's AI Lab")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 🧠 Parkinson's Disease Detection System")
st.markdown(
    "<p style='color:#888;font-size:0.9rem;letter-spacing:1px'>"
    "Hybrid Machine Learning Pipeline · PSO Feature Selection · RF + XGBoost Ensemble"
    "</p>", unsafe_allow_html=True
)

if uploaded is None:
    st.info("👈  Upload **pd_speech_features.csv** in the sidebar to begin.", icon="📂")
    st.stop()

# ── Load & preprocess ─────────────────────────────────────────────────────────
file_bytes = uploaded.read()
with st.spinner("🔄 Loading and preprocessing dataset…"):
    df, X_mi, y, mi_df = load_and_preprocess(file_bytes)

tabs = st.tabs([
    "📊 Dataset Overview",
    "⚙️ Train Model",
    "📈 Results & Charts",
    "🔮 Predict",
])

# ═════════════════════════════════════════════════
# TAB 1 – DATASET OVERVIEW
# ═════════════════════════════════════════════════
with tabs[0]:
    st.markdown("## Dataset Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Samples",   df.shape[0])
    c2.metric("Raw Features",    df.shape[1] - 1)
    c3.metric("Post-MI Features", X_mi.shape[1])

    with st.expander("📄 Raw Data (first 10 rows)"):
        st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### Class Distribution")
    fig_cls = plot_class_distribution(y)
    fig_to_streamlit(fig_cls)

    st.markdown("### Mutual Information Feature Scores")
    fig_mi = plot_mi_scores(mi_df)
    fig_to_streamlit(fig_mi)

    st.markdown("### Top 20 Features by MI")
    st.dataframe(
        mi_df.head(20).reset_index(drop=True).style
            .background_gradient(subset=["MI Score"], cmap="Blues"),
        use_container_width=True
    )


# ═════════════════════════════════════════════════
# TAB 2 – TRAIN MODEL
# ═════════════════════════════════════════════════
with tabs[1]:
    st.markdown("## Train the Hybrid Model")
    st.markdown(
        "Click **Train** to run PSO feature selection followed by "
        "the RF + XGBoost ensemble. "
        "Training is cached – re-running is instant unless you change the dataset."
    )

    col_btn, col_info = st.columns([1, 3])
    train_btn = col_btn.button("🚀  TRAIN", use_container_width=True)

    if train_btn or "results" in st.session_state:
        if train_btn or "results" not in st.session_state:
            with st.spinner("🔄 Running PSO + Hybrid Model training (may take ~60s on first run)…"):
                results = run_pso_and_train(
                    X_mi.to_json(), y.to_json()
                )
                st.session_state["results"] = results

        r = st.session_state["results"]
        st.success("✅ Training Complete!", icon="🎉")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy",      f"{r['acc']*100:.2f}%")
        m2.metric("AUC Score",     f"{r['roc']:.4f}")
        m3.metric("PSO Features",  r["n_pso_feat"])
        m4.metric("Precision (1)", f"{r['report']['1']['precision']:.4f}")

        st.markdown("### PSO Convergence")
        fig_pso = plot_pso_convergence(r["pso_hist"])
        fig_to_streamlit(fig_pso)

        with st.expander("📋 Full Classification Report"):
            report_df = pd.DataFrame(r["report"]).transpose()
            st.dataframe(
                report_df.style.format("{:.4f}").background_gradient(cmap="Blues"),
                use_container_width=True
            )
    else:
        col_info.info("Press **TRAIN** to start the pipeline.", icon="ℹ️")


# ═════════════════════════════════════════════════
# TAB 3 – RESULTS & CHARTS
# ═════════════════════════════════════════════════
with tabs[2]:
    st.markdown("## Model Results & Visualizations")

    if "results" not in st.session_state:
        st.warning("⚠️ Please train the model in **Train Model** tab first.", icon="⚠️")
        st.stop()

    r = st.session_state["results"]

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### Confusion Matrix")
        fig_cm = plot_confusion_matrix(r["cm"])
        fig_to_streamlit(fig_cm)

    with col_r:
        st.markdown("### ROC Curve")
        fig_roc = plot_roc_curve(r["fpr"], r["tpr"], r["roc"])
        fig_to_streamlit(fig_roc)

    st.markdown("### Prediction Probability Distribution")
    fig_prob = plot_probability_distribution(r["y_prob"], r["y_test"].values)
    fig_to_streamlit(fig_prob)

    st.markdown("### Feature Importance (Random Forest component)")
    fig_fi = plot_feature_importance(r["hybrid"], r["n_pso_feat"])
    fig_to_streamlit(fig_fi)

    # Summary stats
    st.markdown("### Performance Summary")
    summary_data = {
        "Metric": ["Accuracy", "AUC-ROC", "Precision (Parkinson's)",
                   "Recall (Parkinson's)", "F1-Score (Parkinson's)",
                   "PSO Selected Features"],
        "Value":  [
            f"{r['acc']*100:.2f}%",
            f"{r['roc']:.4f}",
            f"{r['report']['1']['precision']:.4f}",
            f"{r['report']['1']['recall']:.4f}",
            f"{r['report']['1']['f1-score']:.4f}",
            str(r["n_pso_feat"]),
        ]
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════
# TAB 4 – PREDICT
# ═════════════════════════════════════════════════
with tabs[3]:
    st.markdown("## 🔮 Predict on New Data")

    if "results" not in st.session_state:
        st.warning("⚠️ Please train the model in **Train Model** tab first.", icon="⚠️")
        st.stop()

    r = st.session_state["results"]

    predict_mode = st.radio(
        "Choose prediction mode",
        ["📥 Upload CSV of new samples", "✏️ Enter feature values manually"],
        horizontal=True
    )

    # ── CSV upload ───────────────────────────────────────────
    if "📥" in predict_mode:
        pred_file = st.file_uploader("Upload CSV (no 'class' column needed)", type=["csv"], key="pred_csv")
        if pred_file:
            pred_df = pd.read_csv(pred_file)
            st.write("Uploaded shape:", pred_df.shape)

            try:
                # Step 1: drop non-numeric and known label/id columns
                cols_to_drop = [c for c in pred_df.columns
                                if c.lower() in ("id", "name", "class", "label", "target")
                                or not pd.api.types.is_numeric_dtype(pred_df[c])]
                X_clean = pred_df.drop(columns=cols_to_drop, errors="ignore").astype(float)

                n_scaler = r["scaler"].n_features_in_
                n_cols   = X_clean.shape[1]

                # Step 2: align column count to what scaler expects
                if n_cols >= n_scaler:
                    X_aligned = X_clean.iloc[:, :n_scaler].values
                else:
                    pad = np.zeros((len(X_clean), n_scaler - n_cols))
                    X_aligned = np.hstack([X_clean.values, pad])

                # Step 3: scale then select PSO features
                X_scaled  = r["scaler"].transform(X_aligned.astype(np.float64))
                best_idx  = r["best_idx"]
                valid_idx = best_idx[best_idx < X_scaled.shape[1]]
                X_pso     = X_scaled[:, valid_idx]

                preds = r["hybrid"].predict(X_pso)
                probs = r["hybrid"].predict_proba(X_pso)[:, 1]

                result_df = pred_df.copy()
                result_df["Prediction"] = ["Parkinson's 🔴" if p == 1 else "Healthy 🟢" for p in preds]
                result_df["Probability (Parkinson's)"] = probs.round(4)

                st.success(f"Predictions complete for {len(preds)} sample(s).")
                if cols_to_drop:
                    st.info(f"Dropped non-numeric/label columns before prediction: {cols_to_drop}")

                st.dataframe(
                    result_df[["Prediction", "Probability (Parkinson's)"]],
                    use_container_width=True
                )

                n_park = int((preds == 1).sum())
                n_heal = int((preds == 0).sum())
                ca, cb = st.columns(2)
                ca.metric("Parkinson's 🔴", n_park)
                cb.metric("Healthy 🟢",     n_heal)

            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.caption("Make sure the CSV has numeric feature columns matching the training data.")

    # ── Manual input ─────────────────────────────────────────
    else:
        st.markdown(
            "<p style='color:#888;font-size:0.85rem'>"
            "Enter values for the PSO-selected feature indices. "
            "Unknown features default to 0.</p>",
            unsafe_allow_html=True
        )

        best_idx = r["best_idx"]
        n_show   = min(n_manual, len(best_idx))
        feature_vals = {}

        cols = st.columns(min(n_show, 4))
        for i, feat_idx in enumerate(best_idx[:n_show]):
            col = cols[i % len(cols)]
            feature_vals[feat_idx] = col.number_input(
                f"Feature {feat_idx}",
                value=0.0, format="%.4f", key=f"feat_{i}"
            )

        if st.button("🔮  PREDICT", use_container_width=False):
            # Build full feature vector (zeros for unseen features)
            full_vec = np.zeros(r["scaler"].n_features_in_)
            # fill in only the PSO-selected ones (use fi as the column index, not i)
            for i, fi in enumerate(best_idx):
                if fi in feature_vals and fi < r["scaler"].n_features_in_:
                    full_vec[fi] = feature_vals[fi]

            vec_scaled = r["scaler"].transform(full_vec.reshape(1, -1))
            # Pick only PSO-selected features (guard against out-of-range indices)
            valid_idx = best_idx[best_idx < vec_scaled.shape[1]]
            vec_pso   = vec_scaled[:, valid_idx]

            pred  = r["hybrid"].predict(vec_pso)[0]
            prob  = r["hybrid"].predict_proba(vec_pso)[0, 1]

            st.markdown("---")
            if pred == 1:
                st.error(f"🔴 **Result: Parkinson's Detected**\n\nConfidence: **{prob*100:.2f}%**")
            else:
                st.success(f"🟢 **Result: Healthy**\n\nConfidence: **{(1-prob)*100:.2f}%**")

            # Gauge-style bar
            apply_dark_style()
            fig_g, ax_g = plt.subplots(figsize=(7, 1.5))
            ax_g.barh(0, prob, color=ORANGE if pred == 1 else TEAL,
                      height=0.5, edgecolor=DARK_BG)
            ax_g.barh(0, 1-prob, left=prob, color=GRID_COL,
                      height=0.5, edgecolor=DARK_BG)
            ax_g.axvline(0.5, color=YELLOW, linestyle="--", linewidth=1.5)
            ax_g.set_xlim(0, 1)
            ax_g.set_yticks([])
            ax_g.set_xlabel("Probability of Parkinson's")
            ax_g.set_title(f"Prediction Gauge  ({prob*100:.1f}% Parkinson's)", pad=8)
            ax_g.spines[["top","right","left"]].set_visible(False)
            fig_g.tight_layout()
            fig_to_streamlit(fig_g)
