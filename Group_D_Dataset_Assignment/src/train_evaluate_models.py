"""
Assignment 3B — Model Implementation & Controlled Factorial Evaluation (train_evaluate_models.py)
Group D — STINK3114 Natural Language Processing (UUM)

Implements Group D's unified, lightweight, and fully reproducible sentiment classification
pipeline using TF-IDF feature representation and Multinomial Logistic Regression across
the 2x2 factorial experimental matrix:

    Experiment | Emoji Handling (Track)               | Class Weighting
    -----------|--------------------------------------|---------------------------
    E1         | Removed  (cleaned_text_no_emojis)    | None (Standard CE)
    E2         | Retained (emojis -> text tokens)     | None (Standard CE)
    E3         | Removed  (cleaned_text_no_emojis)    | class_weight="balanced"
    E4         | Retained (emojis -> text tokens)     | class_weight="balanced"

Uses the locked 70% Train (1,198) / 15% Validation (257) / 15% Test (257) stratified splits
produced in Assignment 3A (`random_state=42`). Hyperparameters (`ngram_range`, `C`) are
selected strictly on the Validation split (zero test-set leakage), then locked identically
across E1–E4 for final evaluation on the held-out Test split.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

RANDOM_STATE = 42
CLASS_ORDER = ["Positive", "Neutral", "Negative"]

# Add src/ to path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent))
from emoji_tokenizer import convert_emojis_to_tokens

EXPERIMENTS = [
    {
        "exp_id": "E1",
        "name": "E1_EmojiRemoved_NoWeight",
        "short_label": "E1: Removed + None\n(Baseline)",
        "emoji_handling": "Removed",
        "class_weight_label": "None",
        "class_weight": None,
        "text_column": "cleaned_text_no_emojis",
        "convert_emojis": False,
        "model_filename": "E1_emoji_removed_none.joblib",
        "ro_alignment": "Baseline (RO1/RO2/RO3 reference)",
    },
    {
        "exp_id": "E2",
        "name": "E2_EmojiRetained_NoWeight",
        "short_label": "E2: Retained + None\n(RO1)",
        "emoji_handling": "Retained (Text Tokens)",
        "class_weight_label": "None",
        "class_weight": None,
        "text_column": "cleaned_text_with_emojis",
        "convert_emojis": True,
        "model_filename": "E2_emoji_retained_none.joblib",
        "ro_alignment": "RO1 (Emoji Retention Effect)",
    },
    {
        "exp_id": "E3",
        "name": "E3_EmojiRemoved_Balanced",
        "short_label": "E3: Removed + Balanced\n(RO2)",
        "emoji_handling": "Removed",
        "class_weight_label": "Balanced",
        "class_weight": "balanced",
        "text_column": "cleaned_text_no_emojis",
        "convert_emojis": False,
        "model_filename": "E3_emoji_removed_balanced.joblib",
        "ro_alignment": "RO2 (Class-Imbalance Mitigation)",
    },
    {
        "exp_id": "E4",
        "name": "E4_EmojiRetained_Balanced",
        "short_label": "E4: Retained + Balanced\n(RO3 Combined)",
        "emoji_handling": "Retained (Text Tokens)",
        "class_weight_label": "Balanced",
        "class_weight": "balanced",
        "text_column": "cleaned_text_with_emojis",
        "convert_emojis": True,
        "model_filename": "E4_emoji_retained_balanced.joblib",
        "ro_alignment": "RO3 (Combined Interaction)",
    },
]


def prepare_text_series(df: pd.DataFrame, column: str, convert_emojis: bool) -> pd.Series:
    """Return text series for either Track B (emoji removed) or Track A (emojis -> text tokens)."""
    series = df[column].fillna("").astype(str)
    if convert_emojis:
        return series.apply(convert_emojis_to_tokens)
    return series


def build_pipeline(ngram_range: Tuple[int, int], c_value: float, class_weight) -> Pipeline:
    """Construct the controlled TF-IDF + Logistic Regression scikit-learn pipeline."""
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=2,
        max_features=5000,
        sublinear_tf=True,
        lowercase=True,
        stop_words=None,  # Preserves Malay/English negation terms (tak, tidak, bukan, x, not, no)
        token_pattern=r"(?u)\b\w+\b",  # Retains 1-char Manglish negation token 'x' and emoji_* tokens
    )
    classifier = LogisticRegression(
        C=c_value,
        class_weight=class_weight,
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> Dict:
    """Compute primary (Macro-F1) and secondary metrics plus per-class report & confusion matrix."""
    macro_f1 = float(f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=CLASS_ORDER, average="weighted", zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    macro_prec = float(precision_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0))

    rep = classification_report(
        y_true, y_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0
    )
    per_class = {}
    for cls in CLASS_ORDER:
        per_class[cls] = {
            "precision": round(float(rep[cls]["precision"]), 4),
            "recall": round(float(rep[cls]["recall"]), 4),
            "f1_score": round(float(rep[cls]["f1-score"]), 4),
            "support": int(rep[cls]["support"]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER).tolist()

    return {
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def select_hyperparameters_on_validation(
    train_df: pd.DataFrame, val_df: pd.DataFrame
) -> Tuple[Tuple[int, int], float, pd.DataFrame]:
    """
    Strictly use the Validation split (n=257) to choose a single shared (ngram_range, C)
    configuration maximizing mean Validation Macro-F1 across E1–E4, keeping architecture
    hyperparameters completely fixed across all four experimental conditions.
    """
    ngram_candidates = [(1, 1), (1, 2)]
    c_candidates = [0.5, 1.0, 2.0, 3.0, 5.0]
    rows = []

    y_train = train_df["label"]
    y_val = val_df["label"]

    for ng in ngram_candidates:
        for c_val in c_candidates:
            exp_scores = {}
            for exp in EXPERIMENTS:
                X_tr = prepare_text_series(train_df, exp["text_column"], exp["convert_emojis"])
                X_va = prepare_text_series(val_df, exp["text_column"], exp["convert_emojis"])
                pipe = build_pipeline(ng, c_val, exp["class_weight"])
                pipe.fit(X_tr, y_train)
                preds_va = pipe.predict(X_va)
                exp_scores[exp["exp_id"]] = float(
                    f1_score(y_val, preds_va, labels=CLASS_ORDER, average="macro", zero_division=0)
                )
            mean_val_f1 = float(np.mean(list(exp_scores.values())))
            rows.append(
                {
                    "ngram_range": str(ng),
                    "C": c_val,
                    "E1_val_macro_f1": round(exp_scores["E1"], 4),
                    "E2_val_macro_f1": round(exp_scores["E2"], 4),
                    "E3_val_macro_f1": round(exp_scores["E3"], 4),
                    "E4_val_macro_f1": round(exp_scores["E4"], 4),
                    "mean_val_macro_f1": round(mean_val_f1, 4),
                }
            )

    tuning_df = pd.DataFrame(rows).sort_values("mean_val_macro_f1", ascending=False).reset_index(drop=True)
    best_row = tuning_df.iloc[0]
    best_ngram = eval(best_row["ngram_range"])
    best_c = float(best_row["C"])
    return best_ngram, best_c, tuning_df


def extract_top_features(pipe: Pipeline, top_n: int = 12) -> Dict:
    """Extract top positive-weight lexical and emoji features per sentiment class."""
    vec: TfidfVectorizer = pipe.named_steps["tfidf"]
    clf: LogisticRegression = pipe.named_steps["clf"]
    feature_names = np.array(vec.get_feature_names_out())
    classes = list(clf.classes_)

    top_all = {}
    top_emoji = {}
    emoji_mask = np.array([("emoji_" in f or "emoticon_" in f) for f in feature_names])

    for idx, cls in enumerate(classes):
        coefs = clf.coef_[idx]
        top_idx = np.argsort(coefs)[::-1][:top_n]
        top_all[cls] = [
            {"feature": str(feature_names[i]), "weight": round(float(coefs[i]), 4)}
            for i in top_idx
        ]
        if emoji_mask.any():
            em_indices = np.where(emoji_mask)[0]
            em_sorted = em_indices[np.argsort(coefs[em_indices])[::-1][:top_n]]
            top_emoji[cls] = [
                {"feature": str(feature_names[i]), "weight": round(float(coefs[i]), 4)}
                for i in em_sorted
            ]
    return {"top_overall_features": top_all, "top_emoji_features": top_emoji}


# ==============================================================================
# Publication-Quality Visualizations (300 DPI)
# ==============================================================================

def draw_pipeline_diagram(save_path: Path, best_ngram: Tuple[int, int], best_c: float) -> None:
    """Generate a clean publication-quality Methodology Pipeline Flowchart (matching Group D spec)."""
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def draw_box(cx, cy, w, h, title, subtitle, facecolor="#F0F5FA", edgecolor="#1F4E79", title_color="#0F2942"):
        box = patches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0.12,rounding_size=0.18",
            linewidth=1.5,
            edgecolor=edgecolor,
            facecolor=facecolor,
        )
        ax.add_patch(box)
        ax.text(cx, cy + h * 0.16, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=title_color)
        ax.text(cx, cy - h * 0.22, subtitle, ha="center", va="center", fontsize=8.4, color="#334155")

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>,head_width=0.28,head_length=0.42", color="#1F4E79", lw=1.6),
        )

    # Box 1: Input Dataset
    draw_box(
        5.0, 9.0, 8.8, 1.20,
        "1. Input Dataset (Assignment 3A Finalized Corpus)",
        "1,712 Malaysian YouTube Comments (Malay, Manglish, English)\n"
        "Locked Stratified Splits: 70% Train (1,198) | 15% Validation (257) | 15% Test (257)",
        facecolor="#EBF3FA", edgecolor="#1F4E79"
    )

    # Arrows branching to Track B & Track A
    draw_arrow(3.4, 8.28, 2.75, 7.68)
    draw_arrow(6.6, 8.28, 7.25, 7.68)

    # Box 2 Left: Emoji Removed (E1, E3)
    draw_box(
        2.65, 6.95, 4.25, 1.24,
        "2A. Emoji Removed (Track B: E1, E3)",
        "cleaned_text_no_emojis\nUnicode emojis & emoticons stripped; words & negations kept",
        facecolor="#F8FAFC", edgecolor="#475569", title_color="#1E293B"
    )

    # Box 2 Right: Emoji Retained (E2, E4)
    draw_box(
        7.35, 6.95, 4.25, 1.24,
        "2B. Emoji Retained (Track A: E2, E4)",
        "cleaned_text_with_emojis -> Emoji Text Tokens\n(e.g., U+1F602 -> emoji_face_with_tears_of_joy)",
        facecolor="#FEF9E7", edgecolor="#D97706", title_color="#92400E"
    )

    # Arrows converging to TF-IDF
    draw_arrow(2.75, 6.21, 3.8, 5.58)
    draw_arrow(7.25, 6.21, 6.2, 5.58)

    # Box 3: TF-IDF Feature Representation
    draw_box(
        5.0, 4.85, 8.8, 1.20,
        "3. TF-IDF Feature Representation (Fit on Training Split Only — Zero Leakage)",
        f"TfidfVectorizer(ngram_range={best_ngram}, min_df=2, max_features=5000, sublinear_tf=True, stop_words=None)\n"
        "Preserves Malay/English negation bigrams (tak sedap, x boleh, not bad) + emoji text tokens",
        facecolor="#EEF2FF", edgecolor="#4338CA", title_color="#312E81"
    )

    draw_arrow(5.0, 4.13, 5.0, 3.55)

    # Box 4: Logistic Regression Model
    draw_box(
        5.0, 2.82, 8.8, 1.20,
        f"4. Multinomial Logistic Regression Classifier (C={best_c}, solver='lbfgs', random_state=42)",
        "E1 & E2: class_weight=None (Standard Cross-Entropy)  |  E3 & E4: class_weight='balanced' (Neu=0.648, Pos=1.131, Neg=1.744)\n"
        "Identical architecture and regularization locked across all four conditions for controlled RO1–RO3 ablation",
        facecolor="#ECFDF5", edgecolor="#059669", title_color="#065F46"
    )

    draw_arrow(5.0, 2.10, 5.0, 1.52)

    # Box 5: Prediction & Evaluation
    draw_box(
        5.0, 0.80, 8.8, 1.15,
        "5. 3-Class Sentiment Prediction & Evaluation (Positive · Neutral · Negative)",
        "Primary Metric: Macro-F1  |  Secondary Metrics: Accuracy, Macro Precision, Macro Recall, Per-Class Recall & Confusion Matrices",
        facecolor="#FDF2F8", edgecolor="#BE185D", title_color="#831843"
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_experiment_comparison(results: Dict, save_path: Path) -> None:
    """Plot grouped bar chart comparing Test Macro-F1, Accuracy, Macro Recall, and Minority Negative Recall."""
    exp_ids = ["E1", "E2", "E3", "E4"]
    x_labels = [
        "E1\nRemoved + None\n(Baseline)",
        "E2\nRetained + None\n(RO1)",
        "E3\nRemoved + Bal.\n(RO2)",
        "E4\nRetained + Bal.\n(RO3 Combined)",
    ]

    macro_f1 = [results[e]["test_metrics"]["macro_f1"] for e in exp_ids]
    acc = [results[e]["test_metrics"]["accuracy"] for e in exp_ids]
    macro_rec = [results[e]["test_metrics"]["macro_recall"] for e in exp_ids]
    neg_rec = [results[e]["test_metrics"]["per_class"]["Negative"]["recall"] for e in exp_ids]

    x = np.arange(len(exp_ids))
    w = 0.20

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"]})
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=300)

    b1 = ax.bar(x - 1.5 * w, macro_f1, width=w, label="Macro-F1 (Primary Metric)", color="#1F4E79", edgecolor="#111", lw=0.7)
    b2 = ax.bar(x - 0.5 * w, acc, width=w, label="Overall Accuracy", color="#3B82F6", edgecolor="#111", lw=0.7)
    b3 = ax.bar(x + 0.5 * w, macro_rec, width=w, label="Macro Recall", color="#10B981", edgecolor="#111", lw=0.7)
    b4 = ax.bar(x + 1.5 * w, neg_rec, width=w, label="Minority Negative Recall", color="#EF4444", edgecolor="#111", lw=0.7)

    for bars in (b1, b2, b3, b4):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.012,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.8,
                fontweight="bold",
            )

    ax.set_title("Held-Out Test Set Performance Across Factorial Conditions E1–E4 (N_test = 257)", fontsize=11.5, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9.2, fontweight="bold")
    ax.set_ylabel("Score (0.0 – 1.0)", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 0.85)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.legend(loc="upper left", ncol=2, frameon=True, fontsize=8.8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(results: Dict, save_path: Path) -> None:
    """Plot 2x2 grid of held-out Test Confusion Matrices for E1, E2, E3, and E4."""
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.2), dpi=300)
    exp_titles = {
        "E1": "E1: Emoji Removed + No Weight (Baseline)",
        "E2": "E2: Emoji Retained + No Weight (RO1)",
        "E3": "E3: Emoji Removed + Balanced (RO2)",
        "E4": "E4: Emoji Retained + Balanced (RO3 Combined)",
    }
    for ax, exp_id in zip(axes.flat, ["E1", "E2", "E3", "E4"]):
        cm = np.array(results[exp_id]["test_metrics"]["confusion_matrix"])
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=115)
        ax.set_title(
            f"{exp_titles[exp_id]}\nMacro-F1 = {results[exp_id]['test_metrics']['macro_f1']:.4f} | Neg Recall = {results[exp_id]['test_metrics']['per_class']['Negative']['recall']:.3f}",
            fontsize=9.6,
            fontweight="bold",
            pad=8,
        )
        tick_marks = np.arange(len(CLASS_ORDER))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(CLASS_ORDER, fontsize=9)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(CLASS_ORDER, fontsize=9)
        ax.set_xlabel("Predicted Label", fontsize=9, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=9, fontweight="bold")

        thresh = cm.max() * 0.55
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                row_pct = cm[i, j] / max(1, cm[i].sum()) * 100.0
                ax.text(
                    j, i,
                    f"{cm[i, j]}\n({row_pct:.1f}%)",
                    ha="center", va="center",
                    fontsize=8.8,
                    fontweight="bold",
                    color="white" if cm[i, j] > thresh else "#111827",
                )

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_top_emoji_weights(top_emoji_features: Dict, save_path: Path) -> None:
    """Plot top learned Logistic Regression weights for emoji text tokens in E4 across classes."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), dpi=300)
    colors = {"Positive": "#16A34A", "Neutral": "#2563EB", "Negative": "#DC2626"}

    for ax, cls in zip(axes, CLASS_ORDER):
        items = top_emoji_features.get(cls, [])[:8][::-1]
        labels = [it["feature"].replace("emoji_", "").replace("_", " ")[:28] for it in items]
        weights = [it["weight"] for it in items]
        bars = ax.barh(labels, weights, color=colors[cls], edgecolor="#111", lw=0.6, height=0.65)
        ax.axvline(0, color="#333", lw=0.8)
        ax.set_xlim(0, max(weights) * 1.25 if weights else 1.0)
        ax.set_title(f"Top Emoji Weights: {cls} (E4)", fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("LR Coefficient Weight", fontsize=8.8, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        for bar, w in zip(bars, weights):
            offset = 0.03 if w >= 0 else -0.03
            ha = "left" if w >= 0 else "right"
            ax.text(w + offset, bar.get_y() + bar.get_height() / 2, f"{w:+.2f}", va="center", ha=ha, fontsize=7.8, fontweight="bold")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# Main Execution Pipeline
# ==============================================================================

def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(DATASET_DIR / "train.csv", encoding="utf-8")
    val_df = pd.read_csv(DATASET_DIR / "validation.csv", encoding="utf-8")
    test_df = pd.read_csv(DATASET_DIR / "test.csv", encoding="utf-8")

    print("=" * 78)
    print("GROUP D — MODEL IMPLEMENTATION & CONTROLLED FACTORIAL EVALUATION (E1–E4)")
    print(f"Splits Loaded: Train={len(train_df)}, Validation={len(val_df)}, Test={len(test_df)} (Total={len(train_df)+len(val_df)+len(test_df)})")
    print("=" * 78)

    # 1. Validation-only hyperparameter selection (shared across E1–E4)
    best_ngram, best_c, tuning_df = select_hyperparameters_on_validation(train_df, val_df)
    tuning_df.to_csv(RESULTS_DIR / "validation_tuning_log.csv", index=False, encoding="utf-8")
    print(f"[Validation Selection] Best shared config: ngram_range={best_ngram}, C={best_c} (Mean Val Macro-F1={tuning_df.iloc[0]['mean_val_macro_f1']:.4f})")

    # 2. Majority-Class Sanity Baseline (illustrates Problem 2.2 collapse)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(train_df), 1)), train_df["label"])
    dummy_test_preds = dummy.predict(np.zeros((len(test_df), 1)))
    dummy_metrics = evaluate_predictions(test_df["label"], dummy_test_preds)
    print(
        f"[Sanity Baseline] Majority-Class Dummy -> Test Acc={dummy_metrics['accuracy']:.4f}, "
        f"Macro-F1={dummy_metrics['macro_f1']:.4f}, Neg Recall={dummy_metrics['per_class']['Negative']['recall']:.4f}"
    )

    # 3. Train & Evaluate E1, E2, E3, E4
    all_results: Dict = {}
    comparison_rows: List[Dict] = []
    per_class_rows: List[Dict] = []
    test_preds_df = test_df[["id", "source_id", "cleaned_text_with_emojis", "cleaned_text_no_emojis", "has_emoji", "language", "label"]].copy()

    e4_features = {}

    for exp in EXPERIMENTS:
        exp_id = exp["exp_id"]
        X_tr = prepare_text_series(train_df, exp["text_column"], exp["convert_emojis"])
        X_va = prepare_text_series(val_df, exp["text_column"], exp["convert_emojis"])
        X_te = prepare_text_series(test_df, exp["text_column"], exp["convert_emojis"])

        pipe = build_pipeline(best_ngram, best_c, exp["class_weight"])
        pipe.fit(X_tr, train_df["label"])

        # Save serialized model pipeline
        model_path = MODELS_DIR / exp["model_filename"]
        joblib.dump(pipe, model_path)

        val_preds = pipe.predict(X_va)
        test_preds = pipe.predict(X_te)
        test_preds_df[f"{exp_id}_pred"] = test_preds

        val_metrics = evaluate_predictions(val_df["label"], val_preds)
        test_metrics = evaluate_predictions(test_df["label"], test_preds)

        vec = pipe.named_steps["tfidf"]
        vocab_size = len(vec.vocabulary_)
        emoji_vocab_count = sum(1 for tok in vec.vocabulary_ if ("emoji_" in tok or "emoticon_" in tok))

        feat_info = extract_top_features(pipe, top_n=12)
        if exp_id == "E4":
            e4_features = feat_info

        all_results[exp_id] = {
            "experiment": exp,
            "hyperparameters": {
                "ngram_range": list(best_ngram),
                "min_df": 2,
                "max_features": 5000,
                "sublinear_tf": True,
                "C": best_c,
                "solver": "lbfgs",
                "max_iter": 1000,
                "random_state": RANDOM_STATE,
                "class_weight": exp["class_weight"],
            },
            "vocabulary_size": vocab_size,
            "emoji_token_features_in_vocab": emoji_vocab_count,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "top_features": feat_info,
        }

        comparison_rows.append(
            {
                "Experiment": exp_id,
                "Emoji_Handling": exp["emoji_handling"],
                "Class_Weight": exp["class_weight_label"],
                "Vocab_Size": vocab_size,
                "Emoji_Features": emoji_vocab_count,
                "Val_Macro_F1": val_metrics["macro_f1"],
                "Val_Accuracy": val_metrics["accuracy"],
                "Test_Macro_F1": test_metrics["macro_f1"],
                "Test_Accuracy": test_metrics["accuracy"],
                "Test_Macro_Precision": test_metrics["macro_precision"],
                "Test_Macro_Recall": test_metrics["macro_recall"],
                "Test_Weighted_F1": test_metrics["weighted_f1"],
                "Test_Pos_F1": test_metrics["per_class"]["Positive"]["f1_score"],
                "Test_Neu_F1": test_metrics["per_class"]["Neutral"]["f1_score"],
                "Test_Neg_F1": test_metrics["per_class"]["Negative"]["f1_score"],
                "Test_Neg_Recall": test_metrics["per_class"]["Negative"]["recall"],
            }
        )

        for cls in CLASS_ORDER:
            pc = test_metrics["per_class"][cls]
            per_class_rows.append(
                {
                    "Experiment": exp_id,
                    "Emoji_Handling": exp["emoji_handling"],
                    "Class_Weight": exp["class_weight_label"],
                    "Class": cls,
                    "Precision": pc["precision"],
                    "Recall": pc["recall"],
                    "F1_Score": pc["f1_score"],
                    "Support": pc["support"],
                }
            )

        print(
            f"  [{exp_id}] {exp['emoji_handling']:22s} | CW={exp['class_weight_label']:8s} | "
            f"Vocab={vocab_size:4d} (emoji_feats={emoji_vocab_count:3d}) | "
            f"VAL Macro-F1={val_metrics['macro_f1']:.4f} | "
            f"TEST Macro-F1={test_metrics['macro_f1']:.4f}, Acc={test_metrics['accuracy']:.4f}, "
            f"MacroP={test_metrics['macro_precision']:.4f}, MacroR={test_metrics['macro_recall']:.4f}, "
            f"NegRec={test_metrics['per_class']['Negative']['recall']:.4f}"
        )

    # 4. Compute RO1 / RO2 / RO3 Factorial Deltas on Held-Out Test Set
    e1_f1 = all_results["E1"]["test_metrics"]["macro_f1"]
    e2_f1 = all_results["E2"]["test_metrics"]["macro_f1"]
    e3_f1 = all_results["E3"]["test_metrics"]["macro_f1"]
    e4_f1 = all_results["E4"]["test_metrics"]["macro_f1"]

    deltas = {
        "RO1_emoji_effect_unweighted_E2_minus_E1": round(e2_f1 - e1_f1, 4),
        "RO1_emoji_effect_balanced_E4_minus_E3": round(e4_f1 - e3_f1, 4),
        "RO2_balance_effect_no_emoji_E3_minus_E1": round(e3_f1 - e1_f1, 4),
        "RO2_balance_effect_with_emoji_E4_minus_E2": round(e4_f1 - e2_f1, 4),
        "RO3_combined_gain_E4_minus_E1": round(e4_f1 - e1_f1, 4),
        "RO3_interaction_term_E4_minus_E3_minus_E2_plus_E1": round((e4_f1 - e3_f1) - (e2_f1 - e1_f1), 4),
        "RO3_negative_recall_gain_E4_minus_E1": round(
            all_results["E4"]["test_metrics"]["per_class"]["Negative"]["recall"]
            - all_results["E1"]["test_metrics"]["per_class"]["Negative"]["recall"],
            4,
        ),
    }

    summary_payload = {
        "metadata": {
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "random_state": RANDOM_STATE,
            "splits": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
            "selected_shared_hyperparameters": {"ngram_range": list(best_ngram), "C": best_c},
            "sklearn_version": __import__("sklearn").__version__,
            "python_version": sys.version.split()[0],
        },
        "majority_class_sanity_baseline": dummy_metrics,
        "experiments": all_results,
        "factorial_deltas_test_macro_f1": deltas,
    }

    with open(RESULTS_DIR / "experiment_results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    pd.DataFrame(comparison_rows).to_csv(RESULTS_DIR / "experiment_comparison_table.csv", index=False, encoding="utf-8")
    pd.DataFrame(per_class_rows).to_csv(RESULTS_DIR / "per_class_metrics.csv", index=False, encoding="utf-8")
    test_preds_df.to_csv(RESULTS_DIR / "test_predictions_all_models.csv", index=False, encoding="utf-8")

    # 5. Generate Publication-Quality Figures
    draw_pipeline_diagram(FIGURES_DIR / "methodology_pipeline.png", best_ngram, best_c)
    plot_experiment_comparison(all_results, FIGURES_DIR / "experiment_metrics_comparison.png")
    plot_confusion_matrices(all_results, FIGURES_DIR / "confusion_matrices_e1_e4.png")
    plot_top_emoji_weights(e4_features["top_emoji_features"], FIGURES_DIR / "top_emoji_feature_weights.png")

    print("\n" + "=" * 78)
    print("[Factorial Delta Summary — Held-Out Test Macro-F1]")
    for k, v in deltas.items():
        print(f"  {k}: {v:+.4f}")
    print(f"[Saved Models]  {MODELS_DIR}")
    print(f"[Saved Results] {RESULTS_DIR}")
    print(f"[Saved Figures] {FIGURES_DIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()
