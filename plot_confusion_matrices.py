r"""
Generate Publication-Quality Confusion Matrix & Full Metrics Images for E1, E2, E3, and E4
Reads the exact CSV outputs from:
  - D:\UUM\ASSIGNMENT\SEM5\NLP\ASS\ASS3\E1_E3\results\
  - D:\UUM\ASSIGNMENT\SEM5\NLP\ASS\ASS3\E2_E4\results\

Includes: Accuracy, Macro-F1, Precision (Macro), Recall (Macro), and Confusion Matrix.

Produces:
  1. Combined CSV           : E1_E4_metrics_summary.csv (in ASS3\)
  2. Combined 2x2 CM image  : E1_E4_confusion_matrices.png (in ASS3\)
  3. Metrics comparison img : E1_E4_metrics_comparison.png (in ASS3\)
  4. Paired CM images       : E1_E3\results\E1_E3_confusion_matrices.png
                              E2_E4\results\E2_E4_confusion_matrices.png
  5. Individual CM images   : E1..E4_confusion_matrix.png (in each results\ folder)
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
E1_E3_RES = BASE / "E1_E3" / "results"
E2_E4_RES = BASE / "E2_E4" / "results"

m13 = pd.read_csv(E1_E3_RES / "E1_E3_metrics.csv")
m24 = pd.read_csv(E2_E4_RES / "E2_E4_metrics.csv")
metrics_df = pd.concat([m13, m24], ignore_index=True)

# Sort nicely: E1, E2, E3, E4 (validation then test)
exp_order = {"E1": 1, "E2": 2, "E3": 3, "E4": 4}
split_order = {"validation": 1, "test": 2}
metrics_df["_eo"] = metrics_df["experiment"].map(exp_order)
metrics_df["_so"] = metrics_df["split"].map(split_order)
metrics_df = metrics_df.sort_values(["_eo", "_so"]).drop(columns=["_eo", "_so"]).reset_index(drop=True)
metrics_df.to_csv(BASE / "E1_E4_metrics_summary.csv", index=False)

test_metrics = metrics_df[metrics_df["split"] == "test"].set_index("experiment")

EXPERIMENTS = {
    "E1": (E1_E3_RES / "E1_confusion_matrix.csv", E1_E3_RES, "E1: Emoji Removed + No Weight (Baseline)"),
    "E2": (E2_E4_RES / "E2_confusion_matrix.csv", E2_E4_RES, "E2: Emoji Retained + No Weight (RO1)"),
    "E3": (E1_E3_RES / "E3_confusion_matrix.csv", E1_E3_RES, "E3: Emoji Removed + Balanced Weight (RO2)"),
    "E4": (E2_E4_RES / "E4_confusion_matrix.csv", E2_E4_RES, "E4: Emoji Retained + Balanced Weight (RO3)"),
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
})


def load_cm(csv_path: Path):
    df = pd.read_csv(csv_path, index_col=0, encoding="utf-8-sig")
    return df.values, list(df.index), list(df.columns)


def draw_cm_on_ax(ax, cm, row_labels, col_labels, title, subtitle, vmax=125):
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=vmax)
    ax.set_title(f"{title}\n{subtitle}", fontsize=9.2, fontweight="bold", pad=10)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=9.5, fontweight="bold")
    ax.set_yticklabels(row_labels, fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=9.3, fontweight="bold", labelpad=5)
    ax.set_ylabel("Actual (True) Label", fontsize=9.3, fontweight="bold", labelpad=5)

    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    thresh = vmax * 0.52
    for i in range(cm.shape[0]):
        row_sum = cm[i].sum()
        for j in range(cm.shape[1]):
            pct = cm[i, j] / max(1, row_sum) * 100.0
            ax.text(
                j, i,
                f"{cm[i, j]}\n({pct:.1f}%)",
                ha="center", va="center",
                fontsize=9.3, fontweight="bold",
                color="white" if cm[i, j] > thresh else "#111827",
            )
    return im


def main():
    loaded = {}
    global_max = 0
    for eid, (csv_path, out_dir, title) in EXPERIMENTS.items():
        cm, r_lbls, c_lbls = load_cm(csv_path)
        global_max = max(global_max, int(cm.max()))
        acc = float(test_metrics.loc[eid, "accuracy"])
        f1 = float(test_metrics.loc[eid, "macro_f1"])
        prec = float(test_metrics.loc[eid, "precision"])
        rec = float(test_metrics.loc[eid, "recall"])
        sub = f"Acc = {acc:.4f} | Macro-F1 = {f1:.4f} | Prec = {prec:.4f} | Rec = {rec:.4f}"
        loaded[eid] = (cm, r_lbls, c_lbls, out_dir, title, sub)

    vmax = max(125, global_max)

    # 1. Combined 2x2 Confusion Matrix Figure (E1, E2, E3, E4) with Acc, Macro-F1, Precision, Recall
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.6), dpi=300)
    for ax, eid in zip(axes.flat, ["E1", "E2", "E3", "E4"]):
        cm, r_lbls, c_lbls, _, title, sub = loaded[eid]
        im = draw_cm_on_ax(ax, cm, r_lbls, c_lbls, title, sub, vmax=vmax)

    fig.suptitle(
        "Group D — NLP Sentiment Classification Results & Confusion Matrices (Test Set, N = 257)\n"
        "TF-IDF + Logistic Regression Across E1–E4 (Accuracy, Macro-F1, Macro Precision, Macro Recall)",
        fontsize=12.5, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 0.94, 0.93])
    cbar_ax = fig.add_axes([0.95, 0.12, 0.018, 0.75])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Number of Test Comments", fontsize=9.5, fontweight="bold")
    combined_path = BASE / "E1_E4_confusion_matrices.png"
    fig.savefig(combined_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {combined_path}")

    # 2. Combined Metrics Bar Chart (Accuracy, Macro-F1, Precision, Recall) for E1–E4
    eids = ["E1", "E2", "E3", "E4"]
    x_labels = [
        "E1\nRemoved + None\n(Baseline)",
        "E2\nRetained + None\n(RO1)",
        "E3\nRemoved + Balanced\n(RO2)",
        "E4\nRetained + Balanced\n(RO3)",
    ]
    accs = [float(test_metrics.loc[e, "accuracy"]) for e in eids]
    f1s = [float(test_metrics.loc[e, "macro_f1"]) for e in eids]
    precs = [float(test_metrics.loc[e, "precision"]) for e in eids]
    recs = [float(test_metrics.loc[e, "recall"]) for e in eids]

    x = np.arange(len(eids))
    w = 0.20
    fig, ax = plt.subplots(figsize=(10.8, 5.0), dpi=300)
    b1 = ax.bar(x - 1.5 * w, accs, width=w, label="Accuracy", color="#2563EB", edgecolor="#111", lw=0.7)
    b2 = ax.bar(x - 0.5 * w, f1s, width=w, label="Macro-F1 (Main Metric)", color="#1F4E79", edgecolor="#111", lw=0.7)
    b3 = ax.bar(x + 0.5 * w, precs, width=w, label="Precision (Macro)", color="#F59E0B", edgecolor="#111", lw=0.7)
    b4 = ax.bar(x + 1.5 * w, recs, width=w, label="Recall (Macro)", color="#10B981", edgecolor="#111", lw=0.7)

    for bars in (b1, b2, b3, b4):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.012,
                f"{h:.4f}",
                ha="center", va="bottom",
                fontsize=7.6, fontweight="bold",
            )

    ax.set_title("Group D — Test Set Metrics Comparison Across E1–E4 (Accuracy, Macro-F1, Precision, Recall)", fontsize=11.5, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9.3, fontweight="bold")
    ax.set_ylabel("Score (0.0 – 1.0)", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 0.80)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.legend(loc="upper left", ncol=4, frameon=True, fontsize=8.8)
    plt.tight_layout()
    metrics_bar_path = BASE / "E1_E4_metrics_comparison.png"
    fig.savefig(metrics_bar_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {metrics_bar_path}")

    # 3. Paired CM images (E1 vs E3, and E2 vs E4)
    for pair_name, pair_eids, out_path in [
        ("E1 vs. E3 (Emoji Removed)", ["E1", "E3"], E1_E3_RES / "E1_E3_confusion_matrices.png"),
        ("E2 vs. E4 (Emoji Retained as Text Tokens)", ["E2", "E4"], E2_E4_RES / "E2_E4_confusion_matrices.png"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), dpi=300)
        for ax, eid in zip(axes, pair_eids):
            cm, r_lbls, c_lbls, _, title, sub = loaded[eid]
            draw_cm_on_ax(ax, cm, r_lbls, c_lbls, title, sub, vmax=vmax)
        fig.suptitle(f"Confusion Matrices & Metrics — {pair_name} (Test Set, N = 257)", fontsize=12, fontweight="bold", y=1.01)
        plt.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {out_path}")

    # 4. Individual single-experiment confusion matrix PNGs
    for eid in ["E1", "E2", "E3", "E4"]:
        cm, r_lbls, c_lbls, out_dir, title, sub = loaded[eid]
        fig, ax = plt.subplots(figsize=(6.3, 5.1), dpi=300)
        im = draw_cm_on_ax(ax, cm, r_lbls, c_lbls, title, sub, vmax=vmax)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Comment Count", fontsize=9, fontweight="bold")
        plt.tight_layout()
        single_path = out_dir / f"{eid}_confusion_matrix.png"
        fig.savefig(single_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {single_path}")


if __name__ == "__main__":
    main()
