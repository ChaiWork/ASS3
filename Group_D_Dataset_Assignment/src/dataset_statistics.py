"""
Phase 6 — Comprehensive Dataset Statistics & Exploratory Data Analysis (dataset_statistics.py)
Group D — STINK3114 Natural Language Processing (UUM)

Computes exact descriptive statistics across the final validated dataset
(`dataset/group_d_sentiment_dataset.csv`) and collection logs (`logs/collection_log.csv`),
saves a structured JSON summary (`logs/dataset_statistics_summary.json`), and generates
four 300-DPI publication-quality figures in `figures/`:
  1. `figures/sentiment_distribution.png`
  2. `figures/emoji_distribution.png`
  3. `figures/text_length_distribution.png`
  4. `figures/emoji_by_sentiment.png`
"""

import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import extract_emojis_and_emoticons

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
LOGS_DIR = BASE_DIR / "logs"
FIGURES_DIR = BASE_DIR / "figures"

# Emoji Unicode names/descriptions for clean publication rendering (avoids missing glyph boxes in matplotlib)
EMOJI_LABELS = {
    "😂": "😂 (U+1F602 Face w/ Tears of Joy)",
    "🤣": "🤣 (U+1F923 Rolling on Floor Laughing)",
    "😭": "😭 (U+1F62D Loudly Crying Face)",
    "😅": "😅 (U+1F605 Grinning Face w/ Sweat)",
    "🔥": "🔥 (U+1F525 Fire)",
    "👍": "👍 (U+1F44D Thumbs Up)",
    "👍🏻": "👍🏻 (U+1F44D Thumbs Up Light)",
    "❤": "❤️ (U+2764 Red Heart)",
    "❤️": "❤️ (U+2764 Red Heart)",
    "🎉": "🎉 (U+1F389 Party Popper)",
    "😮": "😮 (U+1F62E Face w/ Open Mouth)",
    "✅": "✅ (U+2705 Check Mark Button)",
    "🇲🇾": "🇲🇾 (U+1F1F2-1F1FE Flag: Malaysia)",
    "😢": "😢 (U+1F622 Crying Face)",
    "😊": "😊 (U+1F60A Smiling Face)",
    "🙏": "🙏 (U+1F64F Folded Hands)",
    "🤲": "🤲 (U+1F932 Palms Up Together)",
    "🤲🏻": "🤲🏻 (U+1F932 Palms Up Light)",
    "😍": "😍 (U+1F60D Heart Eyes)",
    "😡": "😡 (U+1F621 Enraged Face)",
    "🤔": "🤔 (U+1F914 Thinking Face)",
    "😆": "😆 (U+1F606 Grinning Squinting Face)",
    "😁": "😁 (U+1F601 Beaming Face)",
    "🥰": "🥰 (U+1F970 Smiling Face w/ Hearts)",
    "👏": "👏 (U+1F44F Clapping Hands)",
    "💔": "💔 (U+1F494 Broken Heart)",
    "🤮": "🤮 (U+1F92E Face Vomiting)",
    "🥺": "🥺 (U+1F97A Pleading Face)",
}


def format_emoji_name(em: str) -> str:
    if em in EMOJI_LABELS:
        # Return ASCII-safe codepoint + descriptive label for matplotlib font compatibility
        desc = EMOJI_LABELS[em].split("(", 1)[1].rstrip(")")
        return desc
    codepoints = "-".join(f"U+{ord(c):04X}" for c in em)
    return f"Emoji ({codepoints})"


def compute_and_plot_statistics() -> dict:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_DIR / "group_d_sentiment_dataset.csv", encoding="utf-8")
    log_df = pd.read_csv(LOGS_DIR / "collection_log.csv", encoding="utf-8")
    train_df = pd.read_csv(DATASET_DIR / "train.csv", encoding="utf-8")
    val_df = pd.read_csv(DATASET_DIR / "validation.csv", encoding="utf-8")
    test_df = pd.read_csv(DATASET_DIR / "test.csv", encoding="utf-8")

    # 1. Dataset Overview
    total_retrieved = int(log_df["retrieved_count"].sum())
    total_rejected = int(log_df["rejected_total"].sum())
    rej_empty = int(log_df["rejected_empty_or_short"].sum())
    rej_dup = int(log_df["rejected_duplicate"].sum())
    rej_unusable = int(log_df["rejected_unusable_or_script"].sum())
    total_valid = int(len(df))
    unique_comments = int(df["raw_text"].nunique())
    candidate_sources = int(len(log_df))
    active_sources = int(df["source_id"].nunique())
    min_ts = str(df["timestamp"].min())
    max_ts = str(df["timestamp"].max())

    # Map domain from collection_log
    src_to_domain = dict(zip(log_df["source_id"], log_df["domain"]))
    df["domain"] = df["source_id"].map(src_to_domain)
    domain_counts = df["domain"].value_counts().to_dict()

    # 2. Sentiment Distribution
    sentiment_order = ["Neutral", "Positive", "Negative"]
    sent_counts = {k: int((df["label"] == k).sum()) for k in sentiment_order}
    sent_pcts = {k: round(sent_counts[k] / total_valid * 100.0, 2) for k in sentiment_order}

    # 3. Emoji Analysis
    has_emoji_count = int(df["has_emoji"].sum())
    no_emoji_count = int((~df["has_emoji"]).sum())
    has_emoji_pct = round(has_emoji_count / total_valid * 100.0, 2)
    no_emoji_pct = round(no_emoji_count / total_valid * 100.0, 2)
    total_emoji_instances = int(df["emoji_count"].sum())
    mean_emoji_all = round(float(df["emoji_count"].mean()), 3)
    mean_emoji_when_present = round(float(df.loc[df["has_emoji"], "emoji_count"].mean()), 3)
    max_emoji_in_comment = int(df["emoji_count"].max())

    all_emojis_counter = Counter()
    class_emoji_counter = {k: Counter() for k in sentiment_order}
    for _, row in df.iterrows():
        ems = [e.replace("\uFE0F", "").replace("\uFE0E", "") for e in extract_emojis_and_emoticons(str(row["cleaned_text_with_emojis"]))]
        all_emojis_counter.update(ems)
        class_emoji_counter[row["label"]].update(ems)

    top_15_emojis = [
        {
            "emoji": em,
            "codepoint": " ".join(f"U+{ord(c):04X}" for c in em),
            "description": format_emoji_name(em),
            "count": int(cnt),
        }
        for em, cnt in all_emojis_counter.most_common(15)
    ]

    emoji_by_sentiment = {}
    for cls in sentiment_order:
        sub = df[df["label"] == cls]
        em_rows = int(sub["has_emoji"].sum())
        no_em_rows = int((~sub["has_emoji"]).sum())
        tot_cls = len(sub)
        emoji_by_sentiment[cls] = {
            "total_class_samples": tot_cls,
            "with_emoji_count": em_rows,
            "with_emoji_pct": round(em_rows / max(1, tot_cls) * 100.0, 2),
            "without_emoji_count": no_em_rows,
            "without_emoji_pct": round(no_em_rows / max(1, tot_cls) * 100.0, 2),
            "total_emoji_occurrences": int(sub["emoji_count"].sum()),
            "mean_emojis_per_comment": round(float(sub["emoji_count"].mean()), 3),
            "top_5_emojis": [
                {"emoji": e, "count": int(c)} for e, c in class_emoji_counter[cls].most_common(5)
            ],
        }

    # 4. Language Analysis
    lang_order = ["Malay", "Manglish", "English", "Other/Uncertain"]
    lang_counts = {k: int((df["language"] == k).sum()) for k in lang_order}
    lang_pcts = {k: round(lang_counts[k] / total_valid * 100.0, 2) for k in lang_order}

    # 5. Text Statistics
    wc = df["word_count"]
    cc_a = df["character_count"]
    cc_b = df["cleaned_text_no_emojis"].astype(str).str.len()

    text_stats = {
        "word_count": {
            "mean": round(float(wc.mean()), 2),
            "std": round(float(wc.std()), 2),
            "median": round(float(wc.median()), 2),
            "q25": round(float(wc.quantile(0.25)), 2),
            "q75": round(float(wc.quantile(0.75)), 2),
            "min": int(wc.min()),
            "max": int(wc.max()),
        },
        "char_count_track_a_with_emojis": {
            "mean": round(float(cc_a.mean()), 2),
            "std": round(float(cc_a.std()), 2),
            "median": round(float(cc_a.median()), 2),
            "q25": round(float(cc_a.quantile(0.25)), 2),
            "q75": round(float(cc_a.quantile(0.75)), 2),
            "min": int(cc_a.min()),
            "max": int(cc_a.max()),
        },
        "char_count_track_b_no_emojis": {
            "mean": round(float(cc_b.mean()), 2),
            "std": round(float(cc_b.std()), 2),
            "median": round(float(cc_b.median()), 2),
            "min": int(cc_b.min()),
            "max": int(cc_b.max()),
        },
    }

    # ==========================================================================
    # Generate 4 Publication-Quality Visualizations (300 DPI)
    # ==========================================================================
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
    })

    # Figure 1: Sentiment Class Distribution (Overall & Stratified Splits)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300, gridspec_kw={"width_ratios": [1.15, 1.25]})
    palette_sent = {"Neutral": "#4C72B0", "Positive": "#2CA02C", "Negative": "#D62728"}

    # Panel A: Overall Sentiment Distribution
    ax1 = axes[0]
    bars1 = ax1.bar(
        sentiment_order,
        [sent_counts[k] for k in sentiment_order],
        color=[palette_sent[k] for k in sentiment_order],
        edgecolor="#1a1a1a",
        linewidth=0.8,
        width=0.58,
    )
    ax1.set_title(f"(a) Natural Sentiment Class Distribution (N = {total_valid:,})", fontsize=11, fontweight="bold", pad=12)
    ax1.set_ylabel("Number of Comments", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Sentiment Class", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, max(sent_counts.values()) * 1.18)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, k in zip(bars1, sentiment_order):
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            h + 18,
            f"{sent_counts[k]}\n({sent_pcts[k]:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    # Panel B: Stratified Train / Validation / Test Distribution
    ax2 = axes[1]
    x = np.arange(len(sentiment_order))
    w = 0.25
    tr_c = [int((train_df["label"] == k).sum()) for k in sentiment_order]
    va_c = [int((val_df["label"] == k).sum()) for k in sentiment_order]
    te_c = [int((test_df["label"] == k).sum()) for k in sentiment_order]

    b_tr = ax2.bar(x - w, tr_c, width=w, label=f"Train (70%, n={len(train_df)})", color="#1f77b4", edgecolor="#1a1a1a", linewidth=0.6)
    b_va = ax2.bar(x, va_c, width=w, label=f"Validation (15%, n={len(val_df)})", color="#ff7f0e", edgecolor="#1a1a1a", linewidth=0.6)
    b_te = ax2.bar(x + w, te_c, width=w, label=f"Test (15%, n={len(test_df)})", color="#2ca02c", edgecolor="#1a1a1a", linewidth=0.6)

    ax2.set_title("(b) Stratified Split Preservation (Seed = 42)", fontsize=11, fontweight="bold", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(sentiment_order, fontsize=10)
    ax2.set_ylabel("Number of Comments", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Sentiment Class", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, max(tr_c) * 1.18)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend(frameon=True, fontsize=8.5, loc="upper right")
    for b_group in (b_tr, b_va, b_te):
        for bar in b_group:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 8, f"{int(h)}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "sentiment_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Emoji Presence Distribution & Top 10 Most Frequent Emojis
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=300, gridspec_kw={"width_ratios": [1.0, 1.45]})
    ax1 = axes[0]
    em_cats = ["Without Emojis\n(Text Only)", "With Emojis\n(≥1 Emoji/Emoticon)"]
    em_vals = [no_emoji_count, has_emoji_count]
    em_pcts_list = [no_emoji_pct, has_emoji_pct]
    bars_em = ax1.bar(em_cats, em_vals, color=["#6C757D", "#F59E0B"], edgecolor="#1a1a1a", linewidth=0.8, width=0.52)
    ax1.set_title("(a) Emoji Presence Across Dataset", fontsize=11, fontweight="bold", pad=12)
    ax1.set_ylabel("Number of Comments", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, max(em_vals) * 1.18)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, val, pct in zip(bars_em, em_vals, em_pcts_list):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 18,
            f"{val}\n({pct:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    ax2 = axes[1]
    top10 = top_15_emojis[:10][::-1]
    y_labels = [item["description"] for item in top10]
    x_counts = [item["count"] for item in top10]
    bars_top = ax2.barh(y_labels, x_counts, color="#3B82F6", edgecolor="#1a1a1a", linewidth=0.7, height=0.65)
    ax2.set_title(f"(b) Top 10 Most Frequent Emojis (Total Occurrences = {total_emoji_instances})", fontsize=11, fontweight="bold", pad=12)
    ax2.set_xlabel("Total Occurrences in Corpus", fontsize=10, fontweight="bold")
    ax2.set_xlim(0, max(x_counts) * 1.16)
    ax2.grid(axis="x", linestyle="--", alpha=0.4)
    for bar, cnt in zip(bars_top, x_counts):
        ax2.text(cnt + 4, bar.get_y() + bar.get_height() / 2, f"{cnt}", va="center", ha="left", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "emoji_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: Text Length Distribution (Word Count & Character Count Histograms)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=300)
    ax1 = axes[0]
    wc_clipped = np.clip(wc, 0, 80)
    ax1.hist(wc_clipped, bins=28, color="#2563EB", edgecolor="white", alpha=0.85)
    ax1.axvline(wc.mean(), color="#DC2626", linestyle="--", linewidth=1.8, label=f"Mean = {wc.mean():.1f} words")
    ax1.axvline(wc.median(), color="#16A34A", linestyle="-", linewidth=1.8, label=f"Median = {wc.median():.0f} words")
    ax1.set_title("(a) Comment Word Count Distribution (Clipped at 80)", fontsize=11, fontweight="bold", pad=12)
    ax1.set_xlabel("Word Count per Comment", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Frequency (Comments)", fontsize=10, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(frameon=True, fontsize=9)

    ax2 = axes[1]
    cc_clipped = np.clip(cc_a, 0, 450)
    ax2.hist(cc_clipped, bins=28, color="#7C3AED", edgecolor="white", alpha=0.85)
    ax2.axvline(cc_a.mean(), color="#DC2626", linestyle="--", linewidth=1.8, label=f"Mean = {cc_a.mean():.1f} chars")
    ax2.axvline(cc_a.median(), color="#16A34A", linestyle="-", linewidth=1.8, label=f"Median = {cc_a.median():.0f} chars")
    ax2.set_title("(b) Character Length Distribution (Track A, Clipped at 450)", fontsize=11, fontweight="bold", pad=12)
    ax2.set_xlabel("Character Count per Comment", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Frequency (Comments)", fontsize=10, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "text_length_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 4: Emoji Frequency & Presence by Sentiment Class
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)
    ax1 = axes[0]
    x = np.arange(len(sentiment_order))
    w = 0.34
    with_em_vals = [emoji_by_sentiment[k]["with_emoji_count"] for k in sentiment_order]
    no_em_vals = [emoji_by_sentiment[k]["without_emoji_count"] for k in sentiment_order]

    b1 = ax1.bar(x - w / 2, no_em_vals, width=w, label="Without Emoji", color="#64748B", edgecolor="#1a1a1a", linewidth=0.7)
    b2 = ax1.bar(x + w / 2, with_em_vals, width=w, label="With Emoji (≥1)", color="#F59E0B", edgecolor="#1a1a1a", linewidth=0.7)
    ax1.set_title("(a) Emoji-Bearing vs. Text-Only Comments by Class", fontsize=11, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(sentiment_order, fontsize=10)
    ax1.set_ylabel("Number of Comments", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Sentiment Class", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, max(no_em_vals) * 1.18)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(frameon=True, fontsize=9)
    for bar in b1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 10, f"{int(h)}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for bar, k in zip(b2, sentiment_order):
        h = bar.get_height()
        pct = emoji_by_sentiment[k]["with_emoji_pct"]
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 10, f"{int(h)}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8.2, fontweight="bold")

    ax2 = axes[1]
    tot_em_cls = [emoji_by_sentiment[k]["total_emoji_occurrences"] for k in sentiment_order]
    mean_em_cls = [emoji_by_sentiment[k]["mean_emojis_per_comment"] for k in sentiment_order]
    bars_occ = ax2.bar(sentiment_order, tot_em_cls, color=[palette_sent[k] for k in sentiment_order], edgecolor="#1a1a1a", linewidth=0.8, width=0.52)
    ax2.set_title("(b) Total Emoji Occurrences & Density by Sentiment Class", fontsize=11, fontweight="bold", pad=12)
    ax2.set_ylabel("Total Emoji Occurrences", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Sentiment Class", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, max(tot_em_cls) * 1.20)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, tot_c, avg_c in zip(bars_occ, tot_em_cls, mean_em_cls):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 14,
            f"{tot_c} emojis\n(mean={avg_c:.2f}/cmt)",
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "emoji_by_sentiment.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "dataset_overview": {
            "total_retrieved_samples": total_retrieved,
            "total_rejected_samples": total_rejected,
            "rejected_breakdown": {
                "empty_or_too_short": rej_empty,
                "duplicate": rej_dup,
                "unusable_url_or_non_latin_script": rej_unusable,
            },
            "total_valid_samples": total_valid,
            "unique_comments": unique_comments,
            "candidate_video_sources": candidate_sources,
            "active_video_sources": active_sources,
            "collection_timestamp_range": {"start": min_ts, "end": max_ts},
            "domain_distribution": domain_counts,
        },
        "sentiment_distribution": {
            "counts": sent_counts,
            "percentages": sent_pcts,
            "imbalance_ratio_majority_to_minority": round(max(sent_counts.values()) / max(1, min(sent_counts.values())), 2),
        },
        "emoji_analysis": {
            "comments_with_emoji_count": has_emoji_count,
            "comments_with_emoji_pct": has_emoji_pct,
            "comments_without_emoji_count": no_emoji_count,
            "comments_without_emoji_pct": no_emoji_pct,
            "total_emoji_occurrences": total_emoji_instances,
            "mean_emojis_per_comment_overall": mean_emoji_all,
            "mean_emojis_per_emoji_bearing_comment": mean_emoji_when_present,
            "max_emojis_in_single_comment": max_emoji_in_comment,
            "top_15_emojis": top_15_emojis,
            "by_sentiment_class": emoji_by_sentiment,
        },
        "language_analysis": {
            "counts": lang_counts,
            "percentages": lang_pcts,
            "methodology_note": (
                "Automated lexical-marker language identification is an approximation; "
                "code-switched Malaysian Manglish/Bahasa Rojak shares cognate loanwords "
                "and shortforms with both standard Bahasa Melayu and informal English."
            ),
        },
        "text_statistics": text_stats,
    }

    summary_path = LOGS_DIR / "dataset_statistics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print(f"[Phase 6 Complete] Statistics saved to: {summary_path}")
    print(f"[Phase 6 Complete] 4 figures saved to : {FIGURES_DIR}")
    print(json.dumps(summary["sentiment_distribution"], indent=2))
    print(json.dumps(summary["language_analysis"]["counts"], indent=2))
    print(json.dumps(summary["text_statistics"]["word_count"], indent=2))
    print("=" * 70)
    return summary


if __name__ == "__main__":
    compute_and_plot_statistics()
