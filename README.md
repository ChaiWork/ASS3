# STINK3114 NLP Group D — Multilingual Social Media Sentiment Analysis Project

**Course**: STINK3114 Natural Language Processing (Session I 2026/2027 - A261)  
**Lecturer**: Dr. Muhammad Syafiq bin Mohd Pozi  
**Group D Members**:
- **CHAIRIS A/LPUM** (303800)
- **CHEN WEI YU** (303822)
- **AMANDA SOH YUE RU** (303873)
- **LOH YI KUAN** (305590)

---

## 1. Project Overview & Research Objectives

This repository contains Group D's complete dataset design, preprocessing pipeline, and controlled model implementation investigating **3-class sentiment classification (`Positive`, `Neutral`, `Negative`)** on **1,712 Malaysian YouTube comments** spanning **Bahasa Melayu (Malay)**, **English**, and code-switched **Malaysian Manglish (*Bahasa Rojak*)**.

- **RO1 — Emoji Retention vs. Removal**: Evaluates whether preserving emojis (converted into alphanumeric text tokens for TF-IDF) improves classification over stripping emojis.
- **RO2 — Class-Imbalance Mitigation**: Evaluates inverse-frequency cost-sensitive weighting (`class_weight="balanced"`) against majority-class prediction collapse on naturally skewed data (`Neutral`: 51.40%, `Positive`: 29.44%, `Negative`: 19.16%).
- **RO3 — Combined Interaction**: Evaluates the $2 \times 2$ factorial interaction between emoji handling and class-balanced training using a fixed, lightweight **TF-IDF + Multinomial Logistic Regression** pipeline (`random_state=42`).

---

## 2. Project Directory Structure

```text
ASS3/
├── README.md                                # Top-level project overview & execution guide
├── plot_confusion_matrices.py               # Generates combined & per-experiment PNG charts
├── E1_E4_metrics_summary.csv                # Consolidated E1–E4 validation & test metrics
├── E1_E4_confusion_matrices.png             # Combined 2×2 test confusion matrix figure (E1–E4)
├── E1_E4_metrics_comparison.png             # Grouped bar chart (Accuracy, Macro-F1, Precision, Recall)
│
├── Group_D_Dataset_Assignment/              # Assignment 3A: Dataset collection, preprocessing & splits
│   ├── dataset/
│   │   ├── raw_dataset.csv                  # Original unmodified YouTube comments (N = 1,712)
│   │   ├── group_d_sentiment_dataset.csv    # Final aligned dataset (Track A + Track B + metadata)
│   │   ├── train.csv                        # Locked stratified 70% training split (n = 1,198)
│   │   ├── validation.csv                   # Locked stratified 15% validation split (n = 257)
│   │   ├── test.csv                         # Locked stratified 15% testing split (n = 257)
│   │   └── split_manifest.csv               # Record-level split manifest (seed = 42)
│   ├── src/
│   │   ├── collect_data.py                  # YouTube Data API v3 / public Innertube collector
│   │   ├── preprocess.py                    # Non-destructive cleaner & Track A / Track B builder
│   │   ├── annotate_dataset.py              # Dual-pass annotation & adjudication (κ = 0.9131)
│   │   ├── create_splits.py                 # Composite stratified 70/15/15 splitter
│   │   ├── validate_dataset.py              # 12-point automated Quality Assurance test suite
│   │   ├── dataset_statistics.py            # Corpus exploratory statistics & charts
│   │   ├── emoji_tokenizer.py               # Reusable Unicode emoji/emoticon-to-text module
│   │   ├── train_evaluate_models.py         # Unified E1–E4 validation-tuned pipeline
│   │   └── predict_prototype.py             # CLI sentiment prediction prototype
│   ├── logs/                                # Collection, agreement & corpus statistics logs
│   ├── results/                             # Validation tuning log & full evaluation archives
│   ├── .env.example                         # Environment template for YouTube Data API v3 key
│   ├── requirements.txt                     # Python package dependencies
│   └── README.md                            # Detailed dataset pipeline documentation
│
├── E1_E3/                                   # Assignment 3B: Standalone Emoji-Removed Experiments
│   ├── e1_e3.py                             # Runs E1 (Removed + None) & E3 (Removed + Balanced)
│   ├── train.csv                            # Identical locked training split (n = 1,198)
│   ├── validation.csv                       # Identical locked validation split (n = 257)
│   ├── test.csv                             # Identical locked testing split (n = 257)
│   └── results/
│       ├── E1_E3_metrics.csv                # Accuracy, Macro-F1, Precision, Recall (Val & Test)
│       ├── E1_E3_confusion_matrices.png     # Side-by-side E1 vs. E3 confusion matrix image
│       ├── E1_confusion_matrix.{csv,png}    # E1 test confusion matrix (table & plot)
│       ├── E1_classification_report.txt     # E1 per-class test report
│       ├── E1_test_predictions.csv          # E1 per-comment test predictions
│       ├── E3_confusion_matrix.{csv,png}    # E3 test confusion matrix (table & plot)
│       ├── E3_classification_report.txt     # E3 per-class test report
│       └── E3_test_predictions.csv          # E3 per-comment test predictions
│
└── E2_E4/                                   # Assignment 3B: Standalone Emoji-Retained Experiments
    ├── e2_e4.py                             # Runs E2 (Retained + None) & E4 (Retained + Balanced)
    ├── train.csv                            # Identical locked training split (n = 1,198)
    ├── validation.csv                       # Identical locked validation split (n = 257)
    ├── test.csv                             # Identical locked testing split (n = 257)
    └── results/
        ├── E2_E4_metrics.csv                # Accuracy, Macro-F1, Precision, Recall (Val & Test)
        ├── E2_E4_confusion_matrices.png     # Side-by-side E2 vs. E4 confusion matrix image
        ├── E2_confusion_matrix.{csv,png}    # E2 test confusion matrix (table & plot)
        ├── E2_classification_report.txt     # E2 per-class test report
        ├── E2_test_predictions.csv          # E2 per-comment test predictions
        ├── E4_confusion_matrix.{csv,png}    # E4 test confusion matrix (table & plot)
        ├── E4_classification_report.txt     # E4 per-class test report
        └── E4_test_predictions.csv          # E4 per-comment test predictions
```

---

## 3. Experimental Design (`E1–E4`)

All four standalone experiments share identical locked dataset splits (`train.csv` / `validation.csv` / `test.csv`), identical TF-IDF settings (`TfidfVectorizer(max_features=5000, ngram_range=(1, 2), lowercase=True)`, fit on training data only), and identical classifier settings (`LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)`):

| Experiment | Folder & Script | Input Column & Emoji Handling | Class Weighting | Role |
|---|---|---|---|---|
| **E1** | `E1_E3/e1_e3.py` | `cleaned_text_no_emojis` (Removed) | `None` | Baseline (Unweighted, Text Only) |
| **E2** | `E2_E4/e2_e4.py` | `cleaned_text_with_emojis` → Text Tokens (Retained) | `None` | RO1 (Emoji Retention Effect) |
| **E3** | `E1_E3/e1_e3.py` | `cleaned_text_no_emojis` (Removed) | `"balanced"` | RO2 (Class-Imbalance Mitigation) |
| **E4** | `E2_E4/e2_e4.py` | `cleaned_text_with_emojis` → Text Tokens (Retained) | `"balanced"` | RO3 (Combined Interaction) |

---

## 4. Quick Start & Reproducibility

### Prerequisites
```bash
pip install pandas scikit-learn matplotlib
```
*(To re-run the full Assignment 3A dataset collection/report pipeline inside `Group_D_Dataset_Assignment/`, install `pip install -r Group_D_Dataset_Assignment/requirements.txt`.)*

### Run the Model Implementation Experiments (`E1–E4`)
```bash
# 1. Run E1 and E3 (Emoji Removed)
python E1_E3/e1_e3.py

# 2. Run E2 and E4 (Emoji Retained as Text Tokens)
python E2_E4/e2_e4.py

# 3. Generate combined summary CSV, metrics comparison bar chart, and confusion matrix PNGs
python plot_confusion_matrices.py
```

---

## 5. Summary of Held-Out Test Set Results ($N_{\text{test}} = 257$)

| Experiment | Emoji Handling | Class Weight | Accuracy | **Macro-F1 (Main)** | Precision (Macro) | Recall (Macro) | Minority `Negative` Recall |
|---|---|---|---:|---:|---:|---:|---:|
| **E1** | Removed | None | 0.6226 | **0.5036** | 0.5613 | 0.5040 | 14.0% (7/50) |
| **E2** | Retained (Text Tokens) | None | **0.6342** | **0.5266** | 0.5965 | 0.5218 | 18.0% (9/50) |
| **E3** | Removed | Balanced | 0.6304 | **0.6151** | **0.6117** | **0.6398** | **66.0% (33/50)** |
| **E4** | Retained (Text Tokens) | Balanced | 0.6304 | **0.6136** | 0.6113 | 0.6379 | **66.0% (33/50)** |

- **Main Metric**: Macro-F1 (prevents majority `Neutral` accuracy from masking minority `Negative` collapse).
- **Key Takeaway**: Under unweighted training (`E1` → `E2`), retaining emoji text tokens improves Test Macro-F1 by `+0.0231` (`0.5036 → 0.5266`) and Accuracy by `+0.0117`. Applying `class_weight="balanced"` (`E3`, `E4`) resolves majority-class collapse, boosting minority `Negative` recall by **+52.0 percentage points** (`14.0% → 66.0%`) and lifting Macro-F1 above **0.61**.
