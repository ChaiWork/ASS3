# Group D — Multilingual Malaysian Social Media Sentiment Dataset (STINK3114)

**Course**: STINK3114 Natural Language Processing (Session I 2026/2027 - A261)  
**Assignment**: Assignment 3A — Dataset Design & Collection (Week 6–7) & Assignment 3B — Model Implementation / Methodology (Week 8–9)  
**GitHub Repository**: https://github.com/ChaiWork/ASS3  
**Group D Members**:
- **CHAIRIS A/LPUM** (303800)
- **CHEN WEI YU** (303822)
- **AMANDA SOH YUE RU** (303873)
- **LOH YI KUAN** (305590)

**Lecturer**: Dr. Muhammad Syafiq bin Mohd Pozi

---

## 1. Research Overview & Alignment (RO1, RO2, RO3)

This repository contains the complete, reproducible data engineering pipeline and validated multilingual social media sentiment dataset ($N = 1,712$ valid unique comments) designed for Group D's research project:

1. **RO1 — Emoji Retention vs. Emoji Removal**: Evaluates whether preserving sentiment-bearing Unicode emojis (`cleaned_text_with_emojis`, Track A) improves sentiment classification compared with stripping emojis (`cleaned_text_no_emojis`, Track B) under a fixed model architecture.
2. **RO2 — Class Imbalance Mitigation**: Investigates cost-sensitive class weighting (`Negative = 1.7438`, `Positive = 1.1313`, `Neutral = 0.6483`, computed strictly on the training split) and Focal Loss in preventing majority-class prediction collapse across naturally skewed sentiment distributions (`Neutral`: $51.40\%$, `Positive`: $29.44\%$, `Negative`: $19.16\%$).
3. **RO3 — Combined Experimental Interaction**: Examines the $2 \times 2$ interaction between emoji retention/removal and class-imbalance mitigation across Experiments **E1, E2, E3, and E4**.

---

## 2. Repository Structure

```text
Group_D_Dataset_Assignment/
│
├── dataset/
│   ├── raw_dataset.csv                  # Original unmodified comments (N = 1,712)
│   ├── group_d_sentiment_dataset.csv    # Final dataset with Track A, Track B & metadata
│   ├── train.csv                        # Stratified 70% training split (n = 1,198)
│   ├── validation.csv                   # Stratified 15% validation split (n = 257)
│   ├── test.csv                         # Stratified 15% testing split (n = 257)
│   └── split_manifest.csv               # Record-level split & experiment manifest (seed=42)
│
├── src/
│   ├── collect_data.py                  # Phase 2: Dual-mode YouTube comment collection
│   ├── annotate_dataset.py              # Phase 3: Dual-annotator + adjudication pipeline
│   ├── preprocess.py                    # Phase 4: Non-destructive cleaning & Track A/B branching
│   ├── validate_dataset.py              # Phase 5: 12-point automated QA test suite
│   ├── dataset_statistics.py            # Phase 6: Descriptive statistics & 300-DPI plots
│   ├── create_splits.py                 # Phase 7: Composite stratified 70/15/15 splitting
│   ├── generate_reports.py              # Phase 8: Dataset Description DOCX & PDF builder
│   ├── emoji_tokenizer.py               # Model Imp: Converts 145+ Unicode emojis/emoticons to TF-IDF text tokens
│   ├── train_evaluate_models.py         # Model Imp: Controlled 2x2 TF-IDF + Logistic Regression (E1–E4)
│   ├── predict_prototype.py             # Model Imp: Working CLI NLP Sentiment Prediction Prototype
│   └── generate_methodology_report.py   # Model Imp: Methodology Section DOCX & PDF builder
│
├── models/
│   ├── E1_emoji_removed_none.joblib     # E1 Baseline (Emoji Removed, No Class Weight)
│   ├── E2_emoji_retained_none.joblib    # E2 RO1 (Emoji Retained as Text Tokens, No Weight)
│   ├── E3_emoji_removed_balanced.joblib # E3 RO2 (Emoji Removed, class_weight="balanced")
│   └── E4_emoji_retained_balanced.joblib# E4 RO3 Combined (Emoji Retained + Balanced Weight)
│
├── results/
│   ├── experiment_results_summary.json  # Complete Validation & Test metrics + factorial deltas
│   ├── experiment_comparison_table.csv  # E1–E4 summary comparison table
│   ├── per_class_metrics.csv            # Precision, Recall, F1 per class (Pos/Neu/Neg)
│   ├── validation_tuning_log.csv        # Validation-only hyperparameter grid search log
│   └── test_predictions_all_models.csv  # Held-out test set predictions across E1–E4
│
├── reports/
│   ├── Dataset_Description_Group_D.docx            # Ass 3A Editable Dataset Report
│   ├── Dataset_Description_Group_D.pdf             # Ass 3A Submission PDF Dataset Report
│   ├── Methodology_Model_Implementation_Group_D.docx # Ass 3B Editable Methodology Report (2–4 pages)
│   ├── Methodology_Model_Implementation_Group_D.pdf  # Ass 3B Submission PDF Methodology Report
│   ├── Dataset_QA_Report.txt                       # Automated validation report (12/12 PASS)
│   └── Annotation_Guidelines_Group_D.md            # 3-class annotation guide & decision rules
│
├── figures/
│   ├── sentiment_distribution.png       # Ass 3A Fig 1: Class distribution & split preservation
│   ├── emoji_distribution.png           # Ass 3A Fig 2: Emoji presence & Top 10 emojis
│   ├── text_length_distribution.png     # Ass 3A Fig 3: Word & character length histograms
│   ├── emoji_by_sentiment.png           # Ass 3A Fig 4: Emoji density & presence by sentiment class
│   ├── methodology_pipeline.png         # Ass 3B Fig 1: End-to-end NLP methodology flowchart
│   ├── experiment_metrics_comparison.png# Ass 3B Fig 2: Test Macro-F1, Acc, Recall comparison (E1–E4)
│   ├── confusion_matrices_e1_e4.png     # Ass 3B Fig 3: 2x2 Test Confusion Matrices (E1–E4)
│   └── top_emoji_feature_weights.png    # Ass 3B Fig 4: Top learned emoji weights in E4
│
├── logs/
│   ├── collection_log.csv               # Per-video retrieval, rejection & retention counts
│   ├── annotation_disagreements.csv     # Log of 92 adjudicated annotator disagreements
│   ├── annotation_agreement_summary.json# Cohen's Kappa (κ = 0.9131) & confusion matrix
│   ├── dataset_statistics_summary.json  # Exact computed statistics across final corpus
│   └── experimental_readiness_config.json # E1–E4 experimental matrix & train class weights
│
├── .env.example                         # Secure environment variable template
├── requirements.txt                     # Python package dependencies
└── README.md                            # Project documentation & execution instructions
```

---

## 3. Environment Setup & Reproducible Execution

### 3.1 Install Dependencies
```bash
pip install -r requirements.txt
```

### 3.2 Configure API Credentials (Optional)
Copy `.env.example` to `.env` if you wish to collect fresh comments via the authenticated **YouTube Data API v3** (`commentThreads.list`):
```bash
cp .env.example .env
# Edit .env and set YOUTUBE_API_KEY=your_google_cloud_api_key
```
When `YOUTUBE_API_KEY` is not set, `src/collect_data.py` automatically falls back to YouTube's public **Innertube JSON pagination endpoint** (`/youtubei/v1/next`), allowing zero-key execution.

### 3.3 Execute the End-to-End Pipeline
```bash
# Step 1: Collect raw comments across 6 Malaysian domains (saves dataset/raw_dataset.csv & logs/collection_log.csv)
python src/collect_data.py

# Step 2: Clean & generate Track A (emoji retained) and Track B (emoji stripped)
python src/preprocess.py

# Step 3: Run dual-pass sentiment annotation, compute Cohen's Kappa, and adjudicate disagreements
python src/annotate_dataset.py

# Step 4: Create stratified 70/15/15 splits (seed=42) and compute train-only balanced class weights
python src/create_splits.py

# Step 5: Run 12 automated Quality Assurance assertions (generates reports/Dataset_QA_Report.txt)
python src/validate_dataset.py

# Step 6: Compute descriptive statistics and generate 4 publication-quality figures (300 DPI)
python src/dataset_statistics.py

# Step 7: Build Assignment 3A editable DOCX and submission-ready PDF Dataset Description reports
python src/generate_reports.py

# Step 8 (Assignment 3B): Train & evaluate E1–E4 TF-IDF + Logistic Regression models (seed=42)
python src/train_evaluate_models.py

# Step 9 (Assignment 3B): Run the working NLP sentiment prediction prototype (demo or custom text)
python src/predict_prototype.py --demo
python src/predict_prototype.py --text "Makanan kat bazar ni sedap gila bro, harga pun berbaloi 😍🔥👍"

# Step 10 (Assignment 3B): Build editable DOCX and submission-ready PDF Methodology reports
python src/generate_methodology_report.py
```

---

## 5. Model Implementation & Controlled Factorial Results (Assignment 3B)

All four models use identical locked hyperparameters selected strictly on `validation.csv` (`TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000, sublinear_tf=True, stop_words=None, token_pattern=r"(?u)\b\w+\b")`, `LogisticRegression(C=5.0, solver="lbfgs", max_iter=1000, random_state=42)`):

| Experiment | Emoji Handling | Class Weight | Vocab (Emoji Feats) | Val Macro-F1 | **Test Macro-F1** | Test Accuracy | Macro Prec / Rec | Minority Negative Recall |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **E1 (Baseline)** | Removed | None | 2,652 (0) | 0.5898 | **0.6092** | 0.6615 | 0.617 / 0.604 | 0.400 |
| **E2 (RO1)** | Retained (Text Tokens) | None | 2,738 (87) | 0.5602 | **0.5873** | 0.6342 | 0.594 / 0.583 | 0.400 |
| **E3 (RO2)** | Removed | Balanced | 2,652 (0) | 0.6610 | **0.6253** | 0.6498 | 0.622 / 0.640 | 0.600 |
| **E4 (RO3 Combined)** | Retained (Text Tokens) | Balanced | 2,738 (87) | 0.6354 | **0.6368** | **0.6615** | **0.633 / 0.652** | **0.620** |

**Key Findings**:
- **RO2 (Class Weighting)**: `class_weight="balanced"` recovers minority Negative recall by **+20.0 to +22.0 percentage points** (`0.400 → 0.600 / 0.620`).
- **RO3 (Synergistic Interaction)**: Retaining emoji text tokens *without* class weighting (`E2`) slightly reduces Macro-F1 (`-0.0219`), whereas combining emoji text tokens *with* balanced class weighting (`E4`) produces a positive interaction (`+0.0334`), achieving the highest Test Macro-F1 (`0.6368`), highest minority Negative recall (`0.620`), and recovering peak overall Accuracy (`0.6615`).

---

## 4. Key Measured Dataset Statistics ($N = 1,712$)

- **Total Retrieved**: $1,821$ raw comments across $73$ candidate Malaysian YouTube videos ($52$ active videos across $6$ domains).
- **Total Rejected**: $109$ comments ($62$ empty/ultra-short, $42$ URL-only/non-Latin script, $5$ duplicates).
- **Final Valid Unique Samples**: **$1,712$** (`GD_0001` to `GD_1712`).
- **Natural Sentiment Class Distribution**:
  - **Neutral**: $880$ ($51.40\%$) — Train balanced weight: $0.6483$
  - **Positive**: $504$ ($29.44\%$) — Train balanced weight: $1.1313$
  - **Negative**: $328$ ($19.16\%$) — Train balanced weight: $1.7438$ (Majority-to-minority ratio: $2.68:1$)
- **Emoji Statistics**:
  - **With Emojis (`has_emoji = True`)**: $557$ comments ($32.54\%$), containing $1,326$ total emoji instances (mean $2.38$ emojis per emoji-bearing comment; max $26$).
  - **Without Emojis (`has_emoji = False`)**: $1,155$ comments ($67.46\%$).
  - **Emoji Presence by Class**: Positive $39.09\%$ ($197/504$), Negative $31.10\%$ ($102/328$), Neutral $29.32\%$ ($258/880$).
- **Inter-Annotator Reliability**: Raw agreement $= 94.63\%$ ($1,620 / 1,712$), **Cohen's Kappa $\kappa = 0.9131$**.
- **Automated Quality Assurance**: **12 / 12 checks PASS** (`reports/Dataset_QA_Report.txt`).

