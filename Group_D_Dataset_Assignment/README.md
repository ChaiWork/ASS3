# Group D — Multilingual Malaysian Social Media Sentiment Dataset (STINK3114)

**Course**: STINK3114 Natural Language Processing (Session I 2026/2027 - A261)  
**Assignment**: Assignment 3 — Dataset Design & Collection (Week 6–7)  
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
│   └── generate_reports.py              # Phase 8: Automated DOCX & PDF report builder
│
├── reports/
│   ├── Dataset_Description_Group_D.docx # Editable academic report (2–4 pages)
│   ├── Dataset_Description_Group_D.pdf  # Submission-ready PDF academic report
│   ├── Dataset_QA_Report.txt            # Automated validation report (12/12 PASS)
│   └── Annotation_Guidelines_Group_D.md # 3-class annotation guide & decision rules
│
├── figures/
│   ├── sentiment_distribution.png       # Fig 1: Class distribution & split preservation
│   ├── emoji_distribution.png           # Fig 2: Emoji presence & Top 10 emojis
│   ├── text_length_distribution.png     # Fig 3: Word & character length histograms
│   └── emoji_by_sentiment.png           # Fig 4: Emoji density & presence by sentiment class
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

# Step 7: Build editable DOCX and submission-ready PDF academic reports
python src/generate_reports.py
```

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

