"""
Phase 5 — Automated Dataset Quality Assurance & Validation Suite (validate_dataset.py)
Group D — STINK3114 Natural Language Processing (UUM)

Executes 12 automated quality assurance assertions across:
  - `dataset/raw_dataset.csv`
  - `dataset/group_d_sentiment_dataset.csv`
  - `dataset/train.csv`, `dataset/validation.csv`, `dataset/test.csv`
  - `dataset/split_manifest.csv`
  - `logs/collection_log.csv`

Generates `reports/Dataset_QA_Report.txt` documenting every check, measured metric,
and pass/fail status.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple
import pandas as pd

from preprocess import extract_emojis_and_emoticons

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"


def run_validation_suite() -> bool:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(DATASET_DIR / "raw_dataset.csv", encoding="utf-8")
    main_df = pd.read_csv(DATASET_DIR / "group_d_sentiment_dataset.csv", encoding="utf-8")
    train_df = pd.read_csv(DATASET_DIR / "train.csv", encoding="utf-8")
    val_df = pd.read_csv(DATASET_DIR / "validation.csv", encoding="utf-8")
    test_df = pd.read_csv(DATASET_DIR / "test.csv", encoding="utf-8")
    manifest_df = pd.read_csv(DATASET_DIR / "split_manifest.csv", encoding="utf-8")
    log_df = pd.read_csv(LOGS_DIR / "collection_log.csv", encoding="utf-8")

    results: List[Tuple[int, str, bool, str]] = []

    # Check 1: Minimum 1,000 final valid samples
    n_samples = len(main_df)
    c1_pass = n_samples >= 1000
    results.append(
        (1, "Minimum 1,000 final valid samples", c1_pass, f"Final valid samples = {n_samples} (Target: 1,500–2,000; Min: 1,000)")
    )

    # Check 2: Unique IDs
    n_unique_ids = main_df["id"].nunique()
    c2_pass = (n_unique_ids == n_samples) and main_df["id"].notna().all()
    results.append(
        (2, "Unique IDs across all records", c2_pass, f"Unique IDs = {n_unique_ids} / {n_samples}")
    )

    # Check 3: No duplicate raw comments
    norm_raw = main_df["raw_text"].astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    dup_raw_count = int(norm_raw.duplicated().sum())
    c3_pass = dup_raw_count == 0
    results.append(
        (3, "No duplicate raw comments", c3_pass, f"Duplicate normalized raw_text count = {dup_raw_count}")
    )

    # Check 4: No empty cleaned comments
    empty_a = int(main_df["cleaned_text_with_emojis"].isna().sum() + (main_df["cleaned_text_with_emojis"].astype(str).str.strip() == "").sum())
    empty_b = int(main_df["cleaned_text_no_emojis"].isna().sum() + (main_df["cleaned_text_no_emojis"].astype(str).str.strip() == "").sum())
    c4_pass = (empty_a == 0) and (empty_b == 0)
    results.append(
        (4, "No empty cleaned comments in Track A or Track B", c4_pass, f"Empty Track A = {empty_a}, Empty Track B = {empty_b}")
    )

    # Check 5: No missing sentiment labels
    missing_labels = int(main_df["label"].isna().sum() + (main_df["label"].astype(str).str.strip() == "").sum())
    c5_pass = missing_labels == 0
    results.append(
        (5, "No missing sentiment labels", c5_pass, f"Missing sentiment labels = {missing_labels}")
    )

    # Check 6: Valid sentiment classes only
    allowed_classes = {"Positive", "Negative", "Neutral"}
    observed_classes = set(main_df["label"].dropna().unique())
    invalid_classes = observed_classes - allowed_classes
    c6_pass = (len(invalid_classes) == 0) and (observed_classes == allowed_classes)
    results.append(
        (6, "Valid sentiment classes only (Positive, Negative, Neutral)", c6_pass, f"Observed classes = {sorted(observed_classes)}, Invalid = {sorted(invalid_classes)}")
    )

    # Check 7: Both experimental text columns exist
    c7_pass = ("cleaned_text_with_emojis" in main_df.columns) and ("cleaned_text_no_emojis" in main_df.columns)
    results.append(
        (7, "Both experimental text columns exist (Track A & Track B)", c7_pass, "Columns ['cleaned_text_with_emojis', 'cleaned_text_no_emojis'] verified")
    )

    # Check 8: Original emojis are preserved in Track A
    raw_emojis = main_df["raw_text"].apply(extract_emojis_and_emoticons)
    track_a_emojis = main_df["cleaned_text_with_emojis"].apply(extract_emojis_and_emoticons)
    mismatch_emoji_preservation = int((raw_emojis != track_a_emojis).sum())
    track_a_emoji_rows = int((track_a_emojis.apply(len) > 0).sum())
    c8_pass = (mismatch_emoji_preservation == 0) and (track_a_emoji_rows > 0)
    results.append(
        (8, "Original emojis preserved in Track A (cleaned_text_with_emojis)", c8_pass, f"Emoji-bearing rows = {track_a_emoji_rows}, Mismatches vs raw_text = {mismatch_emoji_preservation}")
    )

    # Check 9: Emoji removal is correctly implemented in Track B
    track_b_emojis = main_df["cleaned_text_no_emojis"].apply(extract_emojis_and_emoticons)
    remaining_emojis_in_b = int(track_b_emojis.apply(len).sum())
    c9_pass = remaining_emojis_in_b == 0
    results.append(
        (9, "Emoji removal strictly verified in Track B (cleaned_text_no_emojis)", c9_pass, f"Residual emojis/emoticons in Track B = {remaining_emojis_in_b}")
    )

    # Check 10: Both versions have matching IDs and labels & match raw_dataset.csv
    raw_id_match = (len(raw_df) == len(main_df)) and bool((raw_df["id"] == main_df["id"]).all())
    raw_lbl_match = bool((raw_df["label"] == main_df["label"]).all())
    non_emoji_diff = (
        main_df["cleaned_text_with_emojis"].apply(lambda x: " ".join(str(x).split()))
        != main_df["cleaned_text_no_emojis"].apply(lambda x: " ".join(str(x).split()))
    )
    # For rows without emojis, Track A and Track B must be character-for-character identical
    no_emoji_mask = ~main_df["has_emoji"]
    no_emoji_mismatch = int((main_df.loc[no_emoji_mask, "cleaned_text_with_emojis"] != main_df.loc[no_emoji_mask, "cleaned_text_no_emojis"]).sum())
    c10_pass = raw_id_match and raw_lbl_match and (no_emoji_mismatch == 0)
    results.append(
        (10, "Track A & Track B alignment and raw_dataset.csv synchronization", c10_pass, f"IDs aligned={raw_id_match}, Labels aligned={raw_lbl_match}, Non-emoji text diffs={no_emoji_mismatch}")
    )

    # Check 11: No accidental data leakage across train / validation / test splits
    train_ids = set(train_df["id"])
    val_ids = set(val_df["id"])
    test_ids = set(test_df["id"])
    id_overlap = len(train_ids & val_ids) + len(train_ids & test_ids) + len(val_ids & test_ids)
    total_split_rows = len(train_df) + len(val_df) + len(test_df)

    train_texts = set(train_df["cleaned_text_no_emojis"].str.lower().str.strip())
    val_texts = set(val_df["cleaned_text_no_emojis"].str.lower().str.strip())
    test_texts = set(test_df["cleaned_text_no_emojis"].str.lower().str.strip())
    text_overlap = len(train_texts & val_texts) + len(train_texts & test_texts) + len(val_texts & test_texts)
    manifest_match = (len(manifest_df) == n_samples) and (set(manifest_df["id"]) == set(main_df["id"]))
    c11_pass = (id_overlap == 0) and (text_overlap == 0) and (total_split_rows == n_samples) and manifest_match
    results.append(
        (11, "No accidental data leakage across Train/Validation/Test splits", c11_pass, f"ID overlap={id_overlap}, Text cross-split overlap={text_overlap}, Total split rows={total_split_rows}/{n_samples}, Manifest synced={manifest_match}")
    )

    # Check 12: All collection and filtering counts are reproducible
    tot_retrieved = int(log_df["retrieved_count"].sum())
    tot_rejected = int(log_df["rejected_total"].sum())
    tot_retained = int(log_df["retained_count"].sum())
    row_math_ok = bool(((log_df["retrieved_count"] - log_df["rejected_total"]) == log_df["retained_count"]).all())
    c12_pass = row_math_ok and (tot_retained == n_samples) and (tot_retrieved - tot_rejected == n_samples)
    results.append(
        (12, "Collection and filtering counts are 100% reproducible", c12_pass, f"Retrieved ({tot_retrieved}) - Rejected ({tot_rejected}) = Retained ({tot_retained}) == Dataset Size ({n_samples})")
    )

    all_passed = all(r[2] for r in results)
    ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    report_lines = [
        "=" * 82,
        "GROUP D — AUTOMATED DATASET QUALITY ASSURANCE (QA) REPORT",
        "Course  : STINK3114 Natural Language Processing (Session I 2026/2027 - A261)",
        f"Date    : {ts_str}",
        f"Dataset : group_d_sentiment_dataset.csv (N = {n_samples})",
        f"Status  : {'ALL 12 QUALITY ASSURANCE CHECKS PASSED (12/12)' if all_passed else 'VALIDATION FAILURE DETECTED'}",
        "=" * 82,
        "",
        "SUMMARY OF AUTOMATED VALIDATION TESTS:",
        "-" * 82,
    ]

    for num, title, passed, detail in results:
        status_str = "PASS" if passed else "FAIL"
        report_lines.append(f"[{status_str}] Check {num:02d}: {title}")
        report_lines.append(f"       Evidence: {detail}")
        report_lines.append("")

    report_lines.extend(
        [
            "-" * 82,
            "SPLIT INTEGRITY & EXPERIMENTAL READINESS VERIFICATION:",
            f"  - Total Valid Samples  : {n_samples}",
            f"  - Training Split (70%) : {len(train_df)} samples ({len(train_df)/n_samples*100:.2f}%)",
            f"  - Validation Split(15%): {len(val_df)} samples ({len(val_df)/n_samples*100:.2f}%)",
            f"  - Testing Split (15%)  : {len(test_df)} samples ({len(test_df)/n_samples*100:.2f}%)",
            f"  - Cross-Split ID Leak  : {id_overlap} records",
            f"  - Cross-Split Text Leak: {text_overlap} records",
            "=" * 82,
        ]
    )

    qa_report_path = REPORTS_DIR / "Dataset_QA_Report.txt"
    qa_report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n".join(report_lines))
    return all_passed


if __name__ == "__main__":
    ok = run_validation_suite()
    if not ok:
        raise SystemExit(1)
