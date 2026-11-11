"""
Phase 7 — Stratified Train / Validation / Test Splitting Pipeline (create_splits.py)
Group D — STINK3114 Natural Language Processing (UUM)

Prepares the final dataset for RO1, RO2, and RO3 experimental evaluation using a
reproducible 70% / 15% / 15% stratified split (`random_state=42`), ensuring:
  - Both emoji versions (`cleaned_text_with_emojis` and `cleaned_text_no_emojis`)
    of each comment remain strictly within the same split.
  - Natural class distribution AND emoji-presence distribution are preserved across
    Training (70%), Validation (15%), and Testing (15%) via composite stratification
    (`label` × `has_emoji`).
  - Zero duplicate comments or ID leakage across splits.
  - Computes inverse-frequency balanced class weights from the Training split ONLY
    (preventing test/validation distribution leakage) to support Experiments E1–E4.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
LOGS_DIR = BASE_DIR / "logs"

RANDOM_SEED = 42


def run_split_pipeline(
    input_csv: Path = DATASET_DIR / "group_d_sentiment_dataset.csv",
    train_csv: Path = DATASET_DIR / "train.csv",
    val_csv: Path = DATASET_DIR / "validation.csv",
    test_csv: Path = DATASET_DIR / "test.csv",
    manifest_csv: Path = DATASET_DIR / "split_manifest.csv",
    exp_config_json: Path = LOGS_DIR / "experimental_readiness_config.json",
) -> None:
    df = pd.read_csv(input_csv, encoding="utf-8")

    # Composite stratification key: preserves both sentiment class proportions (RO2)
    # and emoji presence proportions (RO1 & RO3) identically across train/val/test.
    strat_key = df["label"].astype(str) + "__emoji_" + df["has_emoji"].astype(str)

    # First split: 70% Train, 30% Temp (Validation + Test)
    train_df, temp_df, _, temp_strat = train_test_split(
        df,
        strat_key,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=strat_key,
    )

    # Second split: divide the 30% Temp equally into 15% Validation and 15% Test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_strat,
    )

    train_df = train_df.sort_values("id").reset_index(drop=True)
    val_df = val_df.sort_values("id").reset_index(drop=True)
    test_df = test_df.sort_values("id").reset_index(drop=True)

    train_df.to_csv(train_csv, index=False, encoding="utf-8")
    val_df.to_csv(val_csv, index=False, encoding="utf-8")
    test_df.to_csv(test_csv, index=False, encoding="utf-8")

    # Build split_manifest.csv
    id_to_split = {}
    for i in train_df["id"]:
        id_to_split[i] = "train"
    for i in val_df["id"]:
        id_to_split[i] = "validation"
    for i in test_df["id"]:
        id_to_split[i] = "test"

    manifest_df = df[["id", "source_id", "label", "has_emoji", "language"]].copy()
    manifest_df["split"] = manifest_df["id"].map(id_to_split)
    manifest_df["random_seed"] = RANDOM_SEED
    manifest_df["track_a_column"] = "cleaned_text_with_emojis"
    manifest_df["track_b_column"] = "cleaned_text_no_emojis"
    manifest_df["supported_experiments"] = "E1,E2,E3,E4,FocalLoss"
    manifest_df.to_csv(manifest_csv, index=False, encoding="utf-8")

    # Compute class weights strictly on the Training split (to prevent data leakage)
    classes = np.array(["Positive", "Neutral", "Negative"])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["label"].values,
    )
    class_weight_dict = {cls: round(float(w), 4) for cls, w in zip(classes, weights)}

    exp_summary = {
        "random_seed": RANDOM_SEED,
        "split_ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "split_counts": {
            "total": int(len(df)),
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "split_sentiment_distribution": {
            "train": train_df["label"].value_counts().to_dict(),
            "validation": val_df["label"].value_counts().to_dict(),
            "test": test_df["label"].value_counts().to_dict(),
        },
        "split_emoji_counts": {
            "train": int(train_df["has_emoji"].sum()),
            "validation": int(val_df["has_emoji"].sum()),
            "test": int(test_df["has_emoji"].sum()),
        },
        "train_only_balanced_class_weights": class_weight_dict,
        "experimental_matrix": {
            "E1": {
                "text_column": "cleaned_text_with_emojis",
                "emoji_handling": "Retained",
                "imbalance_handling": "No class weighting (Standard Cross-Entropy)",
                "research_objective": "RO1 baseline / RO3 control",
            },
            "E2": {
                "text_column": "cleaned_text_with_emojis",
                "emoji_handling": "Retained",
                "imbalance_handling": "Balanced Class Weighting / Focal Loss",
                "class_weights": class_weight_dict,
                "research_objective": "RO2 & RO3 combined interaction",
            },
            "E3": {
                "text_column": "cleaned_text_no_emojis",
                "emoji_handling": "Stripped",
                "imbalance_handling": "No class weighting (Standard Cross-Entropy)",
                "research_objective": "RO1 comparison baseline",
            },
            "E4": {
                "text_column": "cleaned_text_no_emojis",
                "emoji_handling": "Stripped",
                "imbalance_handling": "Balanced Class Weighting / Focal Loss",
                "class_weights": class_weight_dict,
                "research_objective": "RO2 text-only & RO3 ablation",
            },
        },
    }
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(exp_config_json, "w", encoding="utf-8") as f:
        json.dump(exp_summary, f, indent=2)

    print("=" * 70)
    print(f"[Phase 7 Complete] Total={len(df)} -> Train={len(train_df)} (70.0%), Val={len(val_df)} (15.0%), Test={len(test_df)} (15.0%)")
    print(f"[Phase 7 Complete] Train Class Weights (from Train only): {class_weight_dict}")
    print(f"[Phase 7 Complete] Saved split manifest to: {manifest_csv}")
    print("=" * 70)


if __name__ == "__main__":
    run_split_pipeline()

