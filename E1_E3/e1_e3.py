"""
Assignment 3B - E1 and E3 (Emoji Removed)

Files needed in the SAME folder as this script:
    e1_e3.py
    train.csv
    validation.csv
    test.csv

Install: pip install pandas scikit-learn
Run:     python e1_e3.py

E1 = Logistic Regression without class balancing.
E3 = Same model, with class_weight='balanced'.
Uses your teammate's existing splits; it never creates new splits.
"""

from pathlib import Path

import pandas as pd
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

# 1. Find the 3 CSV files beside this Python script.
FOLDER = Path(__file__).resolve().parent
RESULTS = FOLDER / "results"
RESULTS.mkdir(exist_ok=True)
TEXT_COLUMN = "cleaned_text_no_emojis"
LABEL_COLUMN = "label"

train = pd.read_csv(FOLDER / "train.csv", encoding="utf-8-sig")
validation = pd.read_csv(FOLDER / "validation.csv", encoding="utf-8-sig")
test = pd.read_csv(FOLDER / "test.csv", encoding="utf-8-sig")

# 2. Simple safety checks: correct columns and no repeated IDs across splits.
for split_name, df in [("train", train), ("validation", validation), ("test", test)]:
    for col in ["id", TEXT_COLUMN, LABEL_COLUMN]:
        if col not in df.columns:
            raise ValueError(f"{split_name}.csv is missing column: {col}")
    if df[["id", TEXT_COLUMN, LABEL_COLUMN]].isna().any().any():
        raise ValueError(f"{split_name}.csv contains missing IDs, text or labels")

train_ids = set(train["id"])
validation_ids = set(validation["id"])
test_ids = set(test["id"])
if (train_ids & validation_ids) or (train_ids & test_ids) or (validation_ids & test_ids):
    raise ValueError("The CSV files contain repeated IDs across splits (data leakage)!")

print(f"Data loaded: Train = {len(train)}, Validation = {len(validation)}, Test = {len(test)}")

# 3. TF-IDF converts text into numbers.
#    Learn words from TRAINING data only, never from validation/test.
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    lowercase=True
)

X_train = tfidf.fit_transform(train[TEXT_COLUMN])
X_validation = tfidf.transform(validation[TEXT_COLUMN])
X_test = tfidf.transform(test[TEXT_COLUMN])

y_train = train[LABEL_COLUMN]
y_validation = validation[LABEL_COLUMN]
y_test = test[LABEL_COLUMN]

# 4. Run E1 and E3. The ONLY difference is class_weight.
all_results = []
for experiment, class_weight in [("E1", None), ("E3", "balanced")]:
    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_train, y_train)  # A NEW model is trained from scratch each time.

    # 5. Check the model on validation and test data.
    for split_name, X, true_labels in [
        ("validation", X_validation, y_validation),
        ("test", X_test, y_test),
    ]:
        predictions = model.predict(X)
        accuracy = accuracy_score(true_labels, predictions)
        macro_f1 = f1_score(true_labels, predictions, average="macro", zero_division=0)
        precision = precision_score(true_labels, predictions, average="macro", zero_division=0)
        recall = recall_score(true_labels, predictions, average="macro", zero_division=0)
        all_results.append({
            "experiment": experiment,
            "emoji": "removed",
            "class_weight": "none" if class_weight is None else "balanced",
            "split": split_name,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "precision": precision,
            "recall": recall,
        })
        print(
            f"{experiment} - {split_name}: Accuracy = {accuracy:.4f}, Macro-F1 = {macro_f1:.4f}, "
            f"Precision = {precision:.4f}, Recall = {recall:.4f}"
        )

        # Store detailed TEST results for the later Assignment 3C.
        if split_name == "test":
            pd.DataFrame({
                "id": test["id"],
                "actual_label": true_labels,
                "predicted_label": predictions,
            }).to_csv(RESULTS / f"{experiment}_test_predictions.csv", index=False, encoding="utf-8-sig")

            classes = ["Negative", "Neutral", "Positive"]
            matrix = confusion_matrix(true_labels, predictions, labels=classes)
            pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
                RESULTS / f"{experiment}_confusion_matrix.csv", encoding="utf-8-sig"
            )
            report = classification_report(true_labels, predictions, zero_division=0)
            (RESULTS / f"{experiment}_classification_report.txt").write_text(
                report, encoding="utf-8"
            )

# 6. Save a simple comparison table for E1 versus E3.
pd.DataFrame(all_results).to_csv(RESULTS / "E1_E3_metrics.csv", index=False)
print("\nDone! Open the 'results' folder to see the output.")
