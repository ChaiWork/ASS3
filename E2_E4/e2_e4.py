"""
Assignment 3B - E2 and E4 (Emoji Retained - Converted to Text Tokens)

Files needed in the SAME folder as this script:
    e2_e4.py
    train.csv
    validation.csv
    test.csv

Install: pip install pandas scikit-learn
Run:     python e2_e4.py

E2 = Logistic Regression without class balancing (emojis retained as text tokens).
E4 = Same model, with class_weight='balanced' (emojis retained as text tokens).
Uses the exact same splits, TF-IDF settings, and Logistic Regression parameters as e1_e3.py
so E1 vs E2 vs E3 vs E4 is a fair, controlled 2x2 comparison.
"""

import re
import unicodedata
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
TEXT_COLUMN = "cleaned_text_with_emojis"
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

# 3. Convert Unicode emojis and text emoticons into alphanumeric text tokens.
#    Standard TfidfVectorizer drops emoji symbols by default; converting them to
#    tokens like `emoji_smiling_face_with_smiling_eyes` allows TF-IDF to keep them.
UNICODE_EMOJI_REGEX = re.compile(
    r"(?:"
    r"[\U0001F1E6-\U0001F1FF]{2}"
    r"|"
    r"["
    r"\U0001F300-\U0001F3FA\U0001F400-\U0001F5FF\U0001F600-\U0001F64F"
    r"\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF"
    r"\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F"
    r"\U0001FA70-\U0001FAFF\U00002600-\U000026FF\U00002700-\U000027BF"
    r"\U0000231A-\U0000231B\U000023E9-\U000023FA\U000025AA-\U000025FE"
    r"\U00002934-\U00002935\U00002B05-\U00002B55\U00003030\U0000303D"
    r"\U00003297\U00003299"
    r"](?:[\U0001F3FB-\U0001F3FF\uFE0F\u200D])*"
    r")"
)
EMOJI_MODIFIERS_REGEX = re.compile(r"[\uFE0E\uFE0F\u200D\u20E3\U0001F3FB-\U0001F3FF\U0001F1E6-\U0001F1FF]+")
EMOTICON_REGEX = re.compile(
    r"(?<!\S)(?::-\)|:\)|:-\(|:\(|:-D|:D|;-D|;D|;-\)|;\)|:-P|:P|:-p|:p|"
    r":'\(|:'-\(|:-/|:/|:-\\|:\\|:-O|:O|:-o|:o|<3|</3|T_T|ToT|T\.T|>_<|\^_\^|-_-|=D|=\)|=\()(?!\S)"
)
FLAG_NAMES = {
    "MY": "malaysia", "ID": "indonesia", "SG": "singapore", "US": "usa",
    "GB": "uk", "PS": "palestine", "IL": "israel", "BR": "brazil",
}
EMOTICON_MAP = {
    ":)": "emoticon_smile", ":-)": "emoticon_smile", "=)": "emoticon_smile",
    ":(": "emoticon_sad", ":-(": "emoticon_sad", "=(": "emoticon_sad",
    ":D": "emoticon_laugh", ":-D": "emoticon_laugh", "=D": "emoticon_laugh",
    ";D": "emoticon_wink_laugh", ";-D": "emoticon_wink_laugh",
    ";)": "emoticon_wink", ";-)": "emoticon_wink",
    ":P": "emoticon_tongue", ":-P": "emoticon_tongue", ":p": "emoticon_tongue", ":-p": "emoticon_tongue",
    ":'(": "emoticon_cry", ":'-(": "emoticon_cry",
    ":/": "emoticon_skeptical", ":-/": "emoticon_skeptical", ":\\": "emoticon_skeptical", ":-\\": "emoticon_skeptical",
    ":O": "emoticon_surprised", ":-O": "emoticon_surprised", ":o": "emoticon_surprised", ":-o": "emoticon_surprised",
    "<3": "emoticon_heart", "</3": "emoticon_broken_heart",
    "T_T": "emoticon_crying", "ToT": "emoticon_crying", "T.T": "emoticon_crying",
    ">_<": "emoticon_frustrated", "^_^": "emoticon_happy", "-_-": "emoticon_unamused",
}


def convert_emojis_to_tokens(text: str) -> str:
    def _replace_unicode(match: re.Match) -> str:
        core = re.sub(r"[\uFE0E\uFE0F\u200D\U0001F3FB-\U0001F3FF]", "", match.group(0))
        if not core:
            return " "
        if len(core) == 2 and all(0x1F1E6 <= ord(c) <= 0x1F1FF for c in core):
            iso = "".join(chr(ord("A") + (ord(c) - 0x1F1E6)) for c in core)
            return f" emoji_flag_{FLAG_NAMES.get(iso, iso.lower())} "
        uname = unicodedata.name(core[0], f"u{ord(core[0]):04x}")
        slug = re.sub(r"[^a-z0-9]+", "_", uname.lower()).strip("_")
        return f" emoji_{slug} "

    def _replace_emoticon(match: re.Match) -> str:
        return f" {EMOTICON_MAP.get(match.group(0), 'emoticon_other')} "

    out = UNICODE_EMOJI_REGEX.sub(_replace_unicode, str(text))
    out = EMOJI_MODIFIERS_REGEX.sub(" ", out)
    out = EMOTICON_REGEX.sub(_replace_emoticon, out)
    return re.sub(r"\s+", " ", out).strip()


train_text = train[TEXT_COLUMN].apply(convert_emojis_to_tokens)
validation_text = validation[TEXT_COLUMN].apply(convert_emojis_to_tokens)
test_text = test[TEXT_COLUMN].apply(convert_emojis_to_tokens)

# 4. TF-IDF converts text (including emoji tokens) into numbers.
#    Learn words from TRAINING data only, never from validation/test.
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    lowercase=True,
)

X_train = tfidf.fit_transform(train_text)
X_validation = tfidf.transform(validation_text)
X_test = tfidf.transform(test_text)

y_train = train[LABEL_COLUMN]
y_validation = validation[LABEL_COLUMN]
y_test = test[LABEL_COLUMN]

# 5. Run E2 and E4. The ONLY difference is class_weight.
all_results = []
for experiment, class_weight in [("E2", None), ("E4", "balanced")]:
    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_train, y_train)  # A NEW model is trained from scratch each time.

    # 6. Check the model on validation and test data.
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
            "emoji": "retained",
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

# 7. Save a simple comparison table for E2 versus E4.
pd.DataFrame(all_results).to_csv(RESULTS / "E2_E4_metrics.csv", index=False)
print("\nDone! Open the 'results' folder to see the output.")
