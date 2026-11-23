"""
Working NLP Sentiment Classification Prototype (predict_prototype.py)
Group D — STINK3114 Natural Language Processing (UUM)

Provides a working, reproducible inference prototype loading the trained TF-IDF +
Logistic Regression pipelines (E1–E4) to classify live English, Malay, and Malaysian
Manglish social media comments, comparing Baseline (E1: Emoji Removed + Unweighted)
against Combined (E4: Emoji Retained as Text Tokens + Balanced Class Weights).

Usage:
    # Single comment prediction:
    python src/predict_prototype.py --text "Nasi lemak ni sedap gila bro 😍🔥"

    # Built-in multilingual demo suite:
    python src/predict_prototype.py --demo

    # Interactive CLI loop:
    python src/predict_prototype.py
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import basic_clean_text, strip_emojis
from emoji_tokenizer import convert_emojis_to_tokens

MODEL_FILES = {
    "E1": ("Emoji Removed + None (Baseline)", MODELS_DIR / "E1_emoji_removed_none.joblib", False),
    "E2": ("Emoji Retained + None (RO1)", MODELS_DIR / "E2_emoji_retained_none.joblib", True),
    "E3": ("Emoji Removed + Balanced (RO2)", MODELS_DIR / "E3_emoji_removed_balanced.joblib", False),
    "E4": ("Emoji Retained + Balanced (RO3)", MODELS_DIR / "E4_emoji_retained_balanced.joblib", True),
}

DEMO_COMMENTS = [
    "Makanan kat bazar ni sedap gila bro, harga pun berbaloi 😍🔥👍",
    "RFID tol asyik sangkut je, jalan jem teruk gila tiap hari 😭😡",
    "Anwar dah tak ada rasa malu sebab kuat menipu 😂😂😂😂😂",
    "Menteri Luar Negeri? Bukannya Menteri Luar Negara? 😂",
    "I from San Marino, terbaik Malaysia!! 😊🇲🇾",
    "Servis kurier lambat sangat, barang rosak bila sampai 💔👎",
]


def load_models() -> Dict:
    loaded = {}
    for exp_id, (desc, path, use_emoji) in MODEL_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {path}. Run `python src/train_evaluate_models.py` first."
            )
        loaded[exp_id] = (desc, joblib.load(path), use_emoji)
    return loaded


def predict_comment(raw_text: str, models: Dict) -> Dict:
    """Run identical preprocessing + dual-track transformation and predict across E1–E4."""
    cleaned_with_emojis = basic_clean_text(raw_text)
    cleaned_no_emojis = strip_emojis(cleaned_with_emojis)
    emoji_tokenized = convert_emojis_to_tokens(cleaned_with_emojis)

    predictions = {}
    for exp_id, (desc, pipe, use_emoji) in models.items():
        model_input = emoji_tokenized if use_emoji else cleaned_no_emojis
        pred_label = str(pipe.predict([model_input])[0])
        probs = pipe.predict_proba([model_input])[0]
        prob_map = {str(cls): round(float(p), 4) for cls, p in zip(pipe.classes_, probs)}
        predictions[exp_id] = {
            "description": desc,
            "predicted_label": pred_label,
            "probabilities": prob_map,
        }

    return {
        "raw_text": raw_text,
        "track_b_no_emojis": cleaned_no_emojis,
        "track_a_emoji_tokens": emoji_tokenized,
        "predictions": predictions,
    }


def print_prediction_card(res: Dict) -> None:
    print("-" * 80)
    print(f"Input Comment        : {res['raw_text']}")
    print(f"Track B (No Emojis)  : {res['track_b_no_emojis']}")
    print(f"Track A (Emoji Token): {res['track_a_emoji_tokens']}")
    print("Model Predictions:")
    for exp_id in ["E1", "E2", "E3", "E4"]:
        p = res["predictions"][exp_id]
        probs = p["probabilities"]
        print(
            f"  [{exp_id}] {p['description']:33s} -> {p['predicted_label']:8s} "
            f"(Pos={probs.get('Positive',0):.3f}, Neu={probs.get('Neutral',0):.3f}, Neg={probs.get('Negative',0):.3f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Group D Multilingual Sentiment Prototype (E1–E4)")
    parser.add_argument("--text", type=str, default=None, help="Single social media comment to classify")
    parser.add_argument("--demo", action="store_true", help="Run built-in multilingual Malaysian demo examples")
    args = parser.parse_args()

    models = load_models()

    if args.text:
        print_prediction_card(predict_comment(args.text, models))
        return

    if args.demo:
        print("=" * 80)
        print("GROUP D — WORKING NLP SENTIMENT PROTOTYPE DEMO (TF-IDF + LOGISTIC REGRESSION)")
        print("=" * 80)
        for sample in DEMO_COMMENTS:
            print_prediction_card(predict_comment(sample, models))
        print("-" * 80)
        return

    print("=" * 80)
    print("Group D Interactive Sentiment Prototype (type 'quit' to exit)")
    print("=" * 80)
    while True:
        try:
            user_text = input("\nEnter a comment (EN/MS/Manglish + Emojis): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_text or user_text.lower() in {"quit", "exit", "q"}:
            break
        print_prediction_card(predict_comment(user_text, models))


if __name__ == "__main__":
    main()
