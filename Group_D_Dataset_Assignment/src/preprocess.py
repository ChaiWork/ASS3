"""
Phase 4 — Modular Data Preprocessing Pipeline (preprocess.py)
Group D — STINK3114 Natural Language Processing (UUM)

Implements identical non-destructive basic cleaning for social media comments and
branches each cleaned comment into two strictly aligned experimental tracks:
  - Track A (`cleaned_text_with_emojis`): Preserves all Unicode emojis and emoticons.
  - Track B (`cleaned_text_no_emojis`): Strips all Unicode emojis and emoticons while
    preserving lexical tokens, casing, negation words, and expressive punctuation.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"

# Comprehensive Unicode emoji regex:
# - Matches 2-character Regional Indicator flag sequences (e.g. 🇲🇾) as a single unit
# - Attaches optional skin-tone modifiers (U+1F3FB..U+1F3FF) and variation selector-16 (U+FE0F)
UNICODE_EMOJI_REGEX = re.compile(
    r"(?:"
    r"[\U0001F1E6-\U0001F1FF]{2}"  # Regional indicator flag pairs (e.g. 🇲🇾)
    r"|"
    r"["
    r"\U0001F300-\U0001F3FA"
    r"\U0001F400-\U0001F5FF"  # Misc Symbols and Pictographs (excluding bare skin tones)
    r"\U0001F600-\U0001F64F"  # Emoticons
    r"\U0001F680-\U0001F6FF"  # Transport and Map
    r"\U0001F700-\U0001F77F"
    r"\U0001F780-\U0001F7FF"
    r"\U0001F800-\U0001F8FF"
    r"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    r"\U0001FA00-\U0001FA6F"
    r"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    r"\U00002600-\U000026FF"  # Misc symbols
    r"\U00002700-\U000027BF"  # Dingbats
    r"\U0000231A-\U0000231B"
    r"\U000023E9-\U000023FA"
    r"\U000025AA-\U000025FE"
    r"\U00002934-\U00002935"
    r"\U00002B05-\U00002B55"
    r"\U00003030\U0000303D\U00003297\U00003299"
    r"](?:[\U0001F3FB-\U0001F3FF\uFE0F\u200D])*"
    r")"
)

# Trailing or standalone emoji modifiers (variation selectors, ZWJ, skin tones,孤立 regional indicators)
EMOJI_MODIFIERS_REGEX = re.compile(r"[\uFE0E\uFE0F\u200D\u20E3\U0001F3FB-\U0001F3FF\U0001F1E6-\U0001F1FF]+")

# Documented ASCII/Western/Asian text emoticons (bounded by whitespace or start/end of string)
EMOTICON_REGEX = re.compile(
    r"(?<!\S)(?:"
    r":-\)|:\)|:-\(|:\(|:-D|:D|;-D|;D|;-\)|;\)|:-P|:P|:-p|:p|"
    r":'\(|:'-\(|:-/|:/|:-\\|:\\|:-O|:O|:-o|:o|<3|</3|"
    r"T_T|ToT|T\.T|>_<|\^_\^|-_-|=D|=\)|=\("
    r")(?!\S)"
)

URL_REGEX = re.compile(r"(?:https?://\S+|www\.\S+)", re.IGNORECASE)
MENTION_REGEX = re.compile(r"@[A-Za-z0-9_.-]+")
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Lexical sets for heuristic language identification (Malay vs English vs Manglish)
MALAY_MARKERS = {
    "aku", "saya", "kau", "awak", "dia", "mereka", "kita", "kami", "korang", "anda",
    "ini", "itu", "ni", "tu", "yang", "yg", "dan", "atau", "tapi", "tetapi", "kalau",
    "jika", "kerana", "sebab", "sbb", "untuk", "utk", "dengan", "dgn", "dalam", "dlm",
    "pada", "dari", "daripada", "kepada", "ke", "di", "kat", "kt", "ada", "tiada",
    "tak", "tidak", "bukan", "jangan", "jgn", "belum", "sudah", "dah", "dh", "akan",
    "boleh", "blh", "dapat", "dpt", "kena", "perlu", "mahu", "nak", "nk", "macam",
    "mcm", "lebih", "sangat", "sgt", "paling", "memang", "mmg", "banyak", "byk",
    "semua", "orang", "org", "rakyat", "negara", "kerajaan", "harga", "barang",
    "makanan", "makan", "minum", "sedap", "terbaik", "bagus", "elok", "cantik",
    "teruk", "buruk", "mahal", "murah", "lambat", "cepat", "susah", "senang",
    "masalah", "rosak", "sesak", "jalan", "kereta", "tol", "minyak", "subsidi",
    "banjir", "hujan", "바ru", "baru", "lama", "hari", "tahun", "bulan", "masa",
    "bila", "mana", "apa", "siapa", "kenapa", "mengapa", "bagaimana", "camne",
    "betul", "salah", "faham", "tahu", "tau", "lihat", "tengok", "tgk", "cakap",
    "kata", "buat", "kerja", "beli", "jual", "bayar", "duit", "ringgit", "malaysia",
    "berita", "filem", "cerita", "lagu", "bola", "menang", "kalah", "main", "pasukan",
    "sekali", "lagi", "pun", "juga", "jugak", "je", "jer", "sahaja", "saja", "pula",
    "pulak", "lah", "la", "kan", "weh", "wei", "kot", "sia", "doh", "padu", "gila",
    "x", "xde", "takde", "xleh", "takleh", "xkan", "abg", "kak", "pak", "mak", "cik",
}

ENGLISH_MARKERS = {
    "the", "and", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "shall", "should", "can", "could",
    "may", "might", "must", "this", "that", "these", "those", "it", "its", "they",
    "them", "their", "we", "us", "our", "you", "your", "he", "him", "his", "she",
    "her", "my", "me", "i", "in", "on", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above", "below",
    "from", "up", "down", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "just", "now", "also",
    "good", "great", "best", "better", "bad", "worst", "worse", "nice", "love",
    "like", "hate", "really", "actually", "literally", "always", "never", "still",
    "even", "much", "many", "people", "time", "year", "way", "day", "thing", "man",
    "world", "life", "hand", "part", "child", "eye", "woman", "place", "work",
    "week", "case", "point", "government", "company", "number", "group", "problem",
    "fact", "money", "price", "food", "service", "phone", "video", "movie", "game",
    "watch", "watching", "think", "know", "want", "need", "feel", "look", "make",
    "made", "take", "come", "go", "going", "get", "got", "use", "using", "used",
    "buy", "bought", "worth", "expensive", "cheap", "slow", "fast", "issue", "update",
}

MANGLISH_PARTICLES_AND_SLANG = {
    "lah", "la", "leh", "lor", "meh", "hor", "maa", "wei", "weh", "walao", "alamak",
    "aiyo", "aiyoh", "bojio", "tapau", "mamak", "bossku", "padu", "sempoi", "gempak",
    "koyak", "rabak", "pishang", "syok", "bestnya", "terbaiklah", "mantap", "sia",
    "doh", "kot", "je", "jer", "kan", "xde", "takde", "sbb", "mcm", "nk", "dh",
}


def extract_emojis_and_emoticons(text: str) -> List[str]:
    """Return list of all Unicode emojis and text emoticons present in `text`."""
    if not isinstance(text, str) or not text:
        return []
    emojis = UNICODE_EMOJI_REGEX.findall(text)
    emoticons = EMOTICON_REGEX.findall(text)
    return emojis + emoticons


def basic_clean_text(raw_text: str) -> str:
    """
    Step A: Basic non-destructive cleaning shared identically across Track A and Track B.
    - Removes URLs and email addresses.
    - Anonymizes @username mentions to `@USER`.
    - Normalizes line breaks and excessive whitespace.
    - Preserves original casing, negation tokens, repeated characters, and punctuation.
    """
    if not isinstance(raw_text, str):
        return ""
    text = EMAIL_REGEX.sub("[EMAIL]", raw_text)
    text = URL_REGEX.sub("", text)
    text = MENTION_REGEX.sub("@USER", text)
    # Remove zero-width non-joiner / BOM / control chars except standard space
    text = re.sub(r"[\u200B\u200C\uFEFF\x00-\x1F\x7F]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_emojis(cleaned_text_with_emojis: str) -> str:
    """
    Step B (Track B): Remove all Unicode emojis, emoji modifiers, and documented emoticons
    from `cleaned_text_with_emojis` while preserving all words and punctuation.
    """
    if not isinstance(cleaned_text_with_emojis, str):
        return ""
    text = UNICODE_EMOJI_REGEX.sub("", cleaned_text_with_emojis)
    text = EMOJI_MODIFIERS_REGEX.sub("", text)
    text = EMOTICON_REGEX.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def identify_language(text_no_emojis: str) -> str:
    """
    Categorize comment language into `English`, `Malay`, `Manglish`, or `Other/Uncertain`.
    Note: Automated lexical language identification on short informal social media text
    is an approximation, as code-switched Malaysian Manglish shares loanwords with both
    Bahasa Melayu and English.
    """
    tokens = re.findall(r"[A-Za-z']+", str(text_no_emojis).lower())
    tokens = [t for t in tokens if t != "user"]
    if not tokens:
        return "Other/Uncertain"

    ms_hits = sum(1 for t in tokens if t in MALAY_MARKERS)
    en_hits = sum(1 for t in tokens if t in ENGLISH_MARKERS)
    manglish_hits = sum(1 for t in tokens if t in MANGLISH_PARTICLES_AND_SLANG)

    # Code-switching (both Malay and English markers present) or English + Manglish particles
    if ms_hits >= 1 and en_hits >= 1:
        return "Manglish"
    if en_hits >= 1 and manglish_hits >= 1:
        return "Manglish"
    if ms_hits >= 2 and manglish_hits >= 1 and any(t in {"bro", "sis", "boss", "guys", "admin", "min", "best", "nice", "ok", "try", "test", "support", "level", "padu"} for t in tokens):
        return "Manglish"
    if ms_hits > 0 and en_hits == 0:
        return "Malay"
    if en_hits > 0 and ms_hits == 0:
        return "English"
    return "Other/Uncertain"


def run_preprocessing_pipeline(
    input_csv: Path = DATASET_DIR / "raw_dataset.csv",
    output_csv: Path = DATASET_DIR / "group_d_sentiment_dataset.csv",
) -> pd.DataFrame:
    df = pd.read_csv(input_csv, encoding="utf-8")
    initial_rows = len(df)

    # Drop any missing raw_text or duplicate raw_text
    df = df.dropna(subset=["raw_text"]).copy()
    df["raw_text"] = df["raw_text"].astype(str).str.strip()
    df = df[df["raw_text"] != ""].copy()
    df["dedup_key"] = df["raw_text"].str.lower().str.replace(r"\s+", " ", regex=True)
    df = df.drop_duplicates(subset=["dedup_key"], keep="first").drop(columns=["dedup_key"]).copy()

    # Step A & B: Generate Track A and Track B from the exact same basic-cleaned text
    df["cleaned_text_with_emojis"] = df["raw_text"].apply(basic_clean_text)
    df["cleaned_text_no_emojis"] = df["cleaned_text_with_emojis"].apply(strip_emojis)

    # Exclude any sample where either Track A or Track B becomes empty (e.g., pure @mention + emoji)
    # or has no alphanumeric characters in Track B
    valid_mask = (
        (df["cleaned_text_with_emojis"].str.len() > 0)
        & (df["cleaned_text_no_emojis"].str.len() > 0)
        & (df["cleaned_text_no_emojis"].str.contains(r"[A-Za-z0-9]", regex=True))
    )
    excluded_after_clean = (~valid_mask).sum()
    df = df[valid_mask].copy()

    # Also deduplicate on Track B (`cleaned_text_no_emojis`) so two comments that differed
    # only by emoji or @username cannot cause Track B duplicate leakage across splits
    df["track_b_dedup"] = df["cleaned_text_no_emojis"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    dup_track_b_mask = df["track_b_dedup"].duplicated(keep="first")
    dup_track_b_Ids = set(df.loc[dup_track_b_mask, "source_id"].tolist())
    df = df[~dup_track_b_mask].drop(columns=["track_b_dedup"]).copy().reset_index(drop=True)

    # Re-index IDs sequentially if any rows were excluded during post-cleaning check
    df["id"] = [f"GD_{i + 1:04d}" for i in range(len(df))]

    # Step C: Additional required dataset fields
    emoji_lists = df["cleaned_text_with_emojis"].apply(extract_emojis_and_emoticons)
    df["emoji_count"] = emoji_lists.apply(len)
    df["has_emoji"] = df["emoji_count"] > 0
    df["word_count"] = df["cleaned_text_no_emojis"].apply(lambda s: len(re.findall(r"\S+", str(s))))
    df["character_count"] = df["cleaned_text_with_emojis"].str.len()
    df["language"] = df["cleaned_text_no_emojis"].apply(identify_language)

    # Order columns cleanly
    ordered_cols = [
        "id",
        "timestamp",
        "platform_source",
        "source_id",
        "raw_text",
        "cleaned_text_with_emojis",
        "cleaned_text_no_emojis",
        "has_emoji",
        "emoji_count",
        "word_count",
        "character_count",
        "language",
        "label",
        "annotation_status",
        "annotator_id",
    ]
    for col in ordered_cols:
        if col not in df.columns:
            df[col] = ""
    df = df[ordered_cols]

    df.to_csv(output_csv, index=False, encoding="utf-8")
    # Also synchronize raw_dataset.csv IDs and valid rows so raw_dataset.csv and group_d_sentiment_dataset.csv match 1-to-1
    raw_cols = [
        "id",
        "timestamp",
        "platform_source",
        "source_id",
        "raw_text",
        "label",
        "annotation_status",
        "annotator_id",
    ]
    df[raw_cols].to_csv(input_csv, index=False, encoding="utf-8")

    # Synchronize collection_log.csv retained/rejected counts if post-cleaning Track B deduplication removed rows
    log_csv = BASE_DIR / "logs" / "collection_log.csv"
    if log_csv.exists():
        log_df = pd.read_csv(log_csv, encoding="utf-8")
        retained_by_src = df.groupby("source_id").size().to_dict()
        emoji_by_src = df[df["has_emoji"]].groupby("source_id").size().to_dict()
        for i, r in log_df.iterrows():
            sid = r["source_id"]
            new_ret = int(retained_by_src.get(sid, 0))
            old_ret = int(r["retained_count"])
            if new_ret != old_ret:
                diff = old_ret - new_ret
                log_df.at[i, "rejected_duplicate"] = int(r["rejected_duplicate"]) + diff
                log_df.at[i, "rejected_total"] = int(r["rejected_total"]) + diff
                log_df.at[i, "retained_count"] = new_ret
            log_df.at[i, "emoji_comment_count"] = int(emoji_by_src.get(sid, 0))
        log_df.to_csv(log_csv, index=False, encoding="utf-8")

    print("=" * 70)
    print(f"[Phase 4 Complete] Initial rows: {initial_rows} | Excluded empty-after-clean: {excluded_after_clean}")
    print(f"[Phase 4 Complete] Final aligned rows: {len(df)}")
    print(f"[Phase 4 Complete] Track A & Track B saved to: {output_csv}")
    print("=" * 70)
    return df


if __name__ == "__main__":
    run_preprocessing_pipeline()
