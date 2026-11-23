"""
Emoji-to-Text-Token Converter for TF-IDF Feature Representation (emoji_tokenizer.py)
Group D — STINK3114 Natural Language Processing (UUM)

Standard scikit-learn TfidfVectorizer tokenization (`token_pattern=r"(?u)\\b\\w\\w+\\b"`)
discards Unicode emoji symbols and ASCII emoticons because they are not alphanumeric
word characters. To evaluate RO1 (Emoji Retention vs. Removal) faithfully within a
TF-IDF + Logistic Regression pipeline, this module converts every Unicode emoji,
regional flag pair (e.g., 🇲🇾), and documented ASCII emoticon in Track A
(`cleaned_text_with_emojis`) into a deterministic, human-readable alphanumeric token
(e.g., `😂` -> ` emoji_face_with_tears_of_joy `, `<3` -> ` emoticon_heart `).
"""

import re
import unicodedata
from typing import Dict

from preprocess import UNICODE_EMOJI_REGEX, EMOTICON_REGEX, EMOJI_MODIFIERS_REGEX

# Regional indicator letter mapping for 2-letter ISO country flags (e.g. 🇲🇾 -> my)
REGIONAL_OFFSET = 0x1F1E6

FLAG_NAMES: Dict[str, str] = {
    "MY": "malaysia",
    "ID": "indonesia",
    "SG": "singapore",
    "US": "usa",
    "GB": "uk",
    "PS": "palestine",
    "IL": "israel",
    "BR": "brazil",
    "JP": "japan",
    "KR": "south_korea",
    "CN": "china",
    "TH": "thailand",
    "BN": "brunei",
    "SA": "saudi_arabia",
}

EMOTICON_TOKEN_MAP: Dict[str, str] = {
    ":)": "emoticon_smile",
    ":-)": "emoticon_smile",
    "=)": "emoticon_smile",
    ":(": "emoticon_sad",
    ":-(": "emoticon_sad",
    "=(": "emoticon_sad",
    ":D": "emoticon_laugh",
    ":-D": "emoticon_laugh",
    "=D": "emoticon_laugh",
    ";D": "emoticon_wink_laugh",
    ";-D": "emoticon_wink_laugh",
    ";)": "emoticon_wink",
    ";-)": "emoticon_wink",
    ":P": "emoticon_tongue",
    ":-P": "emoticon_tongue",
    ":p": "emoticon_tongue",
    ":-p": "emoticon_tongue",
    ":'(": "emoticon_cry",
    ":'-(": "emoticon_cry",
    ":/": "emoticon_skeptical",
    ":-/": "emoticon_skeptical",
    ":\\": "emoticon_skeptical",
    ":-\\": "emoticon_skeptical",
    ":O": "emoticon_surprised",
    ":-O": "emoticon_surprised",
    ":o": "emoticon_surprised",
    ":-o": "emoticon_surprised",
    "<3": "emoticon_heart",
    "</3": "emoticon_broken_heart",
    "T_T": "emoticon_crying",
    "ToT": "emoticon_crying",
    "T.T": "emoticon_crying",
    ">_<": "emoticon_frustrated",
    "^_^": "emoticon_happy",
    "-_-": "emoticon_unamused",
}


def _unicode_emoji_to_token(match: re.Match) -> str:
    """Convert a matched Unicode emoji or 2-char flag sequence into an alphanumeric token."""
    raw = match.group(0)
    # Strip variation selectors / ZWJ / skin-tone modifiers so variants map consistently
    # (e.g., ❤️ and ❤ both map to emoji_heavy_black_heart, 👍🏻 and 👍 both map to emoji_thumbs_up_sign)
    core = re.sub(r"[\uFE0E\uFE0F\u200D\U0001F3FB-\U0001F3FF]", "", raw)
    if not core:
        return " "

    # Check 2-character regional indicator flag sequence
    if len(core) == 2 and all(0x1F1E6 <= ord(c) <= 0x1F1FF for c in core):
        iso = "".join(chr(ord("A") + (ord(c) - REGIONAL_OFFSET)) for c in core)
        country = FLAG_NAMES.get(iso, iso.lower())
        return f" emoji_flag_{country} "

    base_char = core[0]
    uname = unicodedata.name(base_char, f"u{ord(base_char):04x}")
    slug = re.sub(r"[^a-z0-9]+", "_", uname.lower()).strip("_")
    return f" emoji_{slug} "


def _emoticon_to_token(match: re.Match) -> str:
    """Convert a matched ASCII/text emoticon into a deterministic alphanumeric token."""
    raw = match.group(0)
    tok = EMOTICON_TOKEN_MAP.get(raw, "emoticon_other")
    return f" {tok} "


def convert_emojis_to_tokens(text: str) -> str:
    """
    Transform all Unicode emojis and text emoticons in `text` into space-delimited
    alphanumeric text tokens (`emoji_<name>`, `emoticon_<name>`) suitable for TF-IDF.
    All non-emoji words, negation terms, and structure are preserved unchanged.
    """
    if not isinstance(text, str) or not text:
        return ""
    out = UNICODE_EMOJI_REGEX.sub(_unicode_emoji_to_token, text)
    # Clean any stray standalone emoji modifiers left behind
    out = EMOJI_MODIFIERS_REGEX.sub(" ", out)
    out = EMOTICON_REGEX.sub(_emoticon_to_token, out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


if __name__ == "__main__":
    samples = [
        "I from (san marino)my nama Mario...terbaik Malaysia..😊",
        "God bless Malaysia 🇲🇾🥰",
        "Tahun 2025 tahun derita😢",
        "Anwar dah tak ada rasa Malu sebab kuat menipu 😂😂😂😂😂",
        "Thanks bro <3 ;)",
    ]
    for s in samples:
        print("ORIG :", s)
        print("TOKEN:", convert_emojis_to_tokens(s))
        print()
