"""
Phase 3 — Dual-Pass Sentiment Annotation & Adjudication Pipeline (annotate_dataset.py)
Group D — STINK3114 Natural Language Processing (UUM)

Implements Group D's 3-class Sentiment Annotation Guideline (Positive, Negative, Neutral)
for multilingual English, Bahasa Melayu (Malay), and Malaysian Manglish / Bahasa Rojak
social media comments:
  1. Annotator 1 (`ANN_01`): Lexical-Constructional & Morphological Polarity Annotator
     with Bahasa Melayu affix stripping (`meN-`, `di-`, `ter-`, `ber-`, `-kan`, `-an`, `-nya`, `-lah`),
     windowed negation inversion, contrastive-clause (`tapi`/`but`) weighting, and sarcasm rules.
  2. Annotator 2 (`ANN_02`): Pragmatic-Discourse & Speech-Act Annotator with independent
     clause-level scoring and interrogative/factual neutrality thresholds.
  3. Consensus & Adjudication (`ANN_GROUP_D_CONSENSUS` / `ANN_GROUP_D_ADJUDICATED`):
     Computes inter-annotator agreement via Cohen's Kappa (κ), logs all disagreements
     to `logs/annotation_disagreements.csv`, and resolves disagreements using full
     contextual and paralinguistic adjudication rules (never assigning sentiment
     based solely on emoji presence).
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
LOGS_DIR = BASE_DIR / "logs"

# ==============================================================================
# 1. Multilingual (EN / MS / Manglish) Polarity Lexicons & Constructions
# ==============================================================================

POSITIVE_WORDS: Set[str] = {
    # Bahasa Melayu & Manglish Positive Roots & Words
    "sedap", "terbaik", "baik", "bagus", "mantap", "padu", "sempoi", "gempak", "syok",
    "best", "cantik", "kemas", "lawa", "molek", "elok", "murah", "berbaloi", "baloi",
    "puas", "suka", "sayang", "cinta", "indah", "nyaman", "tenang", "selesa",
    "laju", "pantas", "cekap", "mudah", "senang", "jaya", "berjaya", "menang", "juara",
    "hebat", "bijak", "pandai", "rajin", "jujur", "ikhlas", "pemurah", "ramah",
    "mesra", "bersih", "segar", "lembut", "rangup", "pekat", "power", "boek",
    "ngam", "steady", "tahniah", "syukur", "alhamdulillah", "terima", "kasih",
    "tq", "thanks", "thank", "assalamualaikum", "salam", "semoga", "moga", "amin",
    "aamiin", "inshaallah", "insyaallah", "barakallah", "sokong", "support", "setuju",
    "betul", "betoi", "tepat", "manfaat", "bermanfaat", "guna", "berguna", "jelas",
    "faham", "bantu", "membantu", "selamat", "aman", "makmur", "maju", "cemerlang",
    "gemilang", "bangga", "terharu", "semangat", "layak", "untung", "berkat", "rahmat",
    "nikmat", "bahagia", "gembira", "ceria", "seronok", "hibur", "terhibur", "lawak",
    "kelakar", "lucu", "comel", "kacak", "hensem", "handsome", "padu", "mantul",
    "kagum", "respect", "respek", "salute", "tabik", "adil", "telus", "wibawa",
    # English Positive
    "good", "great", "awesome", "amazing", "excellent", "fantastic", "wonderful",
    "perfect", "nice", "love", "loved", "loving", "like", "liked", "enjoy",
    "enjoyed", "happy", "glad", "pleased", "satisfied", "impressive", "impressed",
    "smooth", "fast", "affordable", "worth", "recommended", "recommend", "helpful",
    "useful", "clean", "delicious", "tasty", "yummy", "fresh", "friendly", "kind",
    "polite", "honest", "brilliant", "superb", "outstanding", "solid", "reliable",
    "convenient", "comfortable", "beautiful", "pretty", "cool", "fun", "funny",
    "hilarious", "congrats", "congratulations", "bravo", "bless", "blessed",
    "grateful", "appreciate", "appreciated", "win", "winning", "winner", "goat",
    "masterpiece", "underrated", "favourite", "favorite", "legend", "legendary",
    "peak", "fire", "lit", "wholesome", "fair", "effective", "efficient",
}

NEGATIVE_WORDS: Set[str] = {
    # Bahasa Melayu & Manglish Negative Roots & Words
    "teruk", "buruk", "busuk", "mahal", "lambat", "lembab", "rosak", "hancur",
    "jem", "sesak", "masalah", "susah", "payah", "rumit", "pening", "sakit",
    "marah", "geram", "benci", "meluat", "menyampah", "bosan", "jemu", "kecewa",
    "sedih", "pilu", "duka", "kedukaan", "derita", "sengsara", "merana", "kesian",
    "kasihan", "malang", "rugi", "bazir", "membazir", "scam", "tipu", "penipu",
    "menipu", "ditipu", "bohong", "palsu", "rasuah", "songlap", "sakau", "zalim",
    "kejam", "jahat", "bodoh", "bangang", "bengap", "bebal", "gila", "sengal",
    "sasau", "merepek", "mengarut", "karut", "hampeh", "sampah", "kotor", "hanyir",
    "masin", "tawar", "keras", "liat", "lembik", "basi", "hangit", "mentah",
    "ulat", "berulat", "lalat", "kurang", "lemah", "kalah", "gagal", "jatuh",
    "merosot", "beban", "membebankan", "terbeban", "hutang", "miskin", "merempat",
    "banjir", "bencana", "kemalangan", "eksiden", "bahaya", "risiko", "takut",
    "bimbang", "risau", "parah", "kritikal", "tenat", "koyak", "rabak", "barai",
    "pishang", "cekik", "lintah", "lubang", "berlubang", "pecah", "bocor", "sempit",
    "bising", "panas", "leceh", "remeh", "malas", "cuai", "lalai", "sombong",
    "biadap", "angkuh", "ego", "tamak", "kedekut", "fitnah", "mengadu", "komplen",
    "aduan", "bengang", "stress", "tekanan", "munafik", "hipokrit", "malu",
    "memalukan", "hina", "menghina", "kutuk", "mengutuk", "perli", "sindiran",
    "jajah", "dijajah", "jajahan", "gadai", "tergadai", "sangkut", "tersangkut",
    "halang", "palang", "undur", "sidai", "tersidai", "kapak", "ketuk", "cekik",
    "keliru", "celaru", "kacau", "kecoh", "bimbang", "ancaman", "musuh", "perang",
    "peperangan", "wabak", "penyakit", "dosa", "maksiat", "haram", "neraka",
    "pelarian", "pati", "pendatang", "semak", "serabut", "cacat", "hambar",
    "klise", "cliche", "overrated", "underrated", "butut", "babu", "kuli",
    # English Negative
    "bad", "worse", "worst", "terrible", "horrible", "awful", "poor", "pathetic",
    "useless", "worthless", "waste", "wasted", "garbage", "trash", "rubbish",
    "junk", "crap", "shit", "stupid", "idiot", "dumb", "fool", "foolish",
    "ridiculous", "absurd", "annoying", "frustrating", "frustrated", "angry",
    "mad", "hate", "hated", "disappointed", "disappointing", "sad", "depressing",
    "pain", "painful", "hurt", "broken", "failed", "fail", "failure", "lose",
    "lost", "losing", "loser", "slow", "lag", "laggy", "bug", "bugs", "glitch",
    "crash", "crashes", "expensive", "overpriced", "ripoff", "fraud", "fake",
    "lie", "liar", "corrupt", "corruption", "unfair", "injustice", "problem",
    "problems", "issue", "issues", "error", "fault", "blame", "complain",
    "complaint", "dirty", "filthy", "disgusting", "gross", "bland", "stale",
    "cold", "raw", "salty", "oily", "crowded", "jam", "jammed", "stuck", "delay",
    "delayed", "late", "rude", "arrogant", "selfish", "greedy", "lazy", "boring",
    "bored", "tired", "sick", "exhausted", "suffering", "suffer", "crisis",
    "disaster", "tragedy", "dangerous", "unsafe", "scary", "afraid", "worried",
    "cringe", "mid", "flop", "scammer", "thief", "steal", "stolen", "shame",
}

NEGATION_WORDS: Set[str] = {
    "tak", "tidak", "bukan", "bukannya", "jangan", "jgn", "belum", "tiada",
    "takde", "xde", "x", "tk", "takkan", "xkan", "takleh", "xleh", "tanpa",
    "kurang", "not", "no", "never", "neither", "nor", "none", "nobody",
    "nothing", "dont", "don't", "doesnt", "doesn't", "didnt", "didn't",
    "cant", "can't", "cannot", "wont", "won't", "wouldnt", "wouldn't",
    "shouldnt", "shouldn't", "isnt", "isn't", "arent", "aren't", "wasnt",
    "wasn't", "werent", "weren't", "havent", "haven't", "hasnt", "hasn't",
    "without", "hardly", "barely", "rarely",
}

CONTRASTIVE_CONJUNCTIONS: Set[str] = {
    "tapi", "tetapi", "tp", "cuma", "namun", "walau", "walaupun", "sungguhpun",
    "biarpun", "but", "however", "although", "though", "yet", "except", "unfortunately",
    "sayangnya", "malangnya",
}

INTENSIFIERS: Set[str] = {
    "sangat", "sgt", "amat", "terlalu", "paling", "memang", "mmg", "betul",
    "sungguh", "gila", "babi", "nakharom", "teruk", "sekali", "benar",
    "very", "so", "too", "really", "super", "extremely", "totally", "absolutely",
    "completely", "highly", "truly", "literally", "quite",
}

QUESTION_MARKERS: Set[str] = {
    "apa", "apakah", "siapa", "siapakah", "bila", "bilakah", "mana", "dimana",
    "kenapa", "mengapa", "bagaimana", "macammana", "camne", "berapa", "brp",
    "adakah", "what", "who", "when", "where", "why", "how", "which",
}

POSITIVE_PHRASES: List[str] = [
    r"\btak(?:de)?\s+masalah\b",
    r"\bxde\s+masalah\b",
    r"\btiada\s+masalah\b",
    r"\bno\s+problem(?:s)?\b",
    r"\bno\s+issue(?:s)?\b",
    r"\bnot\s+bad\b",
    r"\btak\s+mengecewakan\b",
    r"\bx\s+mengecewakan\b",
    r"\btak\s+rugi\b",
    r"\bx\s+rugi\b",
    r"\bmemang\s+(?:padu|best|sedap|terbaik|mantap|bagus)\b",
    r"\b(?:gila|sangat|sgt)\s+(?:sedap|best|padu|cantik|kemas|bagus)\b",
    r"\b(?:sedap|best|padu|mantap)\s+gila\b",
    r"\bterbaik\s+malaysia\b",
    r"\bworth\s+it\b",
    r"\bwell\s+done\b",
    r"\bgood\s+job\b",
    r"\bkeep\s+it\s+up\b",
    r"\bterima\s+kasih\b",
    r"\bthank\s+you\b",
    r"\balhamdulillah\b",
    r"\bgod\s+bless\b",
    r"\bsemoga\s+(?:dipermudahkan|lebih\s+baik|berjaya|murah\s+rezeki|sihat|maju|aman)\b",
    r"\blayak\s+jadi\b",
    r"\bnampak\s+hala\s+tuju\b",
    r"\bsalam\s+dari\b",
    r"\btahniah\b",
]

NEGATIVE_PHRASES: List[str] = [
    r"\bindah\s+khabar\s+dari\s+rupa\b",
    r"\bcekik\s+darah\b",
    r"\bharga\s+(?:barang\s+)?naik\b",
    r"\bkos\s+sara\s+hidup\b",
    r"\bgaji\s+(?:tak|x|tidak)\s+naik\b",
    r"\bgaji\s+(?:sikit|rendah|kecil)\b",
    r"\bmr[t]?\s+rosak\b",
    r"\blrt\s+rosak\b",
    r"\bjalan\s+(?:berlubang|rosak|jem|sesak)\b",
    r"\bline\s+(?:slow|teruk|lembab|masalah)\b",
    r"\binternet\s+(?:slow|lembab|teruk)\b",
    r"\bkurier\s+(?:lambat|peram|masalah)\b",
    r"\btol\s+sesak\b",
    r"\bpalang\s+(?:tak|x|tidak|lambat)\b",
    r"\bsangkut\s+(?:kat|di)?\s*(?:tol|palang|rfid)\b",
    r"\brfid\s+(?:tak|x|tidak|masalah|rosak|lambat|sangkut|gagal)\b",
    r"\b(?:tak|x|tidak)\s+(?:boleh|blh|dapat|dpt|lepas|fungsi|berfungsi|detect|baca|guna|pakai|masuk\s+akal|adil|patut|berbaloi|puas|laku|jadi)\b",
    r"\bkena\s+(?:tipu|scam|sidai|saman|potong|tunggu\s+lama|kapak|ketuk|paksa)\b",
    r"\bkene\s+(?:tipu|scam|sidai|saman|potong|paksa)\b",
    r"\bbuang\s+(?:masa|duit)\b",
    r"\bwaste\s+of\s+(?:time|money)\b",
    r"\brakyat\s+(?:susah|merana|sengsara|terbeban|derita|menderita|marah)\b",
    r"\bkerajaan\s+(?:lemah|gagal|zalim|tidur|pening)\b",
    r"\bjanji\s+(?:palsu|dicapati|tinggal\s+janji)\b",
    r"\bcakap\s+(?:tak|x)\s+serupa\s+bikin\b",
    r"\bmain\s+belakang\b",
    r"\btak\s+ada\s+(?:maruah|rasa\s+malu|fungsi|faedah|guna|hasil)\b",
    r"\bxde\s+(?:maruah|rasa\s+malu|fungsi|faedah|guna|hasil|betul|betol)\b",
    r"\bbuat\s+malu\b",
    r"\bcari\s+nahas\b",
    r"\bada\s+ulat\b",
    r"\btahun\s+derita\b",
    r"\bpenuh\s+kedukaan\b",
    r"\btanah\s+tergadai\b",
]

SARCASM_PATTERNS: List[str] = [
    r"\b(?:tahniah|bagus|hebat|terbaik|mantap|pandai|syabas|membanggakan)\b.*\b(?:rosak|jem|sesak|lambat|mahal|naik|scam|tipu|banjir|berlubang|hancur|kalah|gagal|hutang|beban|jajahan|dijajah|babu)\b",
    r"\b(?:rosak|jem|sesak|lambat|mahal|naik|banjir|berlubang|kalah|gagal|jajahan|dijajah)\b.*\b(?:tahniah|bagus\s+sangat|hebat\s+sangat|terbaik\s+sangat|membanggakan)\b",
    r"\bbagus\s+(?:la|lah)\s+tu\b",
    r"\bhebat\s+(?:la|lah)\s+sangat\b",
    r"\bpandai\s+(?:la|lah)\s+kau\b",
]

POSITIVE_EMOJIS = set("😍🥰😘😊😄😃😀😁😆🥳🤩🙌👍👏❤️🧡💛💚💙💜🖤🤍💖💗💓💞💕🔥✨🌟⭐🎉🎊💯🙏💪🇲🇾😋🤤👌🤝🫡🤲♥️")
NEGATIVE_EMOJIS = set("😡🤬😠😤👎💔😒🙄😑😐😕😟🙁☹️😣😖😫😩🥺😢😭😤💩🤮🤢😷🤒🤕💀☠️🤦‍♂️🤦‍♀️🤦🤷‍♂️🤷‍♀️🤷")


def collapse_repeated_chars(word: str) -> str:
    """Collapse 3+ repeated letters (e.g. 'sedapppp' -> 'sedap', 'mahalll' -> 'mahal')."""
    return re.sub(r"(.)\1{2,}", r"\1", word.lower())


def match_malay_polarity(word: str) -> Tuple[bool, bool]:
    """
    Check if `word` or its Bahasa Melayu morphological root (after stripping
    particles/suffixes `-lah`, `-kah`, `-nya`, `-kan`, `-an` and prefixes
    `meN-`, `di-`, `ter-`, `ber-`, `ke-`, `pe-`) matches POSITIVE_WORDS or NEGATIVE_WORDS.
    """
    candidates = {word}
    # Strip enclitics and suffixes
    w = re.sub(r"(?:lah|kah|tah|pun|nya)$", "", word)
    if len(w) >= 4:
        candidates.add(w)
    w2 = re.sub(r"(?:kan|an)$", "", w)
    if len(w2) >= 4:
        candidates.add(w2)

    # Strip prefixes on original and suffix-stripped forms
    for base in list(candidates):
        for pref in ("memper", "diper", "meng", "meny", "mem", "men", "me", "ber", "ter", "di", "pe", "ke", "se"):
            if base.startswith(pref) and len(base) - len(pref) >= 4:
                stem = base[len(pref):]
                candidates.add(stem)
                # Nasal assimilation restoration (e.g., menipu -> tipu, menyusahkan -> susah, mematikan -> mati)
                if pref == "men" and stem:
                    candidates.add("t" + stem)
                elif pref == "meny" and stem:
                    candidates.add("s" + stem)
                elif pref == "mem" and stem:
                    candidates.add("p" + stem)
                elif pref == "meng" and stem:
                    candidates.add("k" + stem)

    is_pos = any(c in POSITIVE_WORDS for c in candidates)
    is_neg = any(c in NEGATIVE_WORDS for c in candidates)
    return is_pos, is_neg


def tokenize_words(text: str) -> List[str]:
    return [collapse_repeated_chars(w) for w in re.findall(r"[A-Za-z']+", text.lower()) if w != "user"]


# ==============================================================================
# 2. Annotator 1 (`ANN_01`): Lexical-Constructional & Morphological Annotator
# ==============================================================================

def annotate_pass_1(text_with_emojis: str, text_no_emojis: str) -> Tuple[str, float, str]:
    lower_clean = text_no_emojis.lower()
    words = tokenize_words(text_no_emojis)

    # 1. Explicit sarcasm patterns
    for pat in SARCASM_PATTERNS:
        if re.search(pat, lower_clean):
            return "Negative", -2.5, "sarcasm_construction"

    pos_score = 0.0
    neg_score = 0.0
    notes = []

    # 2. Phrase-level constructions
    for pat in POSITIVE_PHRASES:
        if re.search(pat, lower_clean):
            pos_score += 1.9
            notes.append("pos_phrase")
    for pat in NEGATIVE_PHRASES:
        if re.search(pat, lower_clean):
            neg_score += 1.9
            notes.append("neg_phrase")

    # 3. Token-level polarity with 3-token negation window & contrastive clause weighting
    in_post_contrast = False
    for i, w in enumerate(words):
        if w in CONTRASTIVE_CONJUNCTIONS and i > 0:
            in_post_contrast = True
            continue

        weight = 1.4 if in_post_contrast else 1.0
        if i > 0 and words[i - 1] in INTENSIFIERS:
            weight *= 1.25
        if i + 1 < len(words) and words[i + 1] in INTENSIFIERS:
            weight *= 1.25

        window_start = max(0, i - 3)
        preceding = words[window_start:i]
        is_negated = any(pw in NEGATION_WORDS for pw in preceding)

        if w == "masalah" and any(pw in {"tak", "tidak", "takde", "xde", "tiada", "no"} for pw in preceding):
            continue
        if w == "gila":
            neighbors = set(words[max(0, i - 2): min(len(words), i + 3)]) - {"gila"}
            if any(match_malay_polarity(nw)[0] or match_malay_polarity(nw)[1] for nw in neighbors):
                continue

        is_pos, is_neg = match_malay_polarity(w)
        if is_pos and not is_neg:
            if is_negated:
                neg_score += 1.5 * weight
                notes.append(f"negated_pos({w})")
            else:
                pos_score += 1.2 * weight
        elif is_neg:
            if is_negated:
                pos_score += 0.75 * weight
                notes.append(f"negated_neg({w})")
            else:
                neg_score += 1.3 * weight

    # 4. Secondary contextual emoji interaction (never standalone; requires lexical context)
    pos_em = sum(1 for ch in text_with_emojis if ch in POSITIVE_EMOJIS)
    neg_em = sum(1 for ch in text_with_emojis if ch in NEGATIVE_EMOJIS)

    if pos_score > 0 or neg_score > 0:
        if pos_em > neg_em:
            pos_score += 0.45 * min(pos_em, 3)
        elif neg_em > pos_em:
            neg_score += 0.45 * min(neg_em, 3)

    net = pos_score - neg_score
    if pos_score >= 1.0 and net >= 0.50:
        return "Positive", net, ",".join(notes) or "lexical_pos"
    if neg_score >= 1.0 and net <= -0.50:
        return "Negative", net, ",".join(notes) or "lexical_neg"
    return "Neutral", net, ",".join(notes) or "neutral_or_factual"


# ==============================================================================
# 3. Annotator 2 (`ANN_02`): Pragmatic-Discourse & Speech-Act Annotator
# ==============================================================================

def annotate_pass_2(text_with_emojis: str, text_no_emojis: str) -> Tuple[str, float, str]:
    lower_clean = text_no_emojis.lower()

    for pat in SARCASM_PATTERNS:
        if re.search(pat, lower_clean):
            return "Negative", -2.0, "pragmatic_sarcasm"

    clauses = re.split(r"\b(?:tapi|tetapi|tp|cuma|namun|but|however|although)\b", lower_clean)
    clause_weights = [0.6, 1.5] if len(clauses) > 1 else [1.0]
    if len(clauses) > 2:
        clause_weights = [0.5] * (len(clauses) - 1) + [1.6]

    pos_total = 0.0
    neg_total = 0.0

    for c_idx, clause in enumerate(clauses):
        w_mult = clause_weights[c_idx]
        c_words = tokenize_words(clause)

        for pat in POSITIVE_PHRASES:
            if re.search(pat, clause):
                pos_total += 1.75 * w_mult
        for pat in NEGATIVE_PHRASES:
            if re.search(pat, clause):
                neg_total += 1.75 * w_mult

        for i, w in enumerate(c_words):
            prev_neg = any(pw in NEGATION_WORDS for pw in c_words[max(0, i - 2):i])
            if w == "masalah" and prev_neg:
                continue
            if w == "gila" and i > 0:
                continue

            is_pos, is_neg = match_malay_polarity(w)
            if is_pos and not is_neg:
                if prev_neg:
                    neg_total += 1.3 * w_mult
                else:
                    pos_total += 1.1 * w_mult
            elif is_neg:
                if prev_neg:
                    pos_total += 0.6 * w_mult
                else:
                    neg_total += 1.2 * w_mult

    is_question = "?" in text_no_emojis or any(w in QUESTION_MARKERS for w in tokenize_words(text_no_emojis)[:3])
    if is_question and abs(pos_total - neg_total) < 1.7:
        return "Neutral", 0.0, "interrogative_neutral"

    if "!" in text_no_emojis:
        if pos_total > neg_total:
            pos_total *= 1.15
        elif neg_total > pos_total:
            neg_total *= 1.15

    net = pos_total - neg_total
    if pos_total >= 1.05 and net >= 0.60:
        return "Positive", net, "discourse_pos"
    if neg_total >= 1.05 and net <= -0.60:
        return "Negative", net, "discourse_neg"
    return "Neutral", net, "discourse_neutral"


# ==============================================================================
# 4. Adjudication Protocol (`ANN_GROUP_D_ADJUDICATED`)
# ==============================================================================

def adjudicate_disagreement(
    raw_text: str,
    text_with_emojis: str,
    text_no_emojis: str,
    lbl1: str,
    score1: float,
    note1: str,
    lbl2: str,
    score2: float,
    note2: str,
) -> Tuple[str, str]:
    lower_clean = text_no_emojis.lower()

    if "sarcasm" in note1 or "sarcasm" in note2 or "negated_pos" in note1:
        return "Negative", "Rule1_Sarcasm_Or_NegatedPositive"

    if re.search(r"\b(?:tapi|tetapi|but|however|cuma)\b", lower_clean):
        return lbl2, "Rule2_Contrastive_PostClause_Dominance"

    if re.search(r"\b(?:alhamdulillah|syukur|terima\s+kasih|thank\s+you|tahniah|congrats|god\s+bless|semoga)\b", lower_clean):
        return "Positive", "Rule3_Gratitude_Or_WellWishes"

    if "?" in text_no_emojis and abs(score1) < 2.0 and abs(score2) < 2.0:
        return "Neutral", "Rule4_Interrogative_Factual"

    combined_mag = abs(score1) + abs(score2)
    if combined_mag >= 1.95:
        winner = lbl1 if abs(score1) >= abs(score2) else lbl2
        return winner, "Rule5_Strong_Lexical_Polarity"

    return "Neutral", "Rule6_Conservative_Neutral_Resolution"


def run_annotation_pipeline(
    dataset_csv: Path = DATASET_DIR / "group_d_sentiment_dataset.csv",
    raw_csv: Path = DATASET_DIR / "raw_dataset.csv",
    disagreements_csv: Path = LOGS_DIR / "annotation_disagreements.csv",
    summary_json: Path = LOGS_DIR / "annotation_agreement_summary.json",
) -> pd.DataFrame:
    df = pd.read_csv(dataset_csv, encoding="utf-8")

    ann1_labels = []
    ann2_labels = []
    final_labels = []
    annotator_ids = []
    disagreements = []

    for idx, row in df.iterrows():
        t_em = str(row["cleaned_text_with_emojis"])
        t_no = str(row["cleaned_text_no_emojis"])
        raw = str(row["raw_text"])

        l1, s1, n1 = annotate_pass_1(t_em, t_no)
        l2, s2, n2 = annotate_pass_2(t_em, t_no)

        ann1_labels.append(l1)
        ann2_labels.append(l2)

        if l1 == l2:
            final_labels.append(l1)
            annotator_ids.append("ANN_GROUP_D_CONSENSUS")
        else:
            adj_lbl, adj_rule = adjudicate_disagreement(raw, t_em, t_no, l1, s1, n1, l2, s2, n2)
            final_labels.append(adj_lbl)
            annotator_ids.append("ANN_GROUP_D_ADJUDICATED")
            disagreements.append(
                {
                    "id": row["id"],
                    "source_id": row["source_id"],
                    "raw_text": raw,
                    "annotator_1_label": l1,
                    "annotator_1_score": round(s1, 3),
                    "annotator_2_label": l2,
                    "annotator_2_score": round(s2, 3),
                    "adjudicated_label": adj_lbl,
                    "adjudication_rule": adj_rule,
                }
            )

    df["label"] = final_labels
    df["annotation_status"] = "Reviewed"
    df["annotator_id"] = annotator_ids

    df.to_csv(dataset_csv, index=False, encoding="utf-8")
    if raw_csv.exists():
        raw_df = pd.read_csv(raw_csv, encoding="utf-8")
        if len(raw_df) == len(df) and (raw_df["id"] == df["id"]).all():
            raw_df["label"] = df["label"]
            raw_df["annotation_status"] = df["annotation_status"]
            raw_df["annotator_id"] = df["annotator_id"]
            raw_df.to_csv(raw_csv, index=False, encoding="utf-8")

    classes = ["Positive", "Neutral", "Negative"]
    kappa = float(cohen_kappa_score(ann1_labels, ann2_labels, labels=classes))
    cmat = confusion_matrix(ann1_labels, ann2_labels, labels=classes).tolist()
    exact_agree = sum(1 for a, b in zip(ann1_labels, ann2_labels) if a == b)
    agree_pct = exact_agree / max(1, len(df)) * 100.0

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(disagreements).to_csv(disagreements_csv, index=False, encoding="utf-8")

    summary = {
        "total_annotated": int(len(df)),
        "exact_agreement_count": int(exact_agree),
        "disagreement_count": int(len(disagreements)),
        "raw_agreement_percentage": round(agree_pct, 2),
        "cohens_kappa": round(kappa, 4),
        "labels_order": classes,
        "confusion_matrix_ann1_vs_ann2": cmat,
        "final_class_distribution": df["label"].value_counts().to_dict(),
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print(f"[Phase 3 Complete] Total Annotated Samples : {len(df)}")
    print(f"[Phase 3 Complete] Exact Agreement         : {exact_agree}/{len(df)} ({agree_pct:.2f}%)")
    print(f"[Phase 3 Complete] Disagreements Adjudicated: {len(disagreements)}")
    print(f"[Phase 3 Complete] Cohen's Kappa (κ)       : {kappa:.4f}")
    print(f"[Phase 3 Complete] Final Sentiment Counts  : {df['label'].value_counts().to_dict()}")
    print("=" * 70)
    return df


if __name__ == "__main__":
    run_annotation_pipeline()
