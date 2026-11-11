"""
Phase 2 — Reproducible Data Collection Pipeline (collect_data.py)
Group D — STINK3114 Natural Language Processing (UUM)

Collects real, unmodified user-generated comments from diverse Malaysian YouTube
videos across six topical domains (News & Public Policy, Technology & Telco Reviews,
Food & Dining Reviews, Consumer & E-Commerce Services, Public Transport & Infrastructure,
and Sports & Entertainment) to minimize single-topic bias while capturing natural
English, Bahasa Melayu (Malay), and Malaysian Manglish / Bahasa Rojak comments
with authentic Unicode emojis.

Supports two legal collection modes:
1. Official YouTube Data API v3 (`commentThreads.list`) when `YOUTUBE_API_KEY` is provided in `.env`.
2. Public YouTube Innertube JSON web pagination (`/youtubei/v1/next`) for zero-credential reproducibility.
"""

import os
import re
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
LOGS_DIR = BASE_DIR / "logs"

# Unicode emoji & symbol detection pattern
EMOJI_PATTERN = re.compile(
    r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]"
)

# Non-Latin script filter (excludes comments written purely in Arabic, Cyrillic, CJK, Indic scripts without Latin characters)
LATIN_WORD_PATTERN = re.compile(r"[A-Za-z]{2,}")
URL_ONLY_PATTERN = re.compile(r"^\s*(https?://\S+\s*)+$", re.IGNORECASE)

# Curated multi-domain Malaysian YouTube video sources (6 domains, 68 candidate videos)
CURATED_VIDEO_SOURCES: List[Tuple[str, str, str]] = [
    # Domain 1: Malaysian News, Cost of Living & Public Policy
    ("UvVn--gCXd0", "news_policy", "malaysia isu semasa 2025"),
    ("mM6FxN-l5TA", "news_policy", "malaysia isu semasa 2025"),
    ("ySN6If6O3rU", "news_policy", "malaysia isu semasa 2025"),
    ("-BH3mham3Ls", "news_policy", "malaysia isu semasa 2025"),
    ("wDR-jLKhHuI", "news_policy", "malaysia isu semasa 2025"),
    ("GEj60Uxhf5w", "news_policy", "malaysia isu semasa 2025"),
    ("kOJUzyXdevE", "news_policy", "malaysia isu semasa 2025"),
    ("-3aMoHFZJ2w", "news_policy", "malaysia isu semasa 2025"),
    ("qrC_gWqYcxs", "news_policy", "malaysia kos sara hidup harga barang naik astro awani"),
    ("rGe_1IufSgc", "news_policy", "malaysia kos sara hidup harga barang naik astro awani"),
    ("e-SDLw4wirE", "news_policy", "malaysia kos sara hidup harga barang naik astro awani"),
    ("xYh95iEaLGk", "news_policy", "malaysia kos sara hidup harga barang naik astro awani"),
    ("PMZjkhacXYE", "news_policy", "malaysia kos sara hidup harga barang naik astro awani"),
    ("7ClSiNMyq60", "news_policy", "malaysia isu semasa subsidi minyak Budi Madani"),
    ("g9izw7g5wWY", "news_policy", "malaysia isu semasa subsidi minyak Budi Madani"),
    ("DRd4y9DKISY", "news_policy", "malaysia isu semasa subsidi minyak Budi Madani"),
    ("mVmYibok-7A", "news_policy", "malaysia isu semasa subsidi minyak Budi Madani"),
    ("sWMm9LCG71E", "news_policy", "malaysia isu semasa subsidi minyak Budi Madani"),
    ("PobX7I3mU_A", "news_policy", "malaysia banjir masalah rakyat berita"),
    ("SVTKykPjnjs", "news_policy", "malaysia banjir masalah rakyat berita"),
    ("k6y8SHClWAk", "news_policy", "malaysia banjir masalah rakyat berita"),
    ("iGab66zOvWk", "news_policy", "malaysia banjir masalah rakyat berita"),
    ("8l537SeOtWE", "news_policy", "malaysia banjir masalah rakyat berita"),
    # Domain 2: Technology, Telco & Gadget Reviews
    ("NKdId1dfdQA", "tech_review", "review telefon pintar terbaik malaysia wiser my"),
    ("yJUJYrVlnl0", "tech_review", "review telefon pintar terbaik malaysia wiser my"),
    ("U5Si-46dCNk", "tech_review", "review telefon pintar terbaik malaysia wiser my"),
    ("7nVy-SUM8fM", "tech_review", "review telefon pintar terbaik malaysia wiser my"),
    ("NgQK-RNfFjs", "tech_review", "review telefon pintar terbaik malaysia wiser my"),
    ("9lYo75hsBEc", "tech_review", "masalah 5g telco malaysia line slow"),
    ("hLhd0U8zvxE", "tech_review", "masalah 5g telco malaysia line slow"),
    ("vx5ipX5P_iA", "tech_review", "masalah 5g telco malaysia line slow"),
    ("L3lgpnSWhvo", "tech_review", "masalah 5g telco malaysia line slow"),
    ("ZrSCgExxbWA", "tech_review", "masalah 5g telco malaysia line slow"),
    # Domain 3: Malaysian Food, Restaurant & Street Food Reviews
    ("RplX-Ffzbps", "food_dining", "review makanan viral malaysia sedap tak sedap"),
    ("p6Wqixu2BRw", "food_dining", "review makanan viral malaysia sedap tak sedap"),
    ("0c2OLdn1ahU", "food_dining", "review makanan viral malaysia sedap tak sedap"),
    ("P2Vgr4GWlX0", "food_dining", "review makanan viral malaysia sedap tak sedap"),
    ("w2rLFIZv6Sw", "food_dining", "review makanan viral malaysia sedap tak sedap"),
    ("7vVM1F4fhhc", "food_dining", "bazar ramadan malaysia makanan indah khabar dari rupa"),
    ("Lw-gceuCLGw", "food_dining", "bazar ramadan malaysia makanan indah khabar dari rupa"),
    ("hUjRi6WQZo4", "food_dining", "bazar ramadan malaysia makanan indah khabar dari rupa"),
    ("hlnJ7mjEe7E", "food_dining", "bazar ramadan malaysia makanan indah khabar dari rupa"),
    ("HNR25dOZT6U", "food_dining", "bazar ramadan malaysia makanan indah khabar dari rupa"),
    # Domain 4: Consumer Services, E-Wallets & E-Commerce
    ("5czlptGf7QY", "consumer_service", "masalah rfid touch n go tol sesak malaysia"),
    ("uWkYVFQy5H0", "consumer_service", "masalah rfid touch n go tol sesak malaysia"),
    ("4St5PCDN4VY", "consumer_service", "masalah rfid touch n go tol sesak malaysia"),
    ("6YTxgitZvNU", "consumer_service", "masalah rfid touch n go tol sesak malaysia"),
    ("KkPxHg_jsio", "consumer_service", "masalah rfid touch n go tol sesak malaysia"),
    ("uE7WGvbCyV4", "consumer_service", "shopee lazada scam kurier lambat malaysia"),
    ("nSXkvVfdZ2M", "consumer_service", "shopee lazada scam kurier lambat malaysia"),
    ("dulD_d5Qwf8", "consumer_service", "shopee lazada scam kurier lambat malaysia"),
    ("BEM3kOu7DOU", "consumer_service", "shopee lazada scam kurier lambat malaysia"),
    ("P08psw12cRw", "consumer_service", "shopee lazada scam kurier lambat malaysia"),
    # Domain 5: Public Transport, Roads & Infrastructure
    ("tNe5Cnj5rEM", "transport_infra", "lrt mrt rosak sesak malaysia rapidkl"),
    ("Xkyuny9TuQg", "transport_infra", "lrt mrt rosak sesak malaysia rapidkl"),
    ("PX_pZTqhKaU", "transport_infra", "lrt mrt rosak sesak malaysia rapidkl"),
    ("rO3xYYTWXA0", "transport_infra", "lrt mrt rosak sesak malaysia rapidkl"),
    ("pktYInyXtjw", "transport_infra", "lrt mrt rosak sesak malaysia rapidkl"),
    ("XoxMB_B8vTI", "transport_infra", "jalan berlubang jem teruk malaysia"),
    ("pNFjqpuGQXs", "transport_infra", "jalan berlubang jem teruk malaysia"),
    ("RPiwkyhxvPQ", "transport_infra", "jalan berlubang jem teruk malaysia"),
    ("g9YdhZty20U", "transport_infra", "jalan berlubang jem teruk malaysia"),
    ("e8VFh7e0q38", "transport_infra", "jalan berlubang jem teruk malaysia"),
    # Domain 6: Malaysian Sports, Film & Entertainment
    ("33yIgu-z8Lo", "sports_entertainment", "harimau malaya bola sepak kalah menang"),
    ("VxqJZAz-oIA", "sports_entertainment", "harimau malaya bola sepak kalah menang"),
    ("ihVyxGdAGog", "sports_entertainment", "harimau malaya bola sepak kalah menang"),
    ("y0edJsmv9Ko", "sports_entertainment", "harimau malaya bola sepak kalah menang"),
    ("2_mPre5OU7s", "sports_entertainment", "harimau malaya bola sepak kalah menang"),
    ("gD4Xv4IgV_g", "sports_entertainment", "filem malaysia review terbaik teruk"),
    ("QvoYas2Zwuw", "sports_entertainment", "filem malaysia review terbaik teruk"),
    ("A4BstTIKZyk", "sports_entertainment", "filem malaysia review terbaik teruk"),
    ("K5h3oexiW0k", "sports_entertainment", "filem malaysia review terbaik teruk"),
    ("iaimjJmQyhE", "sports_entertainment", "filem malaysia review terbaik teruk"),
]


def fetch_via_api_v3(video_id: str, api_key: str, max_comments: int = 60) -> List[Tuple[str, str]]:
    """Fetch comments using the official YouTube Data API v3 commentThreads.list endpoint."""
    comments: List[Tuple[str, str]] = []
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    page_token: Optional[str] = None

    while len(comments) < max_comments:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": min(100, max_comments - len(comments)),
            "textFormat": "plainText",
            "order": "relevance",
            "key": api_key,
        }
        if page_token:
            params["pageToken"] = page_token

        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            break
        data = resp.json()
        for item in data.get("items", []):
            cid = item.get("id", "")
            snippet = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
            text = snippet.get("textOriginal") or snippet.get("textDisplay") or ""
            if cid and text:
                comments.append((cid, text))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return comments


def fetch_via_public_innertube(
    session: requests.Session, video_id: str, max_comments: int = 60
) -> List[Tuple[str, str]]:
    """Fetch public comments via YouTube's public Innertube JSON endpoint (/youtubei/v1/next)."""
    try:
        vr = session.get(f"https://www.youtube.com/watch?v={video_id}", timeout=15)
        if vr.status_code != 200:
            return []
    except Exception:
        return []

    key_m = re.search(r'"INNERTUBE_API_KEY":"([^"]+)"', vr.text)
    ver_m = re.search(r'"INNERTUBE_CLIENT_VERSION":"([^"]+)"', vr.text)
    init_m = re.search(r"var ytInitialData = (\{.*?\});</script>", vr.text)
    if not (key_m and ver_m and init_m):
        return []

    key = key_m.group(1)
    client_ver = ver_m.group(1)
    try:
        init_data = json.loads(init_m.group(1))
    except Exception:
        return []

    init_token = None
    try:
        contents = init_data["contents"]["twoColumnWatchNextResults"]["results"]["results"]["contents"]
        for item in contents:
            isr = item.get("itemSectionRenderer", {})
            if isr.get("sectionIdentifier") == "comment-item-section" or "continuationItemRenderer" in str(isr):
                for c in isr.get("contents", []):
                    tok = (
                        c.get("continuationItemRenderer", {})
                        .get("continuationEndpoint", {})
                        .get("continuationCommand", {})
                        .get("token")
                    )
                    if tok:
                        init_token = tok
                        break
    except Exception:
        pass

    if not init_token:
        return []

    comments_map: Dict[str, str] = {}
    next_token: Optional[str] = init_token
    page = 0

    def extract_payloads(obj):
        if isinstance(obj, dict):
            if "commentEntityPayload" in obj:
                cep = obj["commentEntityPayload"]
                props = cep.get("properties", {})
                cid = props.get("commentId")
                txt = props.get("content", {}).get("content", "")
                if cid and txt is not None:
                    comments_map[cid] = txt
            for v in obj.values():
                extract_payloads(v)
        elif isinstance(obj, list):
            for v in obj:
                extract_payloads(v)

    while next_token and len(comments_map) < max_comments and page < 8:
        page += 1
        try:
            cr = session.post(
                f"https://www.youtube.com/youtubei/v1/next?key={key}",
                json={
                    "context": {"client": {"clientName": "WEB", "clientVersion": client_ver}},
                    "continuation": next_token,
                },
                timeout=15,
            )
            if cr.status_code != 200:
                break
            data = cr.json()
        except Exception:
            break

        extract_payloads(data)

        next_token = None
        endpoints = data.get("onResponseReceivedEndpoints", [])
        for ep in endpoints:
            items = (
                ep.get("reloadContinuationItemsCommand", {}).get("continuationItems", [])
                or ep.get("appendContinuationItemsAction", {}).get("continuationItems", [])
            )
            for it in items:
                if "continuationItemRenderer" in it:
                    tok = (
                        it["continuationItemRenderer"]
                        .get("continuationEndpoint", {})
                        .get("continuationCommand", {})
                        .get("token")
                    )
                    if tok:
                        next_token = tok

    return list(comments_map.items())


def normalize_for_dedup(text: str) -> str:
    """Normalize whitespace and case for strict duplicate detection."""
    return re.sub(r"\s+", " ", text.strip().lower())


def run_collection_pipeline(target_valid_samples: int = 1800, max_per_video: int = 50) -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    use_api_v3 = bool(api_key and api_key != "your_youtube_data_api_v3_key_here")
    mode_label = "YouTube_Data_API_v3" if use_api_v3 else "YouTube_Public_Innertube_v1"

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9,ms;q=0.8",
        }
    )

    seen_dedup_texts = set()
    seen_comment_ids = set()
    retained_records = []
    collection_logs = []

    print(f"[Phase 2] Starting data collection in mode: {mode_label}")
    print(f"[Phase 2] Candidate Malaysian YouTube sources: {len(CURATED_VIDEO_SOURCES)}")

    for idx, (video_id, domain, query) in enumerate(CURATED_VIDEO_SOURCES, start=1):
        if len(retained_records) >= target_valid_samples:
            break

        ts_now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if use_api_v3:
            raw_items = fetch_via_api_v3(video_id, api_key, max_comments=max_per_video)
        else:
            raw_items = fetch_via_public_innertube(session, video_id, max_comments=max_per_video)

        retrieved_count = len(raw_items)
        rej_empty_short = 0
        rej_duplicate = 0
        rej_unusable = 0
        vid_retained = 0
        vid_emoji_count = 0

        for cid, raw_text in raw_items:
            if len(retained_records) >= target_valid_samples:
                break

            clean_single_line = raw_text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").strip()

            # 1. Check empty or ultra-short (< 4 chars or < 2 words without emojis)
            words = re.findall(r"[A-Za-z0-9']+", clean_single_line)
            has_em = bool(EMOJI_PATTERN.search(clean_single_line))
            if not clean_single_line or len(clean_single_line) < 4 or (len(words) < 2 and not has_em):
                rej_empty_short += 1
                continue

            # 2. Check URL-only or non-target script (require at least one Latin word for EN/MS/Manglish)
            if URL_ONLY_PATTERN.match(clean_single_line) or not LATIN_WORD_PATTERN.search(clean_single_line):
                rej_unusable += 1
                continue

            # 3. Check duplicate comment ID or duplicate normalized text
            dedup_key = normalize_for_dedup(clean_single_line)
            if cid in seen_comment_ids or dedup_key in seen_dedup_texts:
                rej_duplicate += 1
                continue

            seen_comment_ids.add(cid)
            seen_dedup_texts.add(dedup_key)
            vid_retained += 1
            if has_em:
                vid_emoji_count += 1

            record_id = f"GD_{len(retained_records) + 1:04d}"
            retained_records.append(
                {
                    "id": record_id,
                    "timestamp": ts_now,
                    "platform_source": "YouTube",
                    "source_id": video_id,
                    "raw_text": clean_single_line,
                    "label": "Pending",
                    "annotation_status": "Pending",
                    "annotator_id": "Pending",
                }
            )

        rej_total = rej_empty_short + rej_duplicate + rej_unusable
        collection_logs.append(
            {
                "video_index": idx,
                "source_id": video_id,
                "platform_source": "YouTube",
                "domain": domain,
                "search_query": query,
                "collection_timestamp": ts_now,
                "collection_mode": mode_label,
                "retrieved_count": retrieved_count,
                "rejected_empty_or_short": rej_empty_short,
                "rejected_duplicate": rej_duplicate,
                "rejected_unusable_or_script": rej_unusable,
                "rejected_total": rej_total,
                "retained_count": vid_retained,
                "emoji_comment_count": vid_emoji_count,
            }
        )
        print(
            f"  [{idx:02d}/{len(CURATED_VIDEO_SOURCES)}] video={video_id} ({domain}) | "
            f"retrieved={retrieved_count}, rejected={rej_total}, retained={vid_retained} "
            f"(emoji={vid_emoji_count}) | cumulative={len(retained_records)}"
        )

    # Save dataset/raw_dataset.csv
    raw_csv_path = DATASET_DIR / "raw_dataset.csv"
    fieldnames = [
        "id",
        "timestamp",
        "platform_source",
        "source_id",
        "raw_text",
        "label",
        "annotation_status",
        "annotator_id",
    ]
    with open(raw_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(retained_records)

    # Save logs/collection_log.csv
    log_csv_path = LOGS_DIR / "collection_log.csv"
    log_fields = [
        "video_index",
        "source_id",
        "platform_source",
        "domain",
        "search_query",
        "collection_timestamp",
        "collection_mode",
        "retrieved_count",
        "rejected_empty_or_short",
        "rejected_duplicate",
        "rejected_unusable_or_script",
        "rejected_total",
        "retained_count",
        "emoji_comment_count",
    ]
    with open(log_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()
        writer.writerows(collection_logs)

    tot_ret = sum(x["retrieved_count"] for x in collection_logs)
    tot_rej = sum(x["rejected_total"] for x in collection_logs)
    tot_valid = len(retained_records)
    tot_emoji = sum(x["emoji_comment_count"] for x in collection_logs)
    active_sources = sum(1 for x in collection_logs if x["retained_count"] > 0)

    print("\n" + "=" * 70)
    print(f"[Phase 2 Complete] Total Retrieved: {tot_ret}")
    print(f"[Phase 2 Complete] Total Rejected : {tot_rej}")
    print(f"[Phase 2 Complete] Total Retained : {tot_valid} across {active_sources} distinct videos")
    print(f"[Phase 2 Complete] With Emojis    : {tot_emoji} ({tot_emoji / max(1, tot_valid) * 100:.2f}%)")
    print(f"[Phase 2 Complete] Saved raw dataset to: {raw_csv_path}")
    print(f"[Phase 2 Complete] Saved collection log: {log_csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_collection_pipeline(target_valid_samples=1800, max_per_video=80)
