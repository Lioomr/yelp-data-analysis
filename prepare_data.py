from __future__ import annotations

"""
Rename raw Yelp files, clean business + review JSONL, and build review_small.json.
Run this before Spark training (also invoked from run_full_pipeline.py).

FAST MODE (recommended for huge review.json):
  PREPARE_FAST_SAMPLE=1  → stream-read review.json once, clean rows only,
  write review_small.json (up to FAST_SAMPLE_MAX_LINES), do NOT rewrite full review.json.

Cleaning policy:
- Reviews: unicode NFKC, strip URLs/emails, lowercase, keep letters/digits/apostrophes,
  collapse whitespace, min/max length.
- Businesses: unique business_id, valid stars in [1,5], non-empty deduped categories
  joined as ', ' for downstream Spark split.
"""

import json
import os
import re
import sys
import unicodedata

from pipeline_config import (
    ROOT_DIR,
    OLD_BUSINESS_NAME,
    OLD_REVIEW_NAME,
    BUSINESS_JSON,
    REVIEW_JSON,
    REVIEW_SMALL_JSON,
    OUTPUTS_DIR,
    DATA_QUALITY_JSON,
    EDA_SUMMARY_JSON,
    CLEANING_VERSION,
    REVIEW_SAMPLE_MAX_LINES,
    FAST_SAMPLE_MAX_LINES,
    FAST_SAMPLE_RAW_CAP,
    MIN_REVIEW_CHARS,
    MAX_REVIEW_CHARS,
    STAR_MIN,
    STAR_MAX,
)


# ============================================================================
# TEXT CLEANING FUNCTIONS (formerly in data_cleaning.py)
# ============================================================================

# URLs and emails (replace with space so tokenizer still sees word boundaries)
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
# Replace non-alphanumeric except apostrophe inside words with space
_NON_WORD_RE = re.compile(r"[^a-z0-9'\s]+", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")


def clean_review_text(
    text: str,
    *,
    min_chars: int = 15,
    max_chars: int = 8000,
) -> str | None:
    """
    Normalize and clean a single review body.
    Returns None if the review should be dropped.
    
    Design goals for Yelp reviews:
    - Preserve word-boundary information (letters, digits, apostrophes in contractions).
    - Remove URLs and noise without destroying sentiment-bearing tokens.
    - Normalize unicode; avoid stripping all punctuation in a way that merges words.
    """
    if not text or not isinstance(text, str):
        return None

    t = unicodedata.normalize("NFKC", text).strip()
    if not t:
        return None

    t = _URL_RE.sub(" ", t)
    t = _EMAIL_RE.sub(" ", t)
    t = t.lower()
    t = _NON_WORD_RE.sub(" ", t)
    t = _SPACE_RE.sub(" ", t).strip()

    # Drop isolated apostrophes / garbage fragments
    if len(t) < min_chars:
        return None
    if len(t) > max_chars:
        t = t[:max_chars].rsplit(" ", 1)[0].strip() if " " in t else t[:max_chars]

    return t or None


def format_categories_csv(categories: str | None) -> str | None:
    """
    Normalize Yelp 'categories' string to comma+space separated unique lowercase tokens.
    Spark uses split(..., ', ') in advanced_yelp_ml.py — this format must match.
    """
    if not categories or not isinstance(categories, str):
        return None
    parts = [p.strip().lower() for p in categories.split(",") if p.strip()]
    if not parts:
        return None
    # stable unique order
    seen = set()
    ordered: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ", ".join(ordered)


# ============================================================================
# DATA PREPARATION FUNCTIONS
# ============================================================================

def _rename_if_needed(old_name: str, new_name: str) -> None:
    old_path = old_name if os.path.isabs(old_name) else os.path.join(ROOT_DIR, old_name)
    new_path = new_name if os.path.isabs(new_name) else os.path.join(ROOT_DIR, new_name)
    if os.path.exists(old_path):
        if os.path.exists(new_path) and os.path.abspath(old_path) != os.path.abspath(new_path):
            print(f"ℹ️  Both {old_name} and {new_name} exist; skipping rename.")
            return
        os.rename(old_path, new_path)
        print(f"✅ Renamed {os.path.basename(old_path)} → {os.path.basename(new_path)}")
    elif os.path.exists(new_path):
        print(f"ℹ️  {os.path.basename(new_path)} already exists.")
    else:
        print(f"⚠️  Warning: {os.path.basename(old_path)} not found (optional if you already renamed).")


def clean_business_file() -> dict:
    if not os.path.exists(BUSINESS_JSON):
        print("⚠️  business.json not found, skipping business cleaning.")
        return {"error": "missing_business"}

    print("Cleaning business.json...")
    cleaned = []
    seen_ids = set()
    dropped_dup = 0
    dropped_bad = 0

    with open(BUSINESS_JSON, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                dropped_bad += 1
                continue

            bid = record.get("business_id")
            stars = record.get("stars")
            cats = record.get("categories")

            if not bid:
                dropped_bad += 1
                continue
            if bid in seen_ids:
                dropped_dup += 1
                continue
            if not isinstance(stars, (int, float)) or not (STAR_MIN <= float(stars) <= STAR_MAX):
                dropped_bad += 1
                continue

            cats_norm = format_categories_csv(cats if isinstance(cats, str) else "")
            if not cats_norm:
                dropped_bad += 1
                continue

            record["categories"] = cats_norm
            cleaned.append(record)
            seen_ids.add(bid)

    with open(BUSINESS_JSON, "w", encoding="utf-8") as f:
        for rec in cleaned:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats = {
        "business_records_written": len(cleaned),
        "business_dropped_duplicate_id": dropped_dup,
        "business_dropped_invalid": dropped_bad,
    }
    print(f"✅ Business cleaned: {len(cleaned)} rows written")
    return stats


def clean_review_file() -> dict:
    if not os.path.exists(REVIEW_JSON):
        print("⚠️  review.json not found, skipping review cleaning.")
        return {"error": "missing_review", "review_records_written": 0}

    print("Cleaning review.json...")
    cleaned = []
    seen = set()
    dropped_json = 0
    dropped_empty = 0
    dropped_dup = 0

    with open(REVIEW_JSON, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                dropped_json += 1
                continue

            bid = record.get("business_id")
            text = record.get("text", "")
            cleaned_text = clean_review_text(
                text if isinstance(text, str) else "",
                min_chars=MIN_REVIEW_CHARS,
                max_chars=MAX_REVIEW_CHARS,
            )
            if not bid or not cleaned_text:
                dropped_empty += 1
                continue

            key = (bid, cleaned_text)
            if key in seen:
                dropped_dup += 1
                continue

            record["text"] = cleaned_text
            if "useful" in record:
                try:
                    record["useful"] = int(record["useful"])
                except (TypeError, ValueError):
                    pass

            cleaned.append(record)
            seen.add(key)

    with open(REVIEW_JSON, "w", encoding="utf-8") as f:
        for rec in cleaned:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats = {
        "review_records_written": len(cleaned),
        "review_dropped_malformed_json": dropped_json,
        "review_dropped_empty_short": dropped_empty,
        "review_dropped_duplicate": dropped_dup,
    }
    print(f"✅ Reviews cleaned: {len(cleaned)} rows written")
    return stats


def clean_reviews_stream_to_small_fast() -> tuple[dict, dict]:
    """
    Read review.json sequentially; clean and dedupe until we have FAST_SAMPLE_MAX_LINES
    outputs or FAST_SAMPLE_RAW_CAP source lines scanned. Writes review_small.json only.
    Does not modify review.json (fast).
    """
    if not os.path.exists(REVIEW_JSON):
        print("⚠️  review.json not found — fast sample mode needs a review JSONL source.")
        return {"error": "missing_review"}, {"review_small_lines": 0}

    dropped_json = 0
    dropped_empty = 0
    dropped_dup = 0
    seen = set()
    written = 0
    raw_seen = 0

    print(
        f"FAST: streaming reviews → review_small.json "
        f"(max {FAST_SAMPLE_MAX_LINES:,} cleaned, scan up to {FAST_SAMPLE_RAW_CAP:,} JSONL lines)..."
    )

    tmp_path = REVIEW_SMALL_JSON + ".tmp"
    try:
        with open(REVIEW_JSON, "r", encoding="utf-8") as fin, open(
            tmp_path, "w", encoding="utf-8"
        ) as fout:
            for line in fin:
                if raw_seen >= FAST_SAMPLE_RAW_CAP:
                    break
                if written >= FAST_SAMPLE_MAX_LINES:
                    break
                raw_seen += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    dropped_json += 1
                    continue

                bid = record.get("business_id")
                text = record.get("text", "")
                cleaned_text = clean_review_text(
                    text if isinstance(text, str) else "",
                    min_chars=MIN_REVIEW_CHARS,
                    max_chars=MAX_REVIEW_CHARS,
                )
                if not bid or not cleaned_text:
                    dropped_empty += 1
                    continue

                key = (bid, cleaned_text)
                if key in seen:
                    dropped_dup += 1
                    continue

                record["text"] = cleaned_text
                if "useful" in record:
                    try:
                        record["useful"] = int(record["useful"])
                    except (TypeError, ValueError):
                        pass

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                seen.add(key)

        os.replace(tmp_path, REVIEW_SMALL_JSON)
    except BaseException:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise

    rev_stats = {
        "review_full_rewrite_skipped": True,
        "mode": "fast_sample_stream",
        "review_raw_lines_scanned": raw_seen,
        "review_records_written_small": written,
        "review_dropped_malformed_json": dropped_json,
        "review_dropped_empty_short": dropped_empty,
        "review_dropped_duplicate": dropped_dup,
    }
    small_stats = {"review_small_lines": written, "rebuilt": True}
    print(f"✅ FAST: wrote {written:,} cleaned reviews → review_small.json (scanned {raw_seen:,} lines)")
    return rev_stats, small_stats


def build_review_small() -> dict:
    if not os.path.exists(REVIEW_JSON):
        print("⚠️  No review.json — cannot build review_small.json")
        return {"review_small_lines": 0}

    if os.path.exists(REVIEW_SMALL_JSON):
        n = sum(1 for _ in open(REVIEW_SMALL_JSON, "r", encoding="utf-8"))
        print(f"ℹ️  review_small.json already exists ({n:,} lines). Rebuild with DELETE_SMALL=1 to overwrite.")
        return {"review_small_lines": n, "skipped": True}

    print(f"⏳ Creating review_small.json (max {REVIEW_SAMPLE_MAX_LINES:,} lines)...")
    count = 0
    with open(REVIEW_JSON, "r", encoding="utf-8") as fin, open(
        REVIEW_SMALL_JSON, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if count >= REVIEW_SAMPLE_MAX_LINES:
                break
            fout.write(line)
            count += 1
    print(f"✅ review_small.json created: {count:,} lines")
    return {"review_small_lines": count}


def rebuild_review_small_force():
    if os.path.exists(REVIEW_SMALL_JSON):
        os.remove(REVIEW_SMALL_JSON)
    return build_review_small()


def write_quality_artifact(biz_stats: dict, rev_stats: dict, small_stats: dict) -> None:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    payload = {
        "cleaning_version": CLEANING_VERSION,
        "min_review_chars": MIN_REVIEW_CHARS,
        "max_review_chars": MAX_REVIEW_CHARS,
        "review_sample_cap": REVIEW_SAMPLE_MAX_LINES,
        "fast_sample_max_lines": FAST_SAMPLE_MAX_LINES,
        "fast_sample_raw_cap": FAST_SAMPLE_RAW_CAP,
        "business": biz_stats,
        "review_full": rev_stats,
        "review_small": small_stats,
        "files": {
            "business_json": os.path.basename(BUSINESS_JSON),
            "review_json": os.path.basename(REVIEW_JSON),
            "review_small_json": os.path.basename(REVIEW_SMALL_JSON),
        },
    }

    with open(DATA_QUALITY_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Lightweight EDA summary for dashboard (no huge arrays)
    eda = {
        "cleaning_version": CLEANING_VERSION,
        "n_businesses": biz_stats.get("business_records_written"),
        "n_reviews_full": (
            None
            if rev_stats.get("review_full_rewrite_skipped")
            else rev_stats.get("review_records_written")
        ),
        "fast_mode": bool(rev_stats.get("review_full_rewrite_skipped")),
        "n_reviews_small": small_stats.get("review_small_lines"),
    }
    with open(EDA_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(eda, f, indent=2)

    print(f"✅ Wrote {DATA_QUALITY_JSON}")


def main():
    # Allow optional full path as cwd
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if os.environ.get("DELETE_SMALL") == "1":
        if os.path.exists(REVIEW_SMALL_JSON):
            os.remove(REVIEW_SMALL_JSON)
            print("🗑️  Removed review_small.json (DELETE_SMALL=1)")

    _rename_if_needed(OLD_BUSINESS_NAME, BUSINESS_JSON)
    _rename_if_needed(OLD_REVIEW_NAME, REVIEW_JSON)

    fast_flag = os.environ.get("PREPARE_FAST_SAMPLE", "").strip().lower()
    fast_on = fast_flag in ("1", "true", "yes", "on")

    if fast_on:
        biz_stats = clean_business_file()
        rev_stats, small_stats = clean_reviews_stream_to_small_fast()
        write_quality_artifact(biz_stats, rev_stats, small_stats)
        print(
            "🚀 FAST preparation done — review_small.json updated; "
            "review.json was NOT fully rewritten.\n"
            "   Train with: docker compose exec spark python /app/advanced_yelp_ml.py"
        )
        return 0

    biz_stats = clean_business_file()
    rev_stats = clean_review_file()

    if os.environ.get("REBUILD_SMALL") == "1":
        small_stats = rebuild_review_small_force()
    else:
        small_stats = build_review_small()

    write_quality_artifact(biz_stats, rev_stats, small_stats)

    print("🚀 Data preparation finished. Next: python run_full_pipeline.py or advanced_yelp_ml.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())