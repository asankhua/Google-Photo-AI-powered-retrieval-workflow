"""Collect Google Play reviews for Google Photos."""
from typing import List, Dict

APP_ID = "com.google.android.apps.photos"


def collect(count: int = 300, lang: str = "en", country: str = "us",
            sorts=("newest",)) -> List[Dict]:
    """Return normalized review dicts. Requires: pip install google-play-scraper

    sorts: any of "newest", "relevant". Reviews are deduplicated by review id.
    """
    from google_play_scraper import reviews, Sort

    sort_map = {"newest": Sort.NEWEST, "relevant": Sort.MOST_RELEVANT}
    seen, out = set(), []
    for s in sorts:
        result, _ = reviews(APP_ID, lang=lang, country=country, sort=sort_map[s], count=count)
        for r in result:
            if r["reviewId"] in seen:
                continue
            seen.add(r["reviewId"])
            out.append(
                {
                    "source": "play_store",
                    "url": f"https://play.google.com/store/apps/details?id={APP_ID}",
                    "date": str(r.get("at")),
                    "rating": r.get("score"),
                    "text": r.get("content") or "",
                    "upvotes": r.get("thumbsUpCount", 0),
                    "country": country,
                }
            )
    return out
