"""Collect Apple App Store reviews for Google Photos via the public RSS feed (no key)."""
from typing import List, Dict
import requests

APP_ID = "962194608"  # Google Photos on the App Store


def collect(pages: int = 8, country: str = "us") -> List[Dict]:
    out: List[Dict] = []
    for page in range(1, pages + 1):
        url = (
            f"https://itunes.apple.com/{country}/rss/customerreviews/"
            f"page={page}/id={APP_ID}/sortby=mostrecent/json"
        )
        try:
            data = requests.get(url, timeout=15).json()
        except Exception:
            break
        for e in data.get("feed", {}).get("entry", []):
            if "im:rating" not in e:  # skip app-metadata entry
                continue
            out.append(
                {
                    "source": "app_store",
                    "url": e.get("link", {}).get("attributes", {}).get("href", ""),
                    "date": e.get("updated", {}).get("label", ""),
                    "rating": int(e.get("im:rating", {}).get("label", 0) or 0),
                    "text": e.get("content", {}).get("label", ""),
                    "upvotes": int(e.get("im:voteCount", {}).get("label", 0) or 0),
                }
            )
    return out
