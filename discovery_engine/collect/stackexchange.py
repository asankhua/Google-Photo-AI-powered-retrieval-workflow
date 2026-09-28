"""Collect Stack Exchange questions about finding photos in Google Photos.

Uses the public Stack Exchange API v2.3 (no key needed for up to 300 requests a day).
Sites: Web Applications, Android Enthusiasts, Photography. The question body is the
user's own account of what they remember and what they tried.
"""
import html
import re
import time
from typing import Dict, List

import requests

API = "https://api.stackexchange.com/2.3/search/advanced"
SITES = ["webapps", "android", "photo", "superuser", "apple"]
QUERIES = ["google photos find", "google photos search", "find old photo", "search photos by",
           "google photos face", "locate a photo"]


def _clean(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def collect(pagesize: int = 50) -> List[Dict]:
    out, seen = [], set()
    for site in SITES:
        for q in QUERIES:
            try:
                r = requests.get(API, params={"q": q, "site": site, "pagesize": pagesize,
                                              "order": "desc", "sort": "relevance", "filter": "withbody"},
                                 timeout=20)
                r.raise_for_status()
                data = r.json()
            except (requests.RequestException, ValueError):
                continue
            for it in data.get("items", []):
                key = (site, it["question_id"])
                if key in seen:
                    continue
                seen.add(key)
                text = _clean(it.get("title", "")) + ". " + _clean(it.get("body", ""))
                out.append({"source": f"stackexchange/{site}", "url": it.get("link"),
                            "date": str(it.get("creation_date")), "rating": None, "text": text[:4000],
                            "upvotes": it.get("score", 0)})
            if data.get("backoff"):
                time.sleep(int(data["backoff"]))
            time.sleep(0.3)
    return out
