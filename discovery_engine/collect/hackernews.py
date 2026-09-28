"""Collect Hacker News stories and comments about finding photos in Google Photos.

Uses the public Algolia HN Search API (https://hn.algolia.com/api), which needs no key
and is meant for this kind of querying. Comments are where people describe what
they tried; stories rarely carry text, so we keep comments plus stories with text.
"""
import html
import re
import time
from typing import Dict, List

import requests

API = "https://hn.algolia.com/api/v1/search"
QUERIES = [
    "google photos search", "google photos find old photo", "google photos can't find",
    "find an old photo", "photo search memory", "searching my photos",
    "google photos face grouping", "ask photos gemini",
]
PHOTO_WORDS = re.compile(r"\bphotos?\b|\bpictures?\b|\bpics?\b", re.I)


def _clean(s: str) -> str:
    s = re.sub(r"<p>", "\n", s or "")
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"[ \t]+", " ", html.unescape(s)).strip()


def collect(per_query: int = 100) -> List[Dict]:
    out, seen = [], set()
    for q in QUERIES:
        for tag in ("comment", "story"):
            try:
                r = requests.get(API, params={"query": q, "tags": tag, "hitsPerPage": per_query},
                                 timeout=20)
                r.raise_for_status()
            except requests.RequestException:
                continue
            for h in r.json().get("hits", []):
                text = _clean(h.get("comment_text") or h.get("story_text") or "")
                if tag == "story" and h.get("title"):
                    text = (h["title"] + ". " + text).strip()
                if h["objectID"] in seen or len(text) < 60 or not PHOTO_WORDS.search(text):
                    continue
                seen.add(h["objectID"])
                out.append({"source": f"hackernews/{tag}",
                            "url": f"https://news.ycombinator.com/item?id={h['objectID']}",
                            "date": h.get("created_at"), "rating": None, "text": text[:4000],
                            "upvotes": h.get("points") or 0})
            time.sleep(0.5)
    return out
