"""Collect YouTube comments on videos about searching Google Photos.

Uses the official YouTube Data API v3, which needs a free key: set YOUTUBE_API_KEY in
discovery_engine/.env (console.cloud.google.com -> APIs -> YouTube Data API v3 ->
Credentials -> API key). Costs about 100 quota units a search, 1 per comment page.
"""
import os
from typing import Dict, List

import requests

API = "https://www.googleapis.com/youtube/v3"
QUERIES = ["google photos search tips", "how to find old photos google photos",
           "google photos ask photos gemini", "google photos face groups not working"]


def collect(videos_per_query: int = 5, comments_per_video: int = 100) -> List[Dict]:
    key = os.environ.get("YOUTUBE_API_KEY")
    if not key:
        raise RuntimeError("Missing YOUTUBE_API_KEY (free key from Google Cloud console)")
    out, seen = [], set()
    for q in QUERIES:
        s = requests.get(f"{API}/search", params={"part": "snippet", "q": q, "type": "video",
                                                   "maxResults": videos_per_query, "key": key}, timeout=20)
        s.raise_for_status()
        for v in s.json().get("items", []):
            vid = v["id"]["videoId"]
            c = requests.get(f"{API}/commentThreads", params={"part": "snippet", "videoId": vid,
                                                               "maxResults": comments_per_video,
                                                               "textFormat": "plainText", "key": key}, timeout=20)
            if c.status_code != 200:  # comments disabled
                continue
            for th in c.json().get("items", []):
                sn = th["snippet"]["topLevelComment"]["snippet"]
                if th["id"] in seen or len(sn.get("textDisplay", "")) < 40:
                    continue
                seen.add(th["id"])
                out.append({"source": "youtube", "url": f"https://www.youtube.com/watch?v={vid}&lc={th['id']}",
                            "date": sn.get("publishedAt"), "rating": None, "text": sn["textDisplay"][:4000],
                            "upvotes": sn.get("likeCount", 0)})
    return out
