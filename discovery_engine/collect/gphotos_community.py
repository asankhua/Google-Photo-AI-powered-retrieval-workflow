"""Collect threads from the Google Photos Help Community (support.google.com/photos).

Respects robots.txt: the community *search* is disallowed, so we don't use it. We read
the public thread listing (the latest ~20 threads) plus any thread URLs you list in
data/community_threads.txt (one per line, e.g. found by hand), one request a second.
Each thread page embeds its title and question text; we extract those.
"""
import codecs
import os
import re
import time
from typing import Dict, List

import requests

BASE = "https://support.google.com"
LISTINGS = ["/photos/threads?hl=en", "/photos/community?hl=en"]
UA = {"User-Agent": "Mozilla/5.0 (compatible; gphotos-discovery-research/0.1)"}
URL_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "community_threads.txt")


def _decode(s: str) -> str:
    try:
        s = codecs.decode(s.replace("\\\\", "\\"), "unicode_escape")
        s = s.encode("latin-1", "ignore").decode("utf-8", "ignore")
    except Exception:
        pass
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s)).strip()


def _thread(path: str) -> Dict:
    r = requests.get(BASE + path, headers=UA, timeout=20)
    r.raise_for_status()
    page = r.text
    m = re.search(r"var title='((?:[^'\\]|\\.)*)'", page)
    title = _decode(m.group(1)) if m else ""
    body, category = "", ""
    if title:
        # the thread blob: ...\x22<title>\x22,<id>,null,[0],\x22<question text>\x22,\x22en\x22...
        t = re.escape(m.group(1).replace("\\'", "'"))
        b = re.search(t + r"\\x22,\d+,null,\[\d\],\\x22(.*?)\\x22,\\x22[a-z-]+\\x22", page, re.S)
        if b:
            body = _decode(b.group(1))
        c = re.search(r"\\x22(photos_[a-z_]+)\\x22", page)
        category = c.group(1) if c else ""
    return {"title": title, "body": body, "category": category}


def collect(max_threads: int = 60) -> List[Dict]:
    paths = []
    for listing in LISTINGS:
        try:
            html = requests.get(BASE + listing, headers=UA, timeout=20).text
            paths += re.findall(r"/photos/thread/\d+", html)
        except requests.RequestException:
            pass
        time.sleep(1)
    if os.path.exists(URL_FILE):
        for line in open(URL_FILE):
            m = re.search(r"/photos/thread/\d+", line)
            if m:
                paths.append(m.group(0))
    out = []
    for p in list(dict.fromkeys(paths))[:max_threads]:
        try:
            t = _thread(p + "?hl=en")
        except requests.RequestException:
            continue
        time.sleep(1)
        text = (t["title"] + ". " + t["body"]).strip(". ")
        if len(text) < 30:
            continue
        out.append({"source": "gphotos_community",  # page-level category tags weren't reliable
                    "url": BASE + p, "date": None, "rating": None, "text": text[:4000], "upvotes": 0})
    return out
