"""Retrieval-intent filters shared by the pipeline and the Ask corpus (no LLM, free)."""
import re

# Reviews likely to be about retrieval — used by --filter to focus the LLM budget.
# Broadened to capture more retrieval-intent phrasing (where-is / missing / no-results
# / date-scrolling / album-hunting) so the four discovery questions get richer answers.
RETRIEVAL_KW = re.compile(
    r"find|search|locate|remember|looking for|look for|scroll|old photo|old picture|"
    r"can'?t find|can'?t locate|where('?s| is| are)|no results|doesn'?t show|"
    r"not show|missing|disappear|went missing|years ago|trying to find|"
    r"sort|filter by|by date|album|gallery|memory|memories",
    re.I,
)

# Stricter filter for forum sources (HN, Stack Exchange, community), where most posts
# mention photos in passing. Keep posts where a find/search verb sits next to "photo"
# and a photo app or library is named; rank by how on-topic they are, not by upvotes
# (upvotes favour popular off-topic launches).
FIND_PHOTO = re.compile(
    r"\b(find|finding|locate|search(ing)?|looking for|look for|dig(ging)? up|scroll(ing)?|get back|"
    r"can'?t remember|forgot)\W+(\w+\W+){0,6}?(photos?|pictures?|pics?|images?|screenshots?)\b|"
    r"\b(photos?|pictures?|pics?|screenshots?)\W+(\w+\W+){0,6}?(find|locate|search|remember)\w*",
    re.I,
)
PHOTO_APP = re.compile(r"google photos|photos app|apple photos|icloud photos|camera roll|gallery|"
                       r"my photos|photo library|lightroom|picasa|immich|photoprism", re.I)
LAUNCH = re.compile(r"^(show|launch) hn:", re.I)


def forum_score(text: str) -> int:
    if LAUNCH.search(text) or not PHOTO_APP.search(text):
        return 0
    hits = len(FIND_PHOTO.findall(text))
    return hits + (2 if hits and re.search(r"google photos", text, re.I) else 0)
