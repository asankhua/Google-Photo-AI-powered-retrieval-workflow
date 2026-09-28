"""
Segment filters: keep only feedback from the segment under study.

"parents": the review mentions the reviewer's child (kid, son, daughter, baby,
toddler, twins, grandchild). This is how the engine checks whether the chosen
segment (parents searching a library full of their child) shows up in real feedback.
"""
import re

SEGMENTS = {
    "parents": re.compile(
        r"\b(kids?|child(ren)?|son|daughter|bab(y|ies)|toddlers?|newborn|"
        r"grand ?(son|daughter|kids?|children)|my boy|my girl|twins?)\b",
        re.I,
    ),
}


def in_segment(text: str, segment: str) -> bool:
    return bool(SEGMENTS[segment].search(text or ""))
