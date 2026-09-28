"""
Checks what reviewers say about the new (Gemini / "Ask Photos") search.

Run on any raw review file (list of {"text": ...}); writes data/new_search_mentions.json.
    python -m analyze.new_search_check path/to/raw.json
"""
import json
import os
import re
import sys

NEG = re.compile(r"(new search|ask photos|gemini).{0,120}(bad|worse|worst|useless|terrible|can'?t find|cannot find"
                 r"|fail|slow|old search|bring back)|(bring back|return|want) (the )?old search", re.I)
POS = re.compile(r"(new search|ask photos|gemini).{0,80}(great|love|amazing|awesome|better|helpful)", re.I)


def check(items):
    neg = [x for x in items if NEG.search(x.get("text") or "")]
    pos = [x for x in items if POS.search(x.get("text") or "") and x not in neg]
    return {"reviews_checked": len(items), "negative": len(neg), "positive": len(pos),
            "negative_examples": neg, "positive_examples": pos}


if __name__ == "__main__":
    items = json.load(open(sys.argv[1]))
    out = check(items)
    dest = os.path.join(os.path.dirname(__file__), "..", "data", "new_search_mentions.json")
    json.dump(out, open(dest, "w"), indent=1, default=str)
    print(f"{out['reviews_checked']} reviews · new search: {out['negative']} negative, {out['positive']} positive")
