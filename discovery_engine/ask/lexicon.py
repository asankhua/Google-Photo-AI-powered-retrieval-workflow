"""
Counts at scale, no LLM: across every post that describes searching, how many name each kind
of photo, each memory cue, and each cue they say they can't remember. Regex word lists, so the
numbers are approximate (a mention, not a judgement); the quotes in Ask carry the meaning.
"""
import re
from collections import Counter
from typing import Dict, List

PHOTO_KINDS = {
    "People / faces": r"\bfaces?\b|face group|\bpeople\b|\bperson\b",
    "Children & family": r"\bkids?\b|\bbab(y|ies)\b|\bsons?\b|daughters?|\bchild(ren)?\b|grand(son|daughter|kids?)|"
                         r"\bmom\b|\bdad\b|\bmother\b|\bfather\b|\bfamily\b",
    "Screenshots": r"screen ?shots?",
    "Documents & receipts": r"receipts?|documents?|passport|\bbills?\b|invoices?|tickets?|certificates?|\bid card",
    "Trips & places": r"\btrips?\b|vacation|holiday|travel|beach",
    "Events & occasions": r"wedding|birthday|\bparty\b|christmas|graduation|festival|anniversary",
    "Pets": r"\bdogs?\b|\bcats?\b|\bpets?\b|pupp(y|ies)|kittens?",
    "Food": r"\bfood\b|restaurant|recipe|\bdish(es)?\b",
    "Videos": r"\bvideos?\b",
    "Years-old photos": r"years? ago|old (photos?|pictures?|pics)|\bchildhood\b|from (19|20)\d\d",
}
CUES = {
    "A date or year": r"\b(19|20)\d\d\b|\bdates?\b|\byears?\b|\bmonths?\b",
    "Relative time (\"years ago\", \"when she was little\")": r"years? ago|last (year|summer|month|week)|"
                         r"when (he|she|i|we|they) (was|were)|a while (back|ago)|back in",
    "Place": r"\blocations?\b|\bplaces?\b|\bcity\b|\bmap\b|trip to|\bgps\b",
    "People in it": r"\bfaces?\b|\bperson\b|\bpeople\b|my (son|daughter|wife|husband|mom|dad|friend|kids?)",
    "Objects / what's in it": r"\b(dog|cat|car|cake|beach|tree|shirt|flower|sunset|mountain|food)s?\b",
    "Text in it": r"\btext\b|\bwords?\b|written|\bocr\b",
    "An event": r"birthday|wedding|\bparty\b|christmas|holiday|graduation|festival",
    "File, album, device or app": r"file ?names?|\balbums?\b|\bfolders?\b|\bdevice\b|whatsapp|download|camera roll",
}
SEARCH_ACT = re.compile(r"\b(search(ed|ing)?|typed?|look(ed|ing)? for|find(ing)?|scroll(ed|ing)?|locate|"
                        r"remember(ed)?|can'?t recall|forgot)\b", re.I)
# Saying they forgot a DETAIL of the photo (not "I don't know what you did to this app").
FORGET = re.compile(
    r"(don'?t|do not|can'?t|cannot|couldn'?t|could not|never|didn'?t) (remember|recall|know)\s+(exactly\s+)?"
    r"(the |its |what |which |when |where |who |how long )?(exact |specific )?"
    r"(date|day|month|year|when|where|name|place|location|title|file ?name|album|what it|which (one|album|year|day))|"
    r"forg[oe]t(ten)? (the |its |what |which |when |where |who )(date|day|month|year|name|place|it was|was)|"
    r"forgot (when|where|what|which)|no idea (when|where|which)|not sure (when|where|which|what year)",
    re.I,
)
_K = {k: re.compile(v, re.I) for k, v in PHOTO_KINDS.items()}
_C = {k: re.compile(v, re.I) for k, v in CUES.items()}


def searching(docs: List[Dict]) -> List[Dict]:
    """Posts that describe searching for / trying to find something."""
    return [d for d in docs if SEARCH_ACT.search(d["text"])]


def counts(docs: List[Dict]) -> Dict:
    kinds, cues, forgot = Counter(), Counter(), Counter()
    n_forget = 0
    for d in docs:
        t = d["text"]
        kinds.update(k for k, r in _K.items() if r.search(t))
        cues.update(k for k, r in _C.items() if r.search(t))
        sents = [s for s in re.split(r"(?<=[.!?])\s+", t) if FORGET.search(s)]
        if sents:
            n_forget += 1
            forgot.update({k for s in sents for k, r in _C.items() if r.search(s)})
    return {"n": len(docs), "photo_kinds": kinds.most_common(), "cues_mentioned": cues.most_common(),
            "say_they_forgot": n_forget, "forgotten_cues": forgot.most_common()}


def examples(docs: List[Dict], kind: str, table: str = "kinds", k: int = 3) -> List[Dict]:
    r = (_K if table == "kinds" else _C)[kind]
    return [d for d in docs if r.search(d["text"])][:k]


# Facets the planner can switch on for a question: matching posts are boosted, and also
# ranked on their own so every "I forgot..." post can surface even without the search words.
FACETS = {
    "forgot": FORGET,
    "remembered": re.compile(r"\bi (remember|know (it|there|the))|i recall|i('m| am) sure (it|there)|"
                             r"i know (what|who|where|when)", re.I),
    "search_failed": re.compile(r"no results|nothing (comes|came|shows)|(doesn'?t|does not|didn'?t|won'?t|can'?t) "
                                r"(find|show|recogni[sz]e|understand)|useless search|irrelevant|wrong (results|photos)", re.I),
    "workaround": re.compile(r"scroll(ed|ing)? (through|back|for|forever)|gave up|give up|manually|ask(ed)? (my|him|her)|"
                             r"google drive|whatsapp|another app|went back to|had to", re.I),
    "faces_people": re.compile(r"\bfaces?\b|face group|people|person|recogni[sz]", re.I),
    "time_dates": re.compile(r"\b(19|20)\d\d\b|\bdates?\b|\byears?\b|months?|timeline|years? ago", re.I),
}
