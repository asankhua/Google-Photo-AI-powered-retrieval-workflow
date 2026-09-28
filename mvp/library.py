"""
Moment Finder core: deterministic filters, live counts, moments, and ranking.

No LLM here. The LLM (agent.py) only turns a parent's own words into the same
filters these functions apply. Everything below recombines metadata a photo
library already has: timestamps, face groups, location, events, source.
"""
import json
import math
import os
import re
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

DATA = os.path.join(os.path.dirname(__file__), "data", "library.json")

# Life stages → age windows in months. Wide and overlapping on purpose (risk 8b):
# children reach milestones at different ages.
STAGES = {
    "newborn": ("Newborn", 0, 3),
    "baby": ("Sitting & crawling", 3, 10),
    "first_steps": ("First steps", 10, 16),
    "toddler": ("Toddler, talking", 16, 30),
    "preschool": ("Preschool", 30, 99),
}
STAGE_ORDER = ["any"] + list(STAGES)

FLOOD_THRESHOLD = 60  # sample library; ~500 in a real 20k+ library

STOPWORDS = set("""a an the and or of in on at to for with from my our his her him he she
it its was were is are be been that this these those when where who what which one
photo photos picture pictures pic pics find show me i we us you just very really about
around near there then than some start started hold holding before after stay stayed
week time day remember think maybe got had first""".split())


def load() -> Dict[str, Any]:
    with open(DATA) as f:
        lib = json.load(f)
    lib["birthdate"] = date.fromisoformat(lib["child"]["birthdate"])
    for it in lib["items"]:
        it["_date"] = date.fromisoformat(it["date"])
    return lib


def age_months(lib, d: date) -> float:
    return (d - lib["birthdate"]).days / 30.44


def stem(w: str) -> str:
    w = w.lower()
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            w = w[: -len(suf)]
            break
    return w


SYNONYMS = {"walk": "step", "cri": "cry", "tear": "cry", "rained": "rain", "rainy": "rain",
            "house": "flat", "home": "flat", "hall": "banquet", "haircut": "hair",
            "barber": "hair", "tonsure": "hair", "smash": "messy", "cream": "messy",
            "wobbly": "step", "balloon": "balloon", "creche": "daycare"}


def terms(text: str, extra_stop: Optional[set] = None) -> List[str]:
    stop = STOPWORDS | (extra_stop or set())
    out = []
    for w in re.findall(r"[a-zA-Z]+", text or ""):
        if w.lower() in stop or len(w) < 3:
            continue
        s = stem(w)
        out.append(SYNONYMS.get(s, s))
    return out


def _item_terms(it) -> set:
    """What the machine can see: vision tags, auto events, location. Never the caption."""
    text = " ".join(it.get("tags", []) + [it.get("event") or "", it["location"]])
    return set(terms(text))


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
def default_filters() -> Dict[str, Any]:
    return {"stage": "any", "shift": 0, "residence": "any", "people": [],
            "hide_clutter": True, "keywords": [], "exclude": []}


def stage_window(f) -> Tuple[float, float]:
    if f["stage"] == "any":
        return (-1, 999)
    _, lo, hi = STAGES[f["stage"]]
    return (lo + f.get("shift", 0), hi + f.get("shift", 0))


def passes(lib, it, f, skip: Optional[str] = None) -> bool:
    if f["hide_clutter"] and it["source"] != "camera" and skip != "hide_clutter":
        return False
    if skip != "stage" and f["stage"] != "any":
        lo, hi = stage_window(f)
        if not (lo <= age_months(lib, it["_date"]) < hi):
            return False
    if skip != "residence" and f["residence"] != "any" and it["residence"] != f["residence"]:
        return False
    if skip != "people" and f["people"]:
        if not all(p in it["people"] for p in f["people"]):
            return False
    return True


def apply(lib, f, skip: Optional[str] = None) -> List[Dict[str, Any]]:
    return [it for it in lib["items"] if passes(lib, it, f, skip)]


def count_with(lib, f, **override) -> int:
    g = dict(f)
    g.update(override)
    return len(apply(lib, g))


def chip_counts(lib, f) -> Dict[str, Dict[str, int]]:
    """Live count for each chip option, given the other active filters."""
    return {
        "stage": {s: count_with(lib, f, stage=s, shift=0 if s != f["stage"] else f["shift"])
                  for s in STAGE_ORDER},
        "residence": {r: count_with(lib, f, residence=r)
                      for r in ["any"] + [x["name"] for x in lib["residences"]]},
        "people": {p: count_with(lib, f, people=sorted(set(f["people"]) | {p}))
                   for p in people_options(lib, f)},
    }


def people_options(lib, f) -> List[str]:
    """People ranked by how often they appear with the child in the current results."""
    child = lib["child"]["name"]
    counts: Dict[str, int] = {}
    for it in apply(lib, f, skip="people"):
        for p in it["people"]:
            if p not in (child, "Priya"):
                counts[p] = counts.get(p, 0) + 1
    ranked = sorted(counts, key=lambda p: -counts[p])
    return sorted(set(ranked[:6]) | set(f["people"]), key=lambda p: -counts.get(p, 0))


def chip_group_order(lib, f) -> List[str]:
    """Show first the chip group that splits the current results most evenly."""
    base = len(apply(lib, f)) or 1
    cc = chip_counts(lib, f)

    def balance(opts: Dict[str, int]) -> float:
        vals = [v for k, v in opts.items() if k != "any" and 0 < v < base]
        if not vals:
            return 0.0
        return -sum((v / base) * math.log(v / base) for v in vals)

    groups = {"stage": balance(cc["stage"]), "residence": balance(cc["residence"]),
              "people": balance(cc["people"])}
    active = {"stage": f["stage"] != "any", "residence": f["residence"] != "any",
              "people": bool(f["people"])}
    return sorted(groups, key=lambda g: (active[g], -groups[g]))


# ---------------------------------------------------------------------------
# No dead ends: relax the least certain filter and say which
# ---------------------------------------------------------------------------
def results_no_dead_end(lib, f) -> Tuple[List[Dict[str, Any]], Optional[str], Dict[str, Any]]:
    res = apply(lib, f)
    if res:
        return res, None, f
    g, steps = dict(f), []

    def done(res, g):
        return res, "Nothing matched everything, so I " + "; then ".join(steps) + ".", g

    # 1. people (face grouping is least certain), most recently added first
    while g["people"]:
        steps.append(f"left out {g['people'][-1]}")
        g = dict(g, people=g["people"][:-1])
        res = apply(lib, g)
        if res:
            return done(res, g)
    # 2. widen the life-stage window
    if g["stage"] != "any":
        g = dict(g, shift=0)
        _, lo, hi = STAGES[g["stage"]]
        widened = [it for it in apply(lib, dict(g, stage="any"))
                   if lo - 3 <= age_months(lib, it["_date"]) < hi + 3]
        if widened:
            steps.append("widened the age window by 3 months each side")
            return done(widened, g)
    # 3. drop the home
    if g["residence"] != "any":
        g = dict(g, residence="any")
        steps.append("included both homes")
        res = apply(lib, g)
        if res:
            return done(res, g)
    g = dict(g, stage="any")
    steps.append("showed everything")
    return done(apply(lib, g), g)


# ---------------------------------------------------------------------------
# Moments: group by day + event, collapse bursts, rank, explain
# ---------------------------------------------------------------------------
ANCHOR_DAYS = 14


def _event_dates(lib) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for it in lib["items"]:
        for t in terms(it.get("event") or ""):
            if t.isalpha() and t not in {"birthday", "trip"}:
                out.setdefault(t, [])
                if it["date"] not in out[t]:
                    out[t].append(it["date"])
    return out


def moments(lib, items: List[Dict[str, Any]], f) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for it in items:
        key = (it["date"], it.get("event") or it["location"])
        groups.setdefault(key, []).append(it)
    out = []
    kw = [SYNONYMS.get(stem(k), stem(k)) for k in f.get("keywords", [])]
    ex = [SYNONYMS.get(stem(k), stem(k)) for k in f.get("exclude", [])]
    # A keyword naming a dated event ("around Diwali") is a time anchor, not content:
    # moments within ANCHOR_DAYS of that event score a little; the event itself less.
    anchors = {k: [date.fromisoformat(d) for d in ds] for k, ds in _event_dates(lib).items() if k in kw}
    for (d, _), its in groups.items():
        its.sort(key=lambda x: (not x["best_frame"], x["time"], x["id"]))
        best = its[0]
        t = set().union(*(_item_terms(x) for x in its))
        near = [k for k, ds in anchors.items() if any(abs((best["_date"] - a).days) <= ANCHOR_DAYS for a in ds)]
        hits = [k for k in kw if k in t and k not in anchors]
        misses = [k for k in ex if k in t]
        people = sorted({p for x in its for p in x["people"]} - {lib["child"]["name"], "Priya"})
        score = 3.0 * len(hits) + 1.5 * len(near) - 4.0 * len(misses) + math.log2(1 + len(its)) \
            + (0.5 if best.get("event") else 0.0)
        hits = hits + [f"near {k.title()}" for k in near]
        out.append({
            "id": f"m-{d}-{best['burst_id']}",
            "date": best["_date"],
            "best": best,
            "items": its,
            "people": people,
            "hits": hits,
            "excluded_hits": misses,
            "score": score,
            "untagged": sum(1 for x in its if lib["child"]["name"] not in x["people"]
                            and x["source"] == "camera"),
        })
    out.sort(key=lambda m: (-m["score"], m["date"]))
    return out


def explain(lib, m) -> str:
    d = m["date"]
    res = next(r["short"] for r in lib["residences"] if r["name"] == m["best"]["residence"])
    where = m["best"]["location"]
    where = res if where == m["best"]["residence"] else f"{where} (living at {res.lower()})"
    who = f"with {' & '.join(m['people'])}" if m["people"] else "just family"
    age = age_months(lib, d)
    age_s = f"~{age:.0f} months" if age < 30 else f"~{age / 12:.1f} years"
    n = len(m["items"])
    return f"{where} · {who} · {d.strftime('%d %b %Y')} · {lib['child']['name']} {age_s} · {n} photo{'s' if n > 1 else ''}"


def days_around(lib, m, f, days: int = 3) -> List[Dict[str, Any]]:
    lo, hi = m["date"] - timedelta(days=days), m["date"] + timedelta(days=days)
    its = [it for it in lib["items"] if lo <= it["_date"] <= hi
           and (not f["hide_clutter"] or it["source"] == "camera")]
    ms = moments(lib, its, f)
    ms.sort(key=lambda x: x["date"])
    return ms


# ---------------------------------------------------------------------------
# Standard search (the baseline parents use today)
# ---------------------------------------------------------------------------
def native_search(lib, query: str) -> List[Dict[str, Any]]:
    """Literal match on people and visible content, newest first, every frame shown."""
    q = terms(query)
    names = {p.lower() for p in lib["people"]} | {lib["child"]["name"].lower()}
    words = [w.lower() for w in re.findall(r"[a-zA-Z]+", query)]
    people = [w.title() for w in words if w in names]
    other = [t for t in q if t not in {p.lower() for p in people}]
    res = []
    for it in lib["items"]:
        if people and not all(p in it["people"] for p in people):
            continue
        if other and not any(t in _item_terms(it) for t in other) and not people:
            continue
        res.append(it)
    res.sort(key=lambda x: (x["date"], x["time"]), reverse=True)
    return res


def is_flooded(lib, query: str) -> Tuple[bool, int, str]:
    """Trigger: a search for the child, a stage/category word, >threshold, or zero results."""
    n_child = sum(1 for it in lib["items"] if lib["child"]["name"] in it["people"])
    res = native_search(lib, query)
    low = query.lower()
    child = lib["child"]["name"].lower() in low or re.search(r"\b(my son|him|he|my baby)\b", low)
    category = re.search(r"\b(first|steps?|walk\w*|birthday|haircut|daycare|school|baby)\b", low)
    if child:
        return True, n_child, f"{lib['child']['name']} is in {n_child} photos."
    if len(res) == 0:
        return True, 0, "No photos matched those words."
    if len(res) > FLOOD_THRESHOLD or category:
        return True, len(res), f"That matches {len(res)} photos."
    return False, len(res), ""


def match_saved(saved: Dict[str, Dict[str, Any]], query: str) -> List[Tuple[str, Dict[str, Any]]]:
    qt = set(terms(query))
    out = []
    for label, m in saved.items():
        lt = set(terms(label))
        if lt and (lt <= qt or len(lt & qt) >= max(1, len(lt) - 1)):
            out.append((label, m))
    return out


# ---------------------------------------------------------------------------
# "Which one?": when the top two moments are too close to call, ask one question
# built from what differs between them (Evaluation stage: look-alikes)
# ---------------------------------------------------------------------------
CLOSE_CALL = 2.0
_GENERIC = {"baby", "child", "koramangala", "hsr", "layout", "flat", "old", "new"}


def _moment_tags(m) -> set:
    return {t for x in m["items"] for t in x.get("tags", [])} - _GENERIC


def _place(lib, m) -> str:
    b = m["best"]
    if b["location"] == b["residence"]:
        return "at home (" + next(r["short"] for r in lib["residences"] if r["name"] == b["residence"]).lower() + ")"
    return "at " + b["location"].split(",")[0]


def which_one(lib, ms) -> Optional[Dict[str, Any]]:
    """Returns {question, options:[{label, moment_id, keywords, exclude}]} or None."""
    if len(ms) < 2 or ms[0]["score"] - ms[1]["score"] >= CLOSE_CALL:
        return None
    a, b = ms[0], ms[1]
    ta, tb = _moment_tags(a), _moment_tags(b)
    da, db = sorted(ta - tb)[:3], sorted(tb - ta)[:3]
    if not da and not db:
        return None
    # the place counts as a difference too ("banquet hall" vs home)
    pa, pb = _place(lib, a), _place(lib, b)
    ka = terms(" ".join(da + ([pa] if pa != pb else [])), extra_stop={"home", "flat", "old", "new"})
    kb = terms(" ".join(db + ([pb] if pa != pb else [])), extra_stop={"home", "flat", "old", "new"})

    def label(m, d, place):
        bits = [place] + (["with " + " & ".join(m["people"][:3])] if m["people"] else []) + ([", ".join(d)] if d else [])
        return f"{m['date'].strftime('%d %b %Y')}: " + " · ".join(bits)

    return {"question": "These two look alike. Which is closer to what you remember?",
            "options": [{"label": label(a, da, pa), "moment_id": a["id"],
                         "keywords": list(dict.fromkeys(ka)), "exclude": list(dict.fromkeys(kb))},
                        {"label": label(b, db, pb), "moment_id": b["id"],
                         "keywords": list(dict.fromkeys(kb)), "exclude": list(dict.fromkeys(ka))}]}
