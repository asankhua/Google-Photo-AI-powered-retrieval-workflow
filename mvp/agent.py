"""
Typed-context parser: the only place Moment Finder uses an LLM.

Turns a parent's own words ("the week my in-laws stayed, before we moved, when he'd
just started walking") into the same filters the chips apply: life stage, home,
people, plus keywords to rank by and keywords to exclude.

Uses Groq when GROQ_API_KEY is set; otherwise a rule-based parser, so the MVP still
works offline. Output is always validated against the library's own vocabulary, so the
LLM can't invent a person, home, or stage.
"""
import json
import os
import re
from typing import Any, Dict

from library import STAGES, terms

MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")


def _groq_available() -> bool:
    return bool(os.environ.get("GROQ_API_KEY"))


def _prompt(lib, text: str) -> str:
    people = "\n".join(f"- {k}: {v}" for k, v in lib["people"].items())
    homes = "\n".join(f"- \"{r['name']}\" ({r['short']}, "
                      f"{'until ' + r['until'] if 'until' in r else 'from ' + r['from']})"
                      for r in lib["residences"])
    stages = "\n".join(f"- {k}: {v[0]} ({v[1]}–{v[2]} months)" for k, v in STAGES.items())
    return f"""A parent is looking for a photo of their child {lib['child']['name']}
(born {lib['child']['birthdate']}). Turn what they remember into search filters.

Allowed people (face groups) and who they are to the parent:
{people}

Homes the family lived in:
{homes}

Life stages:
{stages}

Rules:
- Only use people, homes, and stages from the lists above. Use null / [] if not stated.
- "in-laws" means Amma and Appa. "my mother"/"my mom" means Nani.
- "before we moved" means the old flat; "after we moved" means the new flat.
- stage: set it ONLY when the parent states an age or a developmental cue (newborn,
  crawling, walking, talking, school uniform). Do NOT guess a stage from an occasion such
  as a haircut, festival or trip. Exception: a first birthday is in first_steps.
- residence: set it ONLY when the parent mentions the home or the move.
- keywords: short words for what would be visible in the photo, including actions
  (e.g. "walking", "cake", "rain", "haircut", "crying") and named festivals. Exclude
  people's names and home names.
- exclude: words for things the parent says it was NOT (e.g. "not the party at the hall"
  → ["party", "hall"]).

Return ONLY JSON:
{{"stage": stage key or null, "residence": exact home name or null,
  "people": [names], "keywords": [words], "exclude": [words]}}

What the parent said: "{text}"
"""


def _parse_groq(lib, text: str) -> Dict[str, Any]:
    from groq import Groq
    kw = dict(model=MODEL, temperature=0, max_tokens=2000,  # reasoning tokens count too
              response_format={"type": "json_object"},
              messages=[{"role": "user", "content": _prompt(lib, text)}])
    if "gpt-oss" in MODEL:
        kw["reasoning_effort"] = "low"
    resp = Groq().chat.completions.create(**kw)
    return json.loads(resp.choices[0].message.content)


PEOPLE_ALIASES = [
    (r"\bin[- ]?laws?\b|mother[- ]in[- ]law|father[- ]in[- ]law", ["Amma", "Appa"]),
    (r"\b(my (mother|mom|mum|amma)|nani|grandma)\b", ["Nani"]),
    (r"\b(husband|rahul|his dad|his father)\b", ["Rahul"]),
    (r"\b(cousin|meera)\b", ["Meera"]),
    (r"\bamma\b", ["Amma"]),
    (r"\bappa\b", ["Appa"]),
]
STAGE_RULES = [
    (r"newborn|just born|hospital", "newborn"),
    (r"crawl|sitting up|teeth", "baby"),
    (r"first steps?|started walking|just walking|learning to walk|first birthday|turned one", "first_steps"),
    (r"toddler|started talking|first words|running", "toddler"),
    (r"preschool|school uniform", "preschool"),
]
RESIDENCE_WORDS = {"old", "new", "flat", "house", "home", "apartment", "moved", "move",
                   "koramangala", "hsr", "layout", "week", "day", "time", "stayed",
                   "staying", "visited", "visiting", "remember", "think", "maybe"}


def _parse_rules(lib, text: str) -> Dict[str, Any]:
    low = text.lower()
    out: Dict[str, Any] = {"stage": None, "residence": None, "people": [], "keywords": [], "exclude": []}
    for pat, names in PEOPLE_ALIASES:
        if re.search(pat, low):
            out["people"] += [n for n in names if n not in out["people"]]
    for pat, st in STAGE_RULES:
        if re.search(pat, low):
            out["stage"] = st
            break
    old, new = lib["residences"][0]["name"], lib["residences"][1]["name"]
    if re.search(r"old (flat|house|home|apartment|place)|before we moved|koramangala", low):
        out["residence"] = old
    elif re.search(r"new (flat|house|home|apartment|place)|after we moved|hsr", low):
        out["residence"] = new
    excl_spans = re.findall(r"\bnot (?:the |at the |in the )?([a-z ]+?)(?:[,.;]|$)", low)
    excl = set()
    for span in excl_spans:
        excl |= set(terms(span))
    out["exclude"] = sorted(excl)
    names = {p.lower() for p in lib["people"]} | {lib["child"]["name"].lower()}
    kept = []
    for t in terms(re.sub(r"\bnot (?:the |at the |in the )?[a-z ]+?(?:[,.;]|$)", " ", low),
                      extra_stop=names | RESIDENCE_WORDS | {"law", "laws", "mother", "mom", "husband", "son"}):
        if t not in kept:
            kept.append(t)
    out["keywords"] = kept
    return out


def _validate(lib, raw: Dict[str, Any]) -> Dict[str, Any]:
    homes = {r["name"] for r in lib["residences"]}
    stage = raw.get("stage") if raw.get("stage") in STAGES else None
    res = raw.get("residence") if raw.get("residence") in homes else None
    people = [p for p in (raw.get("people") or []) if p in lib["people"] and p != "Priya"]
    kw = [str(k).lower() for k in (raw.get("keywords") or []) if isinstance(k, str)][:8]
    ex = [str(k).lower() for k in (raw.get("exclude") or []) if isinstance(k, str)][:6]
    return {"stage": stage, "residence": res, "people": people, "keywords": kw, "exclude": ex}


def parse_context(lib, text: str) -> Dict[str, Any]:
    """Returns validated filters plus which parser produced them."""
    if _groq_available():
        try:
            return dict(_validate(lib, _parse_groq(lib, text)), parser="groq")
        except Exception:
            pass
    return dict(_validate(lib, _parse_rules(lib, text)), parser="rules")
