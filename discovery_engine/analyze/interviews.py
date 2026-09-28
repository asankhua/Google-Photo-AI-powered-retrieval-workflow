"""
Interview-analysis agent.

Reads interview or test-session transcripts and answers the discovery questions with
evidence, in three steps:

  1. CODE   one LLM call per transcript, using the engine's taxonomy (cues, query
            strategies, failure modes) plus interview-only fields: trigger, first
            query, how the memory was turned into a query, workaround, outcome.
  2. CHECK  every quote the model returns is matched against the transcript text.
            Quotes that aren't in the transcript are dropped (the model can't
            invent evidence), and the drop count is reported.
  3. ANSWER counts are computed in Python (not by the LLM) and compared with the
            public-feedback findings (agree / differ). One final LLM call writes
            the answers, allowed to cite only the verified quotes it is given.

Transcripts marked simulated are labelled so everywhere in the output.
"""
import json
import os
import re
import time
from collections import Counter
from typing import Dict, List, Optional

from groq import RateLimitError

from classify.classifier import _get_client, _safe_json
from classify.taxonomy import CUE_TYPES, FAILURE_MODES, PHOTO_TYPES, QUERY_STRATEGIES

MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
FALLBACK = "openai/gpt-oss-20b"

QUESTIONS = [
    ("Q1", "What kinds of old photos do users struggle to retrieve?"),
    ("Q2", "What information do people actually remember about a photo?"),
    ("Q3", "What information have they forgotten?"),
    ("Q4", "How do users formulate searches when their memory is incomplete?"),
    ("Q5", "Where does the search break, and what do people do when it fails?"),
]

OUTCOMES = ["found", "found_after_long_search", "substitute_sent", "asked_someone", "gave_up", "unclear"]


def _chat(prompt: str, max_tokens: int = 3000) -> Dict:
    """One JSON-mode call; waits on the per-minute limit, falls back to 20b on the daily cap."""
    global MODEL
    for attempt in range(6):
        try:
            r = _get_client().chat.completions.create(
                model=MODEL, temperature=0, max_tokens=max_tokens, reasoning_effort="low",
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}])
            return _safe_json(r.choices[0].message.content)
        except RateLimitError as e:
            if "tokens per day" in str(e):
                if MODEL == FALLBACK:
                    raise
                MODEL = FALLBACK
                continue
            time.sleep(20 * (attempt + 1))
    raise RuntimeError("Groq kept rate-limiting; try again in a few minutes")


# ---------- 1. code one transcript ----------

def _code_prompt(text: str) -> str:
    return f"""You are a UX researcher coding one interview or usability-test transcript about \
finding old photos (Google Photos). Code ONLY what the participant says; ignore the interviewer's \
wording. If something isn't said, use null or []. Quotes must be copied word for word from the \
participant's answers (8-30 words each), never paraphrased.

CUE TYPES: {", ".join(CUE_TYPES)}
QUERY STRATEGIES: {", ".join(QUERY_STRATEGIES)}
PHOTO TYPES: {", ".join(PHOTO_TYPES)}
FAILURE MODES (first funnel stage that broke): {"; ".join(f"{k}: {v[:110]}" for k, v in FAILURE_MODES.items() if k != "NOT_RETRIEVAL")}; NONE if it didn't break
OUTCOMES: {", ".join(OUTCOMES)}

Return ONLY JSON with these keys:
{{
  "participant": "id or first name",
  "target_moment": "the photo they were looking for, in a few words",
  "photo_type": one PHOTO TYPE,
  "trigger": "who asked / why they searched, or null",
  "deadline": "by when, or null",
  "remembered": [{{"cue": CUE TYPE, "detail": "what exactly they remembered", "quote": "verbatim"}}],
  "forgotten": [{{"cue": CUE TYPE, "detail": "what they couldn't recall", "quote": "verbatim"}}],
  "anchors": ["how they worked out WHEN it was, e.g. 'before we moved', 'her age'"],
  "first_query": "exact first thing typed or tapped, or null",
  "later_queries": ["exact later queries, in order"],
  "query_strategies": [QUERY STRATEGIES used, in order],
  "formulation": "one sentence: how they turned a partial memory into a search",
  "failure_mode": one FAILURE MODE or "NONE",
  "failure_detail": "one sentence: what went wrong on screen",
  "workaround": "what they did when search failed, or null",
  "outcome": one OUTCOME,
  "substitute_sent": true/false/null,
  "photo_on_someone_elses_phone": true/false/null,
  "searches_last_month": number or null,
  "first_search_found": number or null,
  "feeling": "one or two words, or null",
  "product_feedback": "only for usability tests: what they liked/disliked about the tool tried, else null",
  "key_quotes": ["2-4 most telling verbatim quotes"]
}}

TRANSCRIPT:
\"\"\"{text}\"\"\""""


def _norm(s: str) -> str:
    s = s.lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("—", "-").replace("–", "-").replace("…", "...")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]+", " ", s)).strip()


SPEAKER = re.compile(r"^\s*\**\s*(interviewer|moderator|facilitator|researcher|q)\s*:", re.I)
HEADER = re.compile(r"^\s*\**\s*(persona|seed|model|result)[^:]*:", re.I)


def clean_transcript(text: str) -> str:
    """Drop set-up lines (persona brief, seed review, model) so only the session is coded."""
    return "\n".join(l for l in text.splitlines() if not HEADER.match(l))


def participant_text(text: str) -> str:
    """Only the participant's words: interviewer / moderator turns removed (for quote checks)."""
    keep, speaking = [], True
    for line in clean_transcript(text).splitlines():
        if SPEAKER.match(line):
            speaking = False
        elif line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            # test-session tables: | step | said | action | screen |; keep only what was said
            if len(cells) >= 3 and cells[0].isdigit():
                keep.append(cells[1])
            continue
        elif re.match(r"^\s*\*\*[^*]+:\*\*", line):
            speaking = True  # a participant turn
        if speaking:
            keep.append(line)
    return "\n".join(keep)


def _verified(quote: Optional[str], transcript_norm: str) -> bool:
    q = _norm(quote or "")
    return len(q.split()) >= 4 and q in transcript_norm


def code_transcript(text: str, pid: str, simulated: bool) -> Dict:
    """LLM-code one transcript, then drop any quote that isn't in the transcript."""
    c = _chat(_code_prompt(clean_transcript(text)[:24000]))
    c["id"], c["simulated"], c["model"] = pid, simulated, MODEL
    return check(c, text)


def check(c: Dict, text: str) -> Dict:
    """Drop quotes that aren't in the participant's own words, and queries not in the session."""
    tn = _norm(participant_text(text))
    dropped = c.get("quotes_dropped", 0)
    for key in ("remembered", "forgotten"):
        keep = []
        for item in c.get(key) or []:
            if not isinstance(item, dict) or item.get("cue") not in CUE_TYPES:
                continue
            if item.get("quote") and not _verified(item["quote"], tn):
                dropped += 1
                item["quote"] = None  # keep the cue, lose the unproven quote
            keep.append(item)
        c[key] = keep
    quotes = [q for q in (c.get("key_quotes") or []) if isinstance(q, str)]
    c["key_quotes"] = [q for q in quotes if _verified(q, tn)]
    dropped += len(quotes) - len(c["key_quotes"])
    full = _norm(clean_transcript(text))
    qs = [c.get("first_query")] + list(c.get("later_queries") or [])
    c["queries_verified"] = [q for q in qs if isinstance(q, str) and _norm(q) and _norm(q) in full]
    c["query_strategies"] = [s for s in (c.get("query_strategies") or []) if s in QUERY_STRATEGIES]
    c["quotes_dropped"] = dropped
    return c


# ---------- 2. aggregate (plain counting, no LLM) ----------

def _count(codes, key, field="cue"):
    """Participants mentioning each cue (counted once per participant)."""
    cnt = Counter()
    for c in codes:
        cnt.update({i.get(field) for i in c.get(key) or [] if i.get(field)})
    return cnt.most_common()


def aggregate(codes: List[Dict]) -> Dict:
    n = len(codes)
    first = Counter()
    for c in codes:
        if c.get("query_strategies"):
            first[c["query_strategies"][0]] += 1
    return {
        "n": n,
        "simulated": sum(1 for c in codes if c.get("simulated")),
        "photo_types": Counter(c.get("photo_type") for c in codes).most_common(),
        "remembered": _count(codes, "remembered"),
        "forgotten": _count(codes, "forgotten"),
        "strategies_any": Counter(s for c in codes for s in set(c.get("query_strategies") or [])).most_common(),
        "strategies_first": first.most_common(),
        "failure_modes": Counter(c.get("failure_mode") for c in codes).most_common(),
        "outcomes": Counter(c.get("outcome") for c in codes).most_common(),
        "substitute_sent": sum(1 for c in codes if c.get("substitute_sent")),
        "someone_elses_phone": sum(1 for c in codes if c.get("photo_on_someone_elses_phone")),
        "first_queries": [(c["id"], c.get("first_query")) for c in codes if c.get("first_query")],
        "quotes_dropped": sum(c.get("quotes_dropped", 0) for c in codes),
    }


# ---------- 3. triangulate against the public-feedback findings ----------

def _shares(pairs, total):
    return {k: round(100 * v / total) for k, v in pairs} if total else {}


def triangulate(agg: Dict, public: Dict, label: str) -> List[Dict]:
    """Compare the top cues/strategies in interviews vs a discovery-engine summary."""
    total = public.get("retrieval_related") or 0
    rows = []
    for dim, ikey, pkey in [("Remembered", "remembered", "remembered_cues_top"),
                            ("Forgotten", "forgotten", "forgotten_cues_top"),
                            ("Search strategy", "strategies_any", "query_strategies_top")]:
        i_sh = _shares(agg[ikey], agg["n"])
        p_sh = _shares(public.get(pkey, []), total)
        i_top = [k for k, _ in agg[ikey][:3]]
        p_top = [k for k, _ in public.get(pkey, [])[:3]]
        for k in dict.fromkeys(i_top + p_top):
            a, b = i_sh.get(k, 0), p_sh.get(k, 0)
            verdict = ("agree" if abs(a - b) <= 20 else "interviews higher" if a > b else "public higher")
            rows.append({"dimension": dim, "item": k, "interviews_pct": a, f"{label}_pct": b,
                         "public_n": total, "verdict": verdict})
    return rows


# ---------- 4. answer the questions ----------

# per question: which coded fields, which counts and which triangulation rows it needs
Q_SPEC = {
    "Q1": (["target_moment", "photo_type", "trigger", "deadline", "key_quotes"], ["photo_types"], None),
    "Q2": (["target_moment", "remembered", "anchors"], ["remembered"], "Remembered"),
    "Q3": (["target_moment", "forgotten"], ["forgotten"], "Forgotten"),
    "Q4": (["remembered", "queries_verified", "formulation", "query_strategies"],
           ["strategies_first", "strategies_any"], "Search strategy"),
    "Q5": (["failure_mode", "failure_detail", "workaround", "outcome", "substitute_sent",
            "photo_on_someone_elses_phone", "key_quotes"],
           ["failure_modes", "outcomes", "substitute_sent", "someone_elses_phone"], None),
}


Q_HINT = {
    "Q1": "Go beyond the photo_type label: group the target_moment values into kinds (e.g. a child's "
          "firsts, occasions) and say what makes them hard.",
    "Q4": "Show how remembered fragments turned into the typed queries; cite the queries themselves "
          "(queries_verified) as evidence, one per line.",
}


def _brief(item, fields):
    out = {"id": item["id"]}
    for f in fields:
        v = item.get(f)
        if f in ("remembered", "forgotten"):
            v = [{k: i.get(k) for k in ("cue", "detail", "quote") if i.get(k)} for i in v or []]
        if v not in (None, [], ""):
            out[f] = v
    return out


def _quote_pools(codes):
    """Per participant: all their checked quotes and queries (normalised), to verify citations."""
    return {c["id"]: _norm(" | ".join(q for q in (c.get("key_quotes") or []) + (c.get("queries_verified") or []) +
                                      [i.get("quote") or "" for i in (c.get("remembered") or []) +
                                       (c.get("forgotten") or [])]))
            for c in codes}


def _cite(evidence, pools, limit=6):
    """Split evidence into single quotes and label each with the participant who really said it.
    Returns (lines, dropped). A quote nobody said is dropped."""
    lines, dropped, seen = [], 0, set()
    for e in evidence or []:
        # outer quotes are usually straight, with curly quotes inside the quote itself
        parts = re.findall(r'"([^"]{3,})"', e or "") or re.findall(r'“([^”]{3,})”', e or "")
        if not parts:
            dropped += 1
            continue
        for q in parts:
            nq = _norm(q)
            owners = [pid for pid, pool in pools.items() if nq and nq in pool]
            if not owners:
                dropped += 1
            elif nq not in seen:
                seen.add(nq)
                lines.append(f'[{", ".join(owners[:3])}] "{q.strip()}"')
    return lines[:limit], dropped


def answer(codes: List[Dict], agg: Dict, tri: Dict[str, List[Dict]], public: Dict[str, Dict]) -> Dict:
    """One small call per question (keeps each request under the free tier's 8k tokens a minute)."""
    pools = _quote_pools(codes)
    note = (f"{agg['simulated']} of {agg['n']} transcripts are AI-simulated, not real people; "
            "say so in the answer." if agg["simulated"] else "All transcripts are real sessions.")
    answers = []
    for qid, q in QUESTIONS:
        fields, counts, dim = Q_SPEC[qid]
        rows = {name: [r for r in t if r["dimension"] == dim] for name, t in tri.items()} if dim else {}
        extra = ""
        if qid == "Q4":
            extra = "PUBLIC QUERY EXAMPLES: " + json.dumps(
                {n: (s.get("query_examples") or [])[:12] for n, s in public.items()})
        if qid == "Q5":
            extra = "PUBLIC FAILURE MODES (share of retrieval-related posts): " + json.dumps(
                {n: [[o["failure_mode"], o["share_pct"]] for o in s.get("opportunities_ranked", [])]
                 for n, s in public.items()})
        if qid == "Q1":
            extra = "PUBLIC PHOTO TYPES: " + json.dumps(
                {n: [s.get("retrieval_related"), (s.get("photo_types_top") or [])[:5]] for n, s in public.items()})
        prompt = f"""You are the research lead. Answer ONE discovery question from coded interviews.
Rules: use the COUNTS given (computed, n={agg['n']}; write "k of {agg['n']}"). Cite participant ids like [P2, T1].
Evidence lines must quote word for word from the quote / key_quotes / queries_verified fields below
(a typed query counts as evidence); never invent a quote or number.
{note}
Then say whether public feedback agrees or differs, using the COMPARISON rows (percentages of
participants vs percentage of retrieval-related public posts). If there are no rows, use the public data given.

QUESTION: {q}
{Q_HINT.get(qid, "")}
COUNTS: {json.dumps({k: agg[k] for k in counts})}
COMPARISON: {json.dumps(rows)}
{extra}
CODED INTERVIEWS: {json.dumps([_brief(c, fields) for c in codes])}

Return ONLY JSON: {{"answer": "2-4 sentences", "evidence": ["[P1] \\"exact quote\\""],
"public_feedback": "agrees / differs / mixed + one sentence", "confidence": "low|medium|high"}}"""
        a = _chat(prompt, max_tokens=1800)
        a["answer"] = re.split(r"\n*\s*Evidence\s*:", a.get("answer", ""))[0].strip()
        a["evidence"], a["evidence_dropped"] = _cite(a.get("evidence"), pools)
        a["id"] = qid
        # any "k of n" must be one of the computed counts for this question
        allowed = {str(v) for k in counts for v in (
            [x[1] for x in agg[k]] if isinstance(agg[k], list) else [agg[k]])}
        bad = [m for m in re.findall(rf"\b(\d+) of {agg['n']}\b", a.get("answer", "")) if m not in allowed]
        if bad:
            a["answer"] += (f" ⚠️ *Check: {', '.join(sorted(set(bad)))} of {agg['n']} isn't in the computed "
                            "counts; see the Counts table.*")
        answers.append(a)
    wrap = _chat(f"""From these answers to discovery questions about finding old photos, list up to 3
surprises or contradictions and 3 open questions to test in real interviews. {note}
ANSWERS: {json.dumps([{k: a.get(k) for k in ("id", "answer", "public_feedback")} for a in answers])}
Return ONLY JSON: {{"surprises": ["..."], "open_questions": ["..."]}}""", max_tokens=1200)
    def _items(v):
        out = []
        for x in v or []:
            out += [p.strip(' "') for p in str(x).split('","') if p.strip(' "')]
        return out
    return {"answers": answers, "surprises": _items(wrap.get("surprises")),
            "open_questions": _items(wrap.get("open_questions"))}


# ---------- report ----------

def _fmt(pairs, n):
    return ", ".join(f"{k} {v}/{n}" for k, v in pairs) or "none"


def report(codes, agg, tri, ans, public_names) -> str:
    n = agg["n"]
    sim = agg["simulated"]
    L = ["# Interview analysis: answers to the discovery questions", ""]
    if sim:
        L += [f"> ⚠️ **{sim} of {n} transcripts are AI-simulated (not real participants).** Treat every "
              "number here as a test of the method, not as evidence about users.", ""]
    L += [f"Generated by `analyze_interviews.py` · model {MODEL} · {n} transcripts · "
          f"{agg['quotes_dropped']} model-proposed quotes dropped because they weren't in the transcript · "
          f"compared with: {', '.join(public_names) or 'none'}", "",
          "*How to read this: the Counts and comparison tables are computed in code and are authoritative. "
          "The answer prose and the surprises are LLM-written from them and can misstate a direction or "
          "number; every quote shown was checked against the participant's own words.*", ""]
    for a in ans.get("answers", []):
        q = dict(QUESTIONS).get(a.get("id"), a.get("id"))
        L += [f"## {a.get('id')}. {q}", "", a.get("answer", ""), ""]
        for e in a.get("evidence") or []:
            L.append(f"- {e}")
        if a.get("public_feedback"):
            L += ["", f"**Public feedback:** {a['public_feedback']}"]
        L += ["", f"*Confidence: {a.get('confidence', '?')}*", ""]
    L += ["## Counts (computed, per participant)", "",
          "| | |", "|---|---|",
          f"| Photo types | {_fmt(agg['photo_types'], n)} |",
          f"| Remembered | {_fmt(agg['remembered'], n)} |",
          f"| Forgotten | {_fmt(agg['forgotten'], n)} |",
          f"| First search move | {_fmt(agg['strategies_first'], n)} |",
          f"| Any search move | {_fmt(agg['strategies_any'], n)} |",
          f"| Where it broke | {_fmt(agg['failure_modes'], n)} |",
          f"| Outcome | {_fmt(agg['outcomes'], n)} |",
          f"| Sent a substitute | {agg['substitute_sent']}/{n} |",
          f"| Photo on someone else's phone | {agg['someone_elses_phone']}/{n} |", ""]
    for name, rows in tri.items():
        pn = rows[0]["public_n"] if rows else 0
        L += [f"## Interviews vs {name}", "",
              f"Share of participants (n={n}) vs share of retrieval-related posts (n={pn}); "
              "'agree' = within 20 points." + (" Small public n: read as direction only." if pn < 30 else ""), "",
              f"| Dimension | Item | Interviews % | {name} % | Verdict |", "|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['dimension']} | {r['item']} | {r['interviews_pct']} | {r[f'{name}_pct']} | {r['verdict']} |")
        L.append("")
    L += ["## Per transcript", "",
          "| ID | Looking for | Trigger | First query | Broke at | Outcome | Workaround |",
          "|---|---|---|---|---|---|---|"]
    for c in codes:
        tag = " (sim)" if c.get("simulated") else ""
        L.append(f"| {c['id']}{tag} | {c.get('target_moment') or ''} | {c.get('trigger') or ''} | "
                 f"{c.get('first_query') or ''} | {c.get('failure_mode') or ''} | {c.get('outcome') or ''} | "
                 f"{c.get('workaround') or ''} |")
    if ans.get("surprises"):
        L += ["", "## Surprises and contradictions", ""] + [f"- {s}" for s in ans["surprises"]]
    if ans.get("open_questions"):
        L += ["", "## Open questions for real interviews", ""] + [f"- {s}" for s in ans["open_questions"]]
    return "\n".join(L).replace("|  |", "| |") + "\n"


def run(transcripts: List[Dict], public: Dict[str, Dict], on_progress=None,
        codes: Optional[List[Dict]] = None) -> Dict:
    """transcripts: [{"id", "text", "simulated"}]; public: {label: summary.json dict}.
    Pass codes= (from a saved run) to skip step 1 and redo only the counts and answers."""
    if codes is not None:  # re-check saved codes with the current quote rules
        texts = {t["id"]: t["text"] for t in transcripts}
        codes = [check(c, texts[c["id"]]) if c["id"] in texts else c for c in codes]
    else:
        codes = []
        for i, t in enumerate(transcripts):
            codes.append(code_transcript(t["text"], t["id"], t.get("simulated", False)))
            if on_progress:
                on_progress(i + 1, len(transcripts))
    agg = aggregate(codes)
    tri = {name: triangulate(agg, s, name) for name, s in public.items()}
    ans = answer(codes, agg, tri, public)
    return {"codes": codes, "aggregate": agg, "triangulation": tri, "answers": ans,
            "report": report(codes, agg, tri, ans, list(public))}
