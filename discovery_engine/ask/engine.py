"""
Ask the discovery engine: answer any discovery question from the collected public feedback.

  corpus   every collected review / forum post that shows retrieval intent (regex, free),
           with the Groq labels attached where the item was classified by the pipeline
  plan     the LLM turns the question into 3-5 keyword searches (users' words, not ours)
  search   BM25 over the corpus for each search, fused by reciprocal rank, mixed across sources
  answer   the LLM reads only the top 28 posts, names 2-5 themes and labels EVERY post with its
           theme(s) or "off"; labels outside the retrieved set are dropped, and each theme's
           support is the number of posts labelled with it

Cost per question: 2 Groq calls (~7k tokens), whatever the corpus size.
"""
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from groq import Groq, RateLimitError

from analyze.filters import RETRIEVAL_KW, forum_score
from ask import lexicon

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CORPUS = os.path.join(DATA, "corpus.json")
MODELS = [os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"), "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]

SOURCE_NAMES = {"play_store": "Play Store", "app_store": "App Store", "hackernews": "Hacker News",
                "stackexchange": "Stack Exchange", "gphotos_community": "Google Photos Community",
                "reddit": "Reddit", "youtube": "YouTube"}


# ---------- corpus ----------

def _src(r):
    return (r.get("source") or "?").split("/")[0]


def _date(v) -> str:
    """ISO date; Stack Exchange gives unix seconds."""
    v = str(v or "")
    if v.isdigit() and len(v) >= 9:
        import datetime
        return datetime.datetime.utcfromtimestamp(int(v)).strftime("%Y-%m-%d")
    return v[:10]


def build_corpus() -> List[Dict]:
    """Merge every collected file, keep items with retrieval intent, attach labels. Saves corpus.json."""
    files = ["store_all.json", "raw.json", "parents_raw.json", "public_forum_all.json", "public_raw.json"]
    raw = []
    for f in files:
        p = os.path.join(DATA, f)
        if os.path.exists(p):
            raw += json.load(open(p))
    labels = {}
    for f in ("classified.json", "parents_classified.json", "public_classified.json"):
        p = os.path.join(DATA, f)
        if os.path.exists(p):
            for c in json.load(open(p)):
                if c.get("source_text") and not c.get("error"):
                    labels[c["source_text"].strip()[:2000].lower()] = c
    seen, out = set(), []
    for r in raw:
        text = (r.get("text") or "").strip()
        key = re.sub(r"\W+", " ", text.lower()).strip()
        if len(text) < 40 or key in seen:
            continue
        forum = _src(r) in ("hackernews", "stackexchange", "gphotos_community")
        if (forum and not forum_score(text)) or (not forum and not RETRIEVAL_KW.search(text)):
            continue
        seen.add(key)
        item = {"id": len(out), "source": r.get("source"), "url": r.get("url"), "date": _date(r.get("date")),
                "rating": r.get("rating"), "country": r.get("country"), "upvotes": r.get("upvotes") or 0,
                "text": text[:3000]}
        lab = labels.get(text[:2000].lower())
        if lab:
            item["label"] = {k: lab.get(k) for k in ("is_retrieval_related", "failure_mode", "photo_type",
                                                     "remembered_cues", "forgotten_cues", "query_strategies")}
        out.append(item)
    with open(CORPUS, "w") as f:
        json.dump(out, f)
    return out


def load_corpus() -> List[Dict]:
    return json.load(open(CORPUS)) if os.path.exists(CORPUS) else build_corpus()


# How much a post is about *searching for* a photo (vs. lost photos, backup, storage).
SEARCH_ACT = re.compile(r"\b(search(ed|ing)?|typed?|look(ed|ing)? for|find(ing)?|scroll(ed|ing)?|locate|"
                        r"remember(ed)?|can'?t recall|forgot)\b", re.I)
LOSS = re.compile(r"\b(delet(e|ed|ing)|backup|backed up|back up|sync(ed|ing)?|storage|lost|recover|restore|"
                  r"subscription|price|pay|space|disappear(ed)?)\b", re.I)


def intent(d: Dict) -> float:
    t = d["text"]
    s = min(3, len(SEARCH_ACT.findall(t))) * 0.5
    if (d.get("label") or {}).get("is_retrieval_related"):
        s += 1.5
    if LOSS.search(t) and len(SEARCH_ACT.findall(t)) < 2:
        return 0.35  # about lost photos / backup, not about finding one
    return 1 + s


# ---------- BM25 (no dependency) ----------

STOP = set("""a an the and or but if of to in on at for with by from is are was were be been being it its this that
these those i me my we our you your he she they them their his her as so not no do does did have has had can could
would should will just very really also than then there here what which who whom when where why how all any some
more most much many into out up down over about after before again only own same too s t don now get got""".split())


def _tok(s: str) -> List[str]:
    out = []
    for w in re.findall(r"[a-z0-9']+", s.lower().replace("’", "'")):
        w = w.strip("'")
        if w.endswith("'s"):
            w = w[:-2]
        if len(w) < 2 or w in STOP:
            continue
        for suf in ("ing", "ed", "es", "s"):
            if len(w) > len(suf) + 3 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        out.append(w)
    return out


class BM25:
    def __init__(self, docs: List[str], k1=1.4, b=0.75):
        self.toks = [_tok(d) for d in docs]
        self.avg = sum(map(len, self.toks)) / max(1, len(self.toks))
        self.tf = [Counter(t) for t in self.toks]
        df = Counter(w for t in self.toks for w in set(t))
        n = len(self.toks)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str) -> Dict[int, float]:
        q = set(_tok(query))
        out = {}
        for i, tf in enumerate(self.tf):
            s = 0.0
            L = len(self.toks[i])
            for w in q:
                f = tf.get(w)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * L / self.avg))
            if s:
                out[i] = s
        return out


# ---------- LLM ----------

_client = None
used_model = None


def _chat(prompt: str, max_tokens: int) -> Dict:
    """JSON-mode call with a model fallback chain on the free tier's daily caps."""
    global _client, used_model
    _client = _client or Groq(timeout=90, max_retries=1)
    last = None
    for model in dict.fromkeys(MODELS):
        for attempt in range(4):
            kw = {"reasoning_effort": "low"} if "gpt-oss" in model else {"reasoning_format": "hidden"}
            try:
                r = _client.chat.completions.create(
                    model=model, temperature=0.2, max_tokens=max_tokens, response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": prompt}], **kw)
                txt = r.choices[0].message.content or ""
                used_model = model
                txt = txt[txt.find("{"): txt.rfind("}") + 1]
                return json.loads(txt)
            except RateLimitError as e:
                last = e
                if "per day" in str(e) or "TPD" in str(e):
                    break  # next model
                time.sleep(12 * (attempt + 1))
            except (json.JSONDecodeError, ValueError) as e:
                last = e  # retry once more, then next model
    raise RuntimeError(f"All Groq models failed or are over their free daily limit ({last})")


PLAN = """You help search a corpus of Google Photos user reviews and forum posts about finding photos.
Turn the research question into 6 short keyword searches, written in the plain words a frustrated
USER would put in a review, and specific to THIS question. Mix two kinds:
- 3 about the experience the question asks about (what they tried, what failed, what they recall)
- 3 naming concrete kinds of photos or details that users in this situation would mention
Keep each search 2-6 words. Don't use research jargon ("formulate", "cues", "retrieval").
The corpus is about FINDING a photo the user knows they have: don't write searches about lost,
deleted or missing photos unless the question asks about those.
Also pick 0-2 FACETS that posts answering this question would show, from: forgot (says they can't
remember a detail), remembered (says what they do remember), search_failed, workaround,
faces_people, time_dates.
QUESTION: {q}
Return ONLY JSON: {{"searches": ["...", "...", "...", "...", "...", "..."], "facets": ["..."]}}"""

ANSWER = """You are a user researcher. Answer the research question using ONLY the numbered user posts
below (real Google Photos users: app-store reviews and forum posts). Rules:
- Group the evidence into 2-5 themes that answer the question, most supported first.
- Then label EVERY post 1..{k}: the ONE theme letter it best supports plus the exact words (3-12,
  copied verbatim from that post) that show it, or "off" if it doesn't address the question.
- Use users' own words; don't invent facts. If the posts don't answer part of the question, say so.
- Ignore posts that don't address the question (e.g. lost/deleted photos, backup, storage, pricing when
  the question is about finding a photo you remember); list their numbers under "off_topic".
- These {k} posts are the best matches out of {n} relevant posts, not a random sample; don't
  present counts as population percentages.

- The COUNTS are keyword mentions across many posts (approximate). Use them for scale where they
  help, e.g. "mentioned in 350 of 2,077 posts that describe searching"; never invent other numbers.

QUESTION: {q}

COUNTS (all {s_n} posts that describe searching): {counts_all}
COUNTS (the {m_n} posts matching this question): {counts_q}

POSTS:
{posts}

Return ONLY JSON:
{{"summary": "2-3 sentence direct answer",
  "themes": [{{"id": "A", "theme": "short name", "explanation": "1-2 sentences"}}],
  "labels": {{"1": {{"theme": "A", "words": "exact words from post 1"}}, "2": "off"}},
  "not_covered": "what the evidence can't answer, one sentence",
  "follow_up_questions": ["2-3 questions to ask next"]}}"""


# ---------- ask ----------

class Engine:
    def __init__(self, corpus: Optional[List[Dict]] = None):
        self.docs = corpus if corpus is not None else load_corpus()
        self.index = BM25([d["text"] for d in self.docs])
        self.prior = [intent(d) for d in self.docs]
        self.counts_all = lexicon.counts(lexicon.searching(self.docs))

    def stats(self) -> Dict:
        return {"docs": len(self.docs), "by_source": Counter(_src(d) for d in self.docs).most_common(),
                "labelled": sum(1 for d in self.docs if d.get("label"))}

    def search(self, queries: List[str], sources: Optional[List[str]] = None, k: int = 20,
               facets: Optional[List[str]] = None):
        """Reciprocal-rank fusion over several BM25 searches (+ one list per facet); at most 60% of
        hits from one source. Posts matching a chosen facet get a 2x boost."""
        fused, matched = defaultdict(float), set()
        boost = [1.0] * len(self.docs)
        for f in facets or []:
            rx = lexicon.FACETS[f]
            fac = [i for i, d in enumerate(self.docs) if rx.search(d["text"])
                   and (not sources or _src(d) in sources)]
            for i in fac:
                boost[i] = 2.0
            if f in ("forgot", "remembered", "search_failed", "workaround"):  # specific enough to list alone
                fac.sort(key=lambda i: -self.prior[i])
                matched.update(fac)
                for rank, i in enumerate(fac[:80]):
                    fused[i] += 2 / (60 + rank)  # the facet is what the question is about
        for q in queries:
            sc = {i: v * boost[i] for i, v in self.index.scores(q).items()}
            ranked = sorted((i for i in sc if not sources or _src(self.docs[i]) in sources),
                            key=lambda i: -sc[i] * self.prior[i])
            if ranked:  # "matching" = strong matches only, not every post that says "photo"
                top = sc[ranked[0]] * self.prior[ranked[0]]
                matched.update(i for i in ranked if sc[i] * self.prior[i] >= 0.4 * top)
            for rank, i in enumerate(ranked[:80]):
                fused[i] += 1 / (60 + rank)
        order = sorted(fused, key=lambda i: -fused[i])
        cap, per, picked = max(4, int(k * 0.6)), Counter(), []
        for i in order:
            s = _src(self.docs[i])
            if per[s] < cap:
                picked.append(i)
                per[s] += 1
            if len(picked) == k:
                break
        return [self.docs[i] for i in picked], len(matched), matched

    def ask(self, question: str, sources: Optional[List[str]] = None, k: int = 28) -> Dict:
        t0 = time.time()
        p = _chat(PLAN.format(q=question), 700)
        searches = [s for s in p.get("searches") or [] if isinstance(s, str) and s.strip()][:6] or [question]
        facets = [f for f in p.get("facets") or [] if f in lexicon.FACETS][:2]
        ql = question.lower()  # the planner sometimes skips the obvious facet
        for word, f in (("forg", "forgot"), ("remember", "remembered"), ("workaround", "workaround"),
                        ("when search fails", "search_failed")):
            if word in ql and f not in facets:
                facets = [f] + facets[:1]
        hits, matched, matched_ids = self.search(searches + [question], sources, k, facets)
        if not hits:
            return {"question": question, "searches": searches, "hits": [], "matched": 0,
                    "summary": "No posts in the corpus match this question.", "themes": []}
        posts = "\n".join(f"[{n}] ({SOURCE_NAMES.get(_src(h), _src(h))}{', ' + str(h['rating']) + '★' if h.get('rating') else ''}"
                          f"{', ' + h['date'] if h.get('date') else ''}) {h['text'][:380]}"
                          for n, h in enumerate(hits, 1))
        counts_q = lexicon.counts([self.docs[i] for i in matched_ids])
        brief = lambda c: json.dumps({k: c[k][:6] if isinstance(c[k], list) else c[k] for k in c})
        out = _chat(ANSWER.format(q=question, posts=posts, k=len(hits), n=matched, s_n=self.counts_all["n"],
                                  m_n=counts_q["n"], counts_all=brief(self.counts_all),
                                  counts_q=brief(counts_q)), 2500)
        # support comes from the per-post labels (the model must label every post it read)
        labels = out.get("labels") if isinstance(out.get("labels"), dict) else {}
        by_theme, off = defaultdict(set), set()
        norm = lambda x: re.sub(r"[^a-z0-9]+", " ", str(x).lower().replace("’", "'")).strip()
        evidence, dropped = {}, 0
        for p, v in labels.items():
            if not str(p).isdigit() or not 1 <= int(p) <= len(hits):
                continue
            p = int(p)
            v = v[0] if isinstance(v, list) and v else v
            theme, words = (v.get("theme"), v.get("words", "")) if isinstance(v, dict) else (v, "")
            theme = str(theme or "").strip().upper()
            if theme in ("OFF", ""):
                off.add(p)
                continue
            by_theme[theme].add(p)
            if words and len(norm(words)) >= 8 and norm(words) in norm(hits[p - 1]["text"]):
                evidence[p] = words  # verbatim: shown as the highlighted phrase
            elif words:
                dropped += 1  # paraphrased evidence: label kept, phrase not shown
        themes = []
        for t in out.get("themes") or []:
            ids = set(by_theme.get(str(t.get("id", "")).strip().upper(), set()))
            ids = sorted(ids)
            if ids:
                themes.append({"theme": t.get("theme", ""), "explanation": t.get("explanation", ""),
                               "posts": ids, "support": len(ids),
                               "sources": Counter(SOURCE_NAMES.get(_src(hits[i - 1]), "?") for i in ids).most_common()})
        themes.sort(key=lambda t: -t["support"])
        return {"question": question, "searches": searches, "facets": facets, "matched": matched, "hits": hits,
                "counts_question": counts_q,
                "summary": out.get("summary", ""), "themes": themes, "not_covered": out.get("not_covered"),
                "off_topic": sorted(off - {i for t in themes for i in t["posts"]}),
                "unlabelled": len(hits) - len({int(p) for p in labels if str(p).isdigit()}),
                "evidence_paraphrased": dropped, "evidence": evidence,
                "follow_up_questions": out.get("follow_up_questions") or [], "model": used_model,
                "seconds": round(time.time() - t0, 1)}
