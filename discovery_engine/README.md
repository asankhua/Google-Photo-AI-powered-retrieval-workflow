# Photo Retrieval — AI Discovery Engine (Part 1)

Analyzes real user feedback at scale to locate **where photo retrieval breaks**, not
just whether users are happy. Built with Claude Code; classification runs on **Groq**.

## Why it's more than sentiment analysis
Every item is classified against the **retrieval funnel** — a retrieval succeeds only
if the user passes all four stages in order:

`Expression → Understanding → Evaluation → Refinement`

For each item we tag: the **failure stage**, the **photo type**, cues the user
**remembered vs forgot**, the **search strategy + actual query text**, and a
**severity** (1–5). We then rank opportunities by `frequency × severity`. This surfaces
the **memory–index schema mismatch**: users recall episodic/associative cues; Photos
indexes literal ones.

### Answers the four discovery questions
| Question | Field(s) that answer it | Summary key |
|---|---|---|
| What kinds of old photos do users struggle to retrieve? | `photo_type` | `photo_types_top` |
| What info do people actually remember? | `remembered_cues` | `remembered_cues_top` |
| What info have they forgotten? | `forgotten_cues` | `forgotten_cues_top` |
| How do users formulate searches with incomplete memory? | `query_strategies`, `query_text` | `query_strategies_top`, `query_examples` |

> Data caveat: generic app-store reviews yield sparse retrieval narratives (~3%), so the
> engine also reads public forums (below).

### Ask the engine (first tab): any discovery question, answered from the whole corpus
Type a question ("What have users forgotten about the photo?", "What do parents say about finding
photos of their kids?") and get themes backed by users' own words, with source, rating, date and link.

| Step | What happens | Guardrail |
|---|---|---|
| Corpus | `data/corpus.json`: **4,278 posts with retrieval intent** from 12,191 unique store reviews (Play + App Store, IN/US/GB/AU/CA) and 1,744 forum posts (HN, Stack Exchange, Community) | Regex filter; forum posts must name a photo app; Groq labels attached where classified |
| Plan | 1 LLM call turns the question into 6 searches in users' words + 0–2 facets (forgot, remembered, search failed, workaround, faces, dates) | No research jargon; obvious facets forced from the question |
| Search | BM25 (pure Python) per search, fused by reciprocal rank; an intent prior demotes lost/backup/pricing posts; ≤60% of hits from one source | Only strong matches (≥0.4 × top score) count as "matching" |
| Answer | 1 LLM call reads the top 28 posts, names 2–5 themes and labels **every** post with one theme (plus exact words from it) or "off" | Support = posts labelled with the theme; evidence phrases shown only if verbatim; off-topic posts listed |
| Scale | Keyword counts over all ~2,000 posts that describe searching: photo kinds, details mentioned, "can't remember when/where…" | Approximate mentions, shown next to the quotes |

```bash
python ask_cli.py --build                                  # rebuild data/corpus.json (needs data/store_all.json)
python ask_cli.py "What information have they forgotten?"  # answer in the terminal
```
Cost: 2 Groq calls per question (~7k tokens), cached per question in the app. Falls back
`gpt-oss-120b → gpt-oss-20b → qwen3.8-27b` when a free daily cap is hit.

### Public discussions: forums and community
```bash
python run_pipeline.py --play 0 --appstore 0 --hn --stackexchange --community --youtube \
    --per-source 18 --prefix public --filter          # add --cached to reuse the last collection
```
| Source | Collector | Needs |
|---|---|---|
| Hacker News | `collect/hackernews.py` (Algolia API) | nothing |
| Stack Exchange | `collect/stackexchange.py` (API v2.3) | nothing |
| Google Photos Help Community | `collect/gphotos_community.py` (thread listing + thread pages, 1 req/s; search is robots-disallowed) | nothing |
| YouTube comments | `collect/youtube.py` (Data API v3) | `YOUTUBE_API_KEY` |
| Reddit | `collect/reddit.py` (PRAW) | `REDDIT_CLIENT_ID/SECRET` |

Forum posts are kept only if a find/search verb sits next to "photo" and a photo app is
named, then ranked by relevance (not upvotes) and capped per source to fit the Groq budget.
Latest run: **1,744 posts → 366 about finding a photo → 36 classified → 12 retrieval-related**
(`data/public_*.json`).

### Segment check: parents
```bash
python run_pipeline.py --play 8000 --countries in,us,gb --sorts newest,relevant --appstore 0 --segment parents
```
Collects Play Store reviews from India, US and UK, keeps those that mention the reviewer's
child (`analyze/segment.py`), and classifies them. Writes `data/parents_*.json`; the base
run is untouched. Latest run: **14,539 reviews → 166 mention a child → 7 about retrieval**;
face grouping is the largest theme (28 of 166). Shown on the dashboard under
"Segment check: parents".

## Architecture
```
Collect                Classify (Groq)             Analyze
------------           -------------------         ------------------------
play_store.py      ─┐                               aggregate.py
app_store.py       ─┤                               opportunity ranking
hackernews.py      ─┤                               cue remembered/forgot
stackexchange.py   ─┼─►  classifier.py  ──► JSON ─► app.py (Streamlit UI)
gphotos_community  ─┤    (taxonomy.py)
youtube.py         ─┤
reddit.py          ─┘
corpus.json ─► ask/engine.py (plan → BM25 search → answer with cited posts) ─► app.py "Ask the engine"
```

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env        # add GROQ_API_KEY (+ optional Reddit creds)
export GROQ_API_KEY=...
```

## Run the pipeline
```bash
python run_pipeline.py --play 300 --appstore 8      # add --reddit if creds set
# outputs: data/raw.json, data/classified.json, data/summary.json
```

## Launch the testable UI
```bash
streamlit run app.py
```
- **Findings**: ranked opportunities + cue analysis.
- **Try it**: paste any feedback, classify live (public test path).
- **How it works**: the 1-slide explanation.

Deploy free on **Streamlit Community Cloud** for the public-link deliverable.

## Validation guardrail (accuracy estimate)
Run `python validate.py` to score the classifier against a hand-labeled gold set
(12 items covering every failure stage + noise). Latest live run on `gpt-oss-120b`:
- **`is_retrieval_related`: 12/12 = 100%** (correctly filters non-retrieval noise)
- **`failure_mode`: 7/8 = 88%** (the one miss is a genuine multi-stage case —
  UNDERSTANDING vs REFINEMENT — not a hallucinated category)

No invented labels were observed, which is the guardrail's core check.

## Status (real runs completed)
- Default model: **`openai/gpt-oss-120b`** on Groq (set `GROQ_MODEL` to override).
- `data/` holds a **real** live run: **~870 reviews** collected (Play + App Store), of
  which 120 matched retrieval keywords and were classified via Groq. Genuine finding:
  **5 were truly retrieval-related**, clustering in the **early funnel** —
  UNDERSTANDING (3) and EXPRESSION (2). This corroborates an earlier run
  (8 of 209, Expression-heavy): retrieval pain is a **small, early-funnel slice** of
  app-store feedback.
- Note on quota: Groq free tier caps **200k tokens/day per model**. A full-corpus pass
  exhausted `gpt-oss-120b`; this run was completed on **`openai/gpt-oss-20b`**
  (`GROQ_MODEL=openai/gpt-oss-20b`), which has its own daily budget. Bulk runs also hit
  a per-minute throttle, so `--filter` is used to keep call volume low.
- Insight: generic app reviews **under-report** retrieval pain. For a richer four-mode
  comparison, run the targeted collectors:
  ```bash
  python run_pipeline.py --reddit --filter    # needs Reddit API creds
  ```
- App Store collection needs **no key** (public RSS); Reddit needs a free "script" app.
