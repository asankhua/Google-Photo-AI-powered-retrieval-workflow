"""
Discovery Engine — public, testable UI (Streamlit).

Tabs:
  0) Ask the engine — type any discovery question; answered from the whole corpus with cited quotes.
  1) Findings  — ranked opportunity analysis from the last pipeline run.
  2) Try it    — paste any user feedback and classify it live with Groq.
  3) How it works — the 1-slide explanation.

Run:     streamlit run app.py
Deploy:  Streamlit Community Cloud (point at this repo).
"""
import json
from collections import Counter
import os

import streamlit as st

# set_page_config MUST be the first Streamlit command in the script.
st.set_page_config(page_title="Photo Retrieval Discovery Engine", layout="wide")

from _env import load_env
load_env()  # load GROQ_API_KEY from local .env (Streamlit Cloud uses st.secrets)

# On Streamlit Cloud, secrets arrive via st.secrets — mirror them into the env.
# Accessing st.secrets with no secrets.toml raises; guard it so local runs are clean.
if not os.environ.get("GROQ_API_KEY"):
    try:
        if "GROQ_API_KEY" in st.secrets:
            os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

from ask import lexicon
from ask.engine import SOURCE_NAMES, Engine, _src
from classify.classifier import classify_one
from classify.taxonomy import FAILURE_MODES

DATA = os.path.join(os.path.dirname(__file__), "data")

st.title("Photo Retrieval — AI Discovery Engine")
st.caption(
    "Analyzes real user feedback to locate WHERE photo retrieval breaks: "
    "Expression -> Understanding -> Evaluation -> Refinement."
)

tab_ask, tab_findings, tab_try, tab_how = st.tabs(["🔎 Ask the engine", "Findings", "Try it", "How it works"])


def _load(name):
    path = os.path.join(DATA, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_resource(show_spinner="Loading the corpus…")
def _engine():
    return Engine()


@st.cache_data(show_spinner=False, max_entries=200)
def _ask(question, sources):
    return _engine().ask(question, list(sources) or None)


EXAMPLES = [
    "What kinds of old photos do users struggle to retrieve?",
    "What information do people actually remember about a photo?",
    "What information have they forgotten?",
    "How do users formulate searches when their memory is incomplete?",
    "What do users do when search fails to find the photo?",
    "Why does face or people search fail for users?",
    "What do parents say about finding photos of their kids?",
    "How do people feel about the new Gemini / Ask Photos search?",
]


def _quote(text, words=None, n=320):
    """Excerpt around the evidence phrase, with the phrase in bold."""
    import re
    text = " ".join(text.split()).replace("*", "")
    m = re.search(re.escape(" ".join(words.split())), text, re.I) if words else None
    if not m:
        return text[:n] + ("…" if len(text) > n else "")
    a = max(0, m.start() - n // 2)
    b = min(len(text), a + n)
    return (("…" if a else "") + text[a:m.start()] + f"**{m.group(0)}**" + text[m.end():b]
            + ("…" if b < len(text) else ""))


def _bars(pairs, label):
    import pandas as pd
    if pairs:
        df = pd.DataFrame(pairs, columns=[label, "posts"]).set_index(label)
        st.bar_chart(df, horizontal=True, height=40 + 28 * len(pairs), color="#0072B2")


with tab_ask:
    eng = _engine()
    st_ = eng.stats()
    st.markdown("Ask any discovery question. The engine searches **every collected review and forum post**, "
                "reads the best matches, and answers in themes, citing the users' own words.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Posts with retrieval intent", f"{st_['docs']:,}")
    m2.metric("…that describe searching", f"{eng.counts_all['n']:,}")
    m3.metric("Sources", len(st_["by_source"]))
    st.caption("From 12,191 unique store reviews (Play Store + App Store, IN/US/GB/AU/CA) and 1,744 forum posts: "
               + " · ".join(f"{SOURCE_NAMES.get(k, k)} {v:,}" for k, v in st_["by_source"]))

    if "ask_q" not in st.session_state:
        st.session_state.ask_q = EXAMPLES[0]
    st.write("**Try a question:**")
    cols = st.columns(4)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 4].button(ex, key=f"ex{i}", width="stretch"):
            st.session_state.ask_q = ex
    q = st.text_input("Your question", key="ask_q")
    srcs = st.multiselect("Only these sources (optional)", [k for k, _ in st_["by_source"]],
                          format_func=lambda k: SOURCE_NAMES.get(k, k))
    if q.strip():
        if not os.environ.get("GROQ_API_KEY"):
            st.error("GROQ_API_KEY is not set, so the engine can't answer.")
        else:
            try:
                with st.spinner("Searching the corpus and reading the best matches…"):
                    r = _ask(q.strip(), tuple(sorted(srcs)))
            except Exception as e:
                st.error(f"Couldn't answer: {e}")
                r = None
            if r:
                st.subheader("Answer")
                st.write(r["summary"])
                off = set(r.get("off_topic") or [])
                st.caption(f"Searched for: {', '.join(r['searches'])}"
                           + (f" · facets: {', '.join(r['facets'])}" if r.get("facets") else "")
                           + f" · {r['matched']:,} strong matches · read the top {len(r['hits'])}"
                           + (f" ({len(off)} off-topic)" if off else "")
                           + f" · {r.get('model')} · {r.get('seconds')}s")
                for t in r["themes"]:
                    with st.container(border=True):
                        st.markdown(f"**{t['theme']}** — {t['support']} of {len(r['hits'])} posts read · "
                                    + ", ".join(f"{s} {n}" for s, n in t["sources"]))
                        st.write(t["explanation"])
                        ev = r.get("evidence") or {}
                        for i in sorted(t["posts"], key=lambda i: i not in ev)[:3]:
                            h = r["hits"][i - 1]
                            meta = " · ".join(x for x in [SOURCE_NAMES.get(_src(h), _src(h)),
                                                          f"{h['rating']}★" if h.get("rating") else "",
                                                          h.get("date") or ""] if x)
                            link = f" · [link]({h['url']})" if h.get("url") and _src(h) not in ("play_store",) else ""
                            st.markdown(f"> {_quote(h['text'], ev.get(i))}  \n*[{i}] {meta}{link}*")
                        rest = [i for i in t["posts"] if i not in sorted(t["posts"], key=lambda i: i not in ev)[:3]]
                        if rest:
                            st.caption("Also: " + ", ".join(f"[{i}]" for i in rest) + " (see the table below)")
                if r.get("not_covered"):
                    st.info("**What this evidence can't answer:** " + r["not_covered"])
                if r.get("follow_up_questions"):
                    st.write("**Ask next:**")
                    fcols = st.columns(len(r["follow_up_questions"][:3]))
                    for j, fq in enumerate(r["follow_up_questions"][:3]):
                        if fcols[j].button(fq, key=f"fu{j}", width="stretch"):
                            st.session_state.ask_q = fq
                            st.rerun()
                with st.expander(f"All {len(r['hits'])} posts the engine read"):
                    cited = {i for t in r["themes"] for i in t["posts"]}
                    themes_of = {i: t["theme"] for t in r["themes"] for i in t["posts"]}
                    st.dataframe([{"#": n, "theme": themes_of.get(n, "off-topic" if n in off else ""),
                                   "evidence": (r.get("evidence") or {}).get(n, ""),
                                   "source": SOURCE_NAMES.get(_src(h), _src(h)), "rating": h.get("rating"),
                                   "date": h.get("date"), "text": h["text"][:300], "url": h.get("url")}
                                  for n, h in enumerate(r["hits"], 1)],
                                 hide_index=True, width="stretch",
                                 column_config={"url": st.column_config.LinkColumn("link")})

    st.divider()
    st.subheader("At scale: what the posts that describe searching mention")
    st.caption(f"Keyword counts across all {eng.counts_all['n']:,} posts that describe searching (approximate: "
               "a mention, not a judgement). The quotes above carry the meaning; these give the scale.")
    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown("**Q1 · Kinds of photos named**")
        _bars(eng.counts_all["photo_kinds"], "kind")
    with a2:
        st.markdown("**Q2/Q4 · Details they mention or search with**")
        _bars(eng.counts_all["cues_mentioned"], "detail")
    with a3:
        st.markdown(f"**Q3 · What they say they forgot** ({eng.counts_all['say_they_forgot']} posts)")
        _bars(eng.counts_all["forgotten_cues"], "detail")
        fx = [d for d in eng.docs if lexicon.FORGET.search(d["text"])][:4]
        for d in fx:
            m = lexicon.FORGET.search(d["text"])
            st.caption(f"“…{d['text'][max(0, m.start() - 80):m.end() + 60].strip()}…” — {SOURCE_NAMES.get(_src(d))}")


with tab_findings:
    summary = _load("summary.json")
    if not summary:
        st.info("No pipeline run yet. Run `python run_pipeline.py` to populate findings.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Feedback items analyzed", summary.get("total_items", 0))
        c2.metric("Retrieval-related", summary.get("retrieval_related", 0))
        c3.metric("Retrieval share", f"{summary.get('retrieval_share_pct', 0)}%")

        st.subheader("Opportunity ranking (frequency x severity)")
        for o in summary.get("opportunities_ranked", []):
            with st.expander(
                f"{o['failure_mode']} — score {o['opportunity_score']} · "
                f"{o['count']} items · {o['share_pct']}% · sev {o['avg_severity']}"
            ):
                st.write(FAILURE_MODES.get(o["failure_mode"], ""))
                st.write("Top photo types:", o["top_photo_types"])
                st.write("Evidence quotes:")
                for ex in o["examples"]:
                    st.markdown(f"> {ex['quote']}  \n*— {ex['source']}*")

        st.divider()
        st.subheader("The four discovery questions")
        q1, q2 = st.columns(2)
        q1.markdown("**Q1 — Kinds of photos users struggle to retrieve**")
        q1.table(summary.get("photo_types_top", []))
        q2.markdown("**Q4 — How users formulate searches (incomplete memory)**")
        q2.table(summary.get("query_strategies_top", []))
        if summary.get("query_examples"):
            q2.caption("Example queries: " + "; ".join(summary["query_examples"][:8]))

        cA, cB = st.columns(2)
        cA.markdown("**Q2 — What users REMEMBER**")
        cA.table(summary.get("remembered_cues_top", []))
        cB.markdown("**Q3 — What users FORGET / lack**")
        cB.table(summary.get("forgotten_cues_top", []))

    parents = _load("parents_summary.json")
    if parents:
        st.divider()
        st.subheader("Segment check: parents")
        st.caption(
            "Play Store reviews (India, US, UK) that mention the reviewer's child, "
            "classified the same way. Run: `python run_pipeline.py --play 8000 "
            "--countries in,us,gb --sorts newest,relevant --appstore 0 --segment parents`"
        )
        p1, p2, p3 = st.columns(3)
        p1.metric("Reviews collected", parents.get("collected_items", 0))
        p2.metric("Mention a child", parents.get("segment_items", 0))
        p3.metric("Retrieval-related", parents.get("retrieval_related", 0))
        for o in parents.get("opportunities_ranked", []):
            with st.expander(
                f"{o['failure_mode']} — score {o['opportunity_score']} · "
                f"{o['count']} items · {o['share_pct']}% · sev {o['avg_severity']}"
            ):
                for ex in o["examples"]:
                    st.markdown(f"> {ex['quote']}  \n*— {ex['source']}*")
        pa, pb = st.columns(2)
        pa.markdown("**What parents REMEMBER**")
        pa.table(parents.get("remembered_cues_top", []))
        pb.markdown("**What parents FORGET / lack**")
        pb.table(parents.get("forgotten_cues_top", []))

    public = _load("public_summary.json")
    pub_items = _load("public_classified.json") or []
    if public:
        st.divider()
        st.subheader("Public discussions: forums and community")
        st.caption(
            "Hacker News (Algolia API), Stack Exchange (webapps, android, photo, superuser, apple) and the "
            "Google Photos Help Community listing: posts about trying to find a photo, the most upvoted "
            "per source. Reddit needs API credentials and YouTube an API key (see README); social "
            "media needs login, so it isn't collected. Run: `python run_pipeline.py --play 0 "
            "--appstore 0 --hn --stackexchange --community --youtube --per-source 30 --prefix public --filter`"
        )
        for n in public.get("notes", []):
            st.caption("• " + n)
        if public.get("forum_on_topic"):
            st.caption("On-topic (find-a-photo) posts by source: " + ", ".join(
                f"{k} {v} of {public.get('forum_collected', {}).get(k, '?')}" for k, v in public["forum_on_topic"].items()))
        u1, u2, u3 = st.columns(3)
        u1.metric("Posts collected", public.get("collected_items", 0))
        u2.metric("Classified", public.get("total_items", 0))
        u3.metric("Retrieval-related", public.get("retrieval_related", 0))
        rel = [c for c in pub_items if c.get("is_retrieval_related")]
        by_src = {}
        for c in rel:
            by_src.setdefault((c.get("source") or "?").split("/")[0], []).append(c)
        if by_src:
            st.markdown("**By source** (retrieval-related posts)")
            rows = []
            for src, cs in by_src.items():
                top = lambda key: ", ".join(k for k, _ in Counter(x for c in cs for x in c.get(key) or []).most_common(3))
                rows.append({"source": src, "posts": len(cs),
                             "top photo types": ", ".join(k for k, _ in Counter(c.get("photo_type") for c in cs).most_common(3)),
                             "remembered": top("remembered_cues"), "forgotten": top("forgotten_cues"),
                             "search moves": top("query_strategies")})
            st.dataframe(rows, hide_index=True, width="stretch")
        for o in public.get("opportunities_ranked", []):
            with st.expander(
                f"{o['failure_mode']} — score {o['opportunity_score']} · "
                f"{o['count']} items · {o['share_pct']}% · sev {o['avg_severity']}"
            ):
                for ex in o["examples"]:
                    st.markdown(f"> {ex['quote']}  \n*— {ex['source']}*")
        v1, v2 = st.columns(2)
        v1.markdown("**Q1 — Photo types**")
        v1.table(public.get("photo_types_top", []))
        v2.markdown("**Q4 — Search moves**")
        v2.table(public.get("query_strategies_top", []))
        if public.get("query_examples"):
            v2.caption("Example queries: " + "; ".join(public["query_examples"][:8]))
        w1, w2 = st.columns(2)
        w1.markdown("**Q2 — What they REMEMBER**")
        w1.table(public.get("remembered_cues_top", []))
        w2.markdown("**Q3 — What they FORGET / lack**")
        w2.table(public.get("forgotten_cues_top", []))


with tab_try:
    st.write("Paste a review, Reddit post, or describe a retrieval struggle:")
    example = ("I know I have a photo of a small cafe from our Goa trip a few years "
               "ago but I can't remember the name and searching 'cafe' shows nothing "
               "useful. I gave up scrolling.")
    text = st.text_area("Feedback", value=example, height=140)
    if st.button("Classify", type="primary"):
        if not os.environ.get("GROQ_API_KEY"):
            st.error("GROQ_API_KEY is not set in the environment.")
        else:
            with st.spinner("Classifying with Groq..."):
                try:
                    st.json(classify_one(text))
                except Exception as e:
                    st.error(f"Error: {e}")


with tab_how:
    st.markdown(
        """
**Pipeline:** Collect -> Classify (Groq) -> Aggregate/Rank.

1. **Collect** real feedback: Play Store, App Store, Reddit, Photos Help community.
2. **Classify** each item against the retrieval funnel — tagging the failure stage,
   photo type, cues *remembered* vs *forgotten*, and severity (1-5).
3. **Aggregate** into an opportunity ranking = frequency x severity, so we can
   *compare* distinct retrieval problems with evidence (not just sentiment).

**Insight it surfaces:** the *memory-index schema mismatch* — users remember
episodic/associative cues; Photos indexes literal ones.
"""
    )
