"""Streamlit page: the interview-analysis agent. Upload or paste transcripts and get answers
to the discovery questions, with checked quotes, compared with the public-feedback findings."""
import json
import os
import sys

import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _env import load_env  # noqa: E402

load_env(os.path.join(HERE, ".env"))
if not os.environ.get("GROQ_API_KEY"):
    try:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
from analyze import interviews  # noqa: E402
from analyze_interviews import DEFAULT, load_public  # noqa: E402

SAVED = os.path.join(HERE, "data", "interviews_analysis")

st.title("🗣️ Interview-analysis agent")
st.caption("Codes each transcript with the discovery engine's taxonomy, drops any quote that isn't "
           "in the participant's own words, counts in code (not the LLM), then answers the "
           "discovery questions and compares them with what public feedback says.")

tab_saved, tab_new, tab_how = st.tabs(["Latest analysis", "Analyse new transcripts", "How it works"])

with tab_saved:
    if os.path.exists(SAVED + ".md"):
        st.markdown(open(SAVED + ".md").read())
    else:
        st.info("No saved analysis yet. Run `python analyze_interviews.py` or use the next tab.")

with tab_new:
    st.write("Upload real session notes (.md / .txt), or paste one. Mark interviewer turns with "
             "`Interviewer:` or `Moderator:` so quotes are checked against the participant only.")
    files = st.file_uploader("Transcripts", type=["md", "txt"], accept_multiple_files=True)
    pasted = st.text_area("…or paste a transcript", height=180)
    real = st.checkbox("These are real participants", value=True)
    add_sim = st.checkbox("Also include the 9 AI-simulated transcripts (P1–P6, T1–T3)", value=False)
    if st.button("Analyse", type="primary"):
        ts = [{"id": os.path.splitext(f.name)[0], "text": f.read().decode("utf-8", "ignore"),
               "simulated": not real} for f in files or []]
        if pasted.strip():
            ts.append({"id": f"pasted{len(ts) + 1}", "text": pasted, "simulated": not real})
        if add_sim:
            ts += [{"id": os.path.splitext(os.path.basename(p))[0], "text": open(p).read(),
                    "simulated": True} for p in DEFAULT]
        if not ts:
            st.warning("Add at least one transcript.")
        elif not os.environ.get("GROQ_API_KEY"):
            st.error("GROQ_API_KEY is not set.")
        else:
            bar = st.progress(0.0, "Coding transcripts…")
            try:
                res = interviews.run(ts, load_public(),
                                     on_progress=lambda i, n: bar.progress(i / n, f"Coded {i}/{n}"))
            except Exception as e:
                st.error(f"Analysis failed: {e}")
            else:
                bar.empty()
                st.markdown(res["report"])
                st.download_button("Download report (.md)", res["report"], "interview_analysis.md")
                st.download_button("Download codes (.json)",
                                   json.dumps({k: v for k, v in res.items() if k != "report"}, indent=2,
                                              default=str), "interview_analysis.json")

with tab_how:
    st.markdown("""
1. **Code**: one Groq call per transcript (`openai/gpt-oss-120b`, temperature 0) fills a fixed
   sheet: target photo, trigger, what was remembered or forgotten (engine cue types), first query,
   how the memory became a query, where the funnel broke, workaround, outcome, key quotes.
2. **Check**: every quote is matched against the participant's own lines (interviewer turns and
   persona briefs removed). Quotes that don't match are dropped, and the count is shown.
3. **Count**: shares per participant are computed in Python, so the LLM can't miscount.
4. **Triangulate**: interview shares vs the engine's findings from app reviews and public forums:
   agree, interviews higher, or public higher.
5. **Answer**: one final call writes answers to Q1–Q5. It may cite only the checked quotes and
   the computed counts; any evidence line that doesn't match is removed.

**Limits:** the coding is an LLM's judgement (spot-check the per-transcript table); small n; the
simulated transcripts test the method and say nothing about real users.
""")
