"""
Moment Finder: Streamlit prototype (Part 5 MVP).

A parent searches for their child, gets flooded, and narrows by what they actually
remember: how old he was, which home, who was there. Results come back as a few
explained moments instead of thousands of frames.

Run:  streamlit run app.py
"""
import os
import time

import streamlit as st

import agent
import library as L
from _env import load_env

load_env()
st.set_page_config(page_title="Moment Finder", page_icon="📷", layout="wide")

# On Streamlit Cloud the key arrives via st.secrets; without it the rules parser is used.
if not os.environ.get("GROQ_API_KEY"):
    try:
        if "GROQ_API_KEY" in st.secrets:
            os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

BLUE, ORANGE, GREY = "#0072B2", "#E69F00", "#6B6B6B"  # colourblind-safe
EMOJI = [("cake", "🎂"), ("hair", "✂️"), ("rain", "☔"), ("umbrella", "☔"), ("walking", "👣"),
         ("standing", "🧍"), ("beach", "🏖️"), ("diya", "🪔"), ("sparklers", "🎇"), ("rangoli", "🪔"),
         ("uniform", "🎒"), ("stage", "🎭"), ("park", "🌳"), ("bath", "🛁"), ("hospital", "🏥"),
         ("boxes", "📦"), ("food", "🍌"), ("toy", "🧸"), ("sleeping", "😴")]


@st.cache_data
def get_lib():
    return L.load()


lib = get_lib()
CHILD = lib["child"]["name"]
HOMES = ["any"] + [r["name"] for r in lib["residences"]]
SHORT = {r["name"]: r["short"] for r in lib["residences"]}
# Chip options are fixed labels (live counts go in a caption), so widgets keep their state.
STAGE_NAME = {"any": "Any age"}
for _k, (_n, _lo, _hi) in L.STAGES.items():
    STAGE_NAME[_k] = f"{_n} ({_lo}–{_hi} mo)" if _hi < 99 else f"{_n} ({_lo}+ mo)"
NAME_STAGE = {v: k for k, v in STAGE_NAME.items()}
HOME_NAME = {"any": "Either", **SHORT}
NAME_HOME = {v: k for k, v in HOME_NAME.items()}
PEOPLE = [p for p in lib["people"] if p != "Priya"]

ss = st.session_state
ss.setdefault("f", L.default_filters())
ss.setdefault("query", "")
ss.setdefault("log", [])
ss.setdefault("t0", None)
ss.setdefault("saved", {})
ss.setdefault("open", None)
ss.setdefault("around", None)
ss.setdefault("found", None)
ss.setdefault("parsed", None)
ss.setdefault("flood_msg", "")
ss.setdefault("std_pages", 1)
ss.setdefault("std_query", "")


def log(event, **kw):
    t = round(time.time() - ss.t0, 1) if ss.t0 else 0.0
    ss.log.append(dict(t=t, event=event, **kw))


def on_std_search():
    ss.std_query, ss.std_pages = ss.w_std.strip(), 1


def on_std_more():
    ss.std_pages += 1


def taps():
    return sum(1 for e in ss.log if e["event"] not in ("search", "found"))


def sync_widgets():
    f = ss.f
    ss.w_stage, ss.w_res = STAGE_NAME[f["stage"]], HOME_NAME[f["residence"]]
    ss.w_people, ss.w_clutter = list(f["people"]), f["hide_clutter"]


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def on_search():
    q = ss.w_query.strip()
    if not q:
        return
    ss.query, ss.t0, ss.log, ss.found, ss.open, ss.around, ss.parsed = q, time.time(), [], None, None, None, None
    f = L.default_filters()
    p = agent._parse_rules(lib, q)  # cheap: pick up a stage word already in the query
    f["stage"] = p["stage"] or "any"
    f["keywords"] = [w for w in L.terms(q) if w != CHILD.lower()]
    ss.f = f
    flooded, n, msg = L.is_flooded(lib, q)
    ss.flood_msg = msg if flooded else ""
    sync_widgets()
    log("search", query=q, flooded=flooded, native_results=len(L.native_search(lib, q)))


def on_chip(group):
    ss.open = ss.around = None
    log("chip", group=group, value=ss.get({"stage": "w_stage", "residence": "w_res", "people": "w_people",
                                            "hide_clutter": "w_clutter"}[group]))


def on_shift(delta):
    ss.f["shift"] += delta
    log("chip", group="stage_shift", value=ss.f["shift"])


def on_context():
    text = ss.w_context.strip()
    if not text:
        return
    if not ss.t0:
        ss.t0 = time.time()
    p = agent.parse_context(lib, text)
    f = ss.f
    f.update(stage=p["stage"] or f["stage"], shift=0, residence=p["residence"] or f["residence"],
             people=p["people"] or f["people"],
             keywords=sorted(set(f["keywords"]) | set(p["keywords"])), exclude=p["exclude"])
    ss.parsed = p
    ss.open = ss.around = None
    sync_widgets()
    log("typed_context", text=text, parser=p["parser"])


def on_which(opt):
    f = ss.f
    f["keywords"] = list(dict.fromkeys(f["keywords"] + opt["keywords"]))
    f["exclude"] = list(dict.fromkeys(f["exclude"] + opt["exclude"]))
    ss.open = ss.around = None
    log("which_one", picked=opt["moment_id"])


def on_reset():
    ss.f = L.default_filters()
    ss.f["keywords"] = [w for w in L.terms(ss.query) if w != CHILD.lower()]
    ss.parsed = None
    sync_widgets()
    log("chip", group="reset", value=None)


def on_open(mid):
    ss.open = None if ss.open == mid else mid
    log("open_moment", moment=mid)


def on_around(mid):
    ss.around = None if ss.around == mid else mid
    log("days_around", moment=mid)


def on_found(m_id, date_s, caption):
    log("found", moment=m_id, date=date_s)
    ss.found = {"moment": m_id, "date": date_s, "caption": caption, "seconds": ss.log[-1]["t"], "taps": taps()}


def on_save(m_id, date_s, caption):
    label = ss.get(f"label_{m_id}", "").strip() or caption
    ss.saved[label] = {"moment": m_id, "date": date_s, "caption": caption}
    log("save_milestone", label=label, moment=m_id)


def on_open_saved(label):
    m = ss.saved[label]
    ss.t0 = ss.t0 or time.time()
    log("found", moment=m["moment"], date=m["date"], via="saved milestone")
    ss.found = {"moment": m["moment"], "date": m["date"], "caption": m["caption"],
                "seconds": ss.log[-1]["t"], "taps": taps(), "via": "saved"}


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def emoji(it):
    tags = it.get("tags", [])
    return next((e for k, e in EMOJI if k in tags), "📷")


def card(m, rank):
    b = m["best"]
    cues = ""
    if m["hits"]:
        cues += f"<span style='color:{BLUE}'>matches: {', '.join(m['hits'])}</span> "
    if m["excluded_hits"]:
        cues += f"<span style='color:{ORANGE}'>looks like what you ruled out: {', '.join(m['excluded_hits'])}</span>"
    untagged = (f"<div style='color:{GREY};font-size:14px'>{m['untagged']} photo(s) here where {CHILD}'s face "
                f"wasn't recognised, included because of the day.</div>") if m["untagged"] else ""
    border = BLUE if rank == 1 else "#CCCCCC"
    st.markdown(f"""
<div style="border:2px solid {border};border-radius:10px;padding:12px;margin-bottom:6px">
  <div style="font-size:44px;line-height:1">{emoji(b)}</div>
  <div style="font-size:17px;font-weight:600;margin-top:6px">{b['caption']}</div>
  <div style="font-size:14px;color:{GREY};margin-top:4px">{L.explain(lib, m)}</div>
  <div style="font-size:14px;margin-top:4px">{cues}</div>{untagged}
</div>""", unsafe_allow_html=True)


def frames(items, cols=4):
    c = st.columns(cols)
    for i, it in enumerate(items):
        with c[i % cols]:
            st.markdown(f"<div style='border:1px solid #DDD;border-radius:6px;padding:6px;font-size:14px'>"
                        f"<span style='font-size:26px'>{emoji(it)}</span><br>{it['caption']}<br>"
                        f"<span style='color:{GREY}'>{it['date']} {it['time']} · {it['source']}</span></div>",
                        unsafe_allow_html=True)


# Demo links (e.g. ?demo=first_steps) open the app mid-task, for the deck and walkthroughs.
DEMOS = {"first_steps": dict(query=CHILD, stage="first_steps", residence="Old flat, Koramangala", people=["Amma"]),
         "cake": dict(query=f"{CHILD} birthday cake", stage="first_steps", residence="Old flat, Koramangala", people=[])}
_qp = st.query_params if hasattr(st, "query_params") else st.experimental_get_query_params()
_demo = _qp.get("demo")
_demo = _demo[0] if isinstance(_demo, list) else _demo
if _demo in DEMOS and not ss.get("demo_loaded"):
    ss.demo_loaded = True
    d = DEMOS[_demo]
    ss.w_query = d["query"]
    on_search()
    ss.f.update(stage=d["stage"], residence=d["residence"], people=d["people"])
    sync_widgets()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.markdown("## 📷 Moment Finder")
st.caption(f"Prototype on a synthetic library of {len(lib['items'])} items · "
           f"typed context uses {'Groq (' + agent.MODEL + ')' if agent._groq_available() else 'the offline rules parser'}")

with st.expander(f"You are Priya: {CHILD}'s details and four things to try"):
    st.markdown(f"{CHILD} was born **{lib['birthdate'].strftime('%d %b %Y')}**. You moved from the "
                f"old flat (Koramangala) to the new flat (HSR Layout) in **April 2023**. "
                "**Amma & Appa** = your in-laws · **Nani** = your mother · **Rahul** = your husband")
    st.markdown("1. His **first steps**, the week your in-laws stayed, before the move.\n"
                "2. His **first haircut**; your mother was holding him.\n"
                "3. His **first day at daycare**; he cried at the gate, it was raining.\n"
                "4. The **cake smash** at home on his first birthday, not the party at the hall.")
    st.caption(f"Start by searching for what you'd type today, e.g. \"{CHILD}\".")

tab_mf, tab_std, tab_how = st.tabs(["Moment Finder", "Standard search (today)", "How it works"])

def moment_finder():
    st.text_input("Search your photos", key="w_query", on_change=on_search,
                  placeholder=f"e.g. {CHILD}, {CHILD} walking, birthday cake")
    if not ss.query:
        st.info(f"Type what you'd search today and press Enter. Try **{CHILD}**.")
        return

    for label, m in L.match_saved(ss.saved, ss.query):
        st.button(f"⭐ Your saved milestone: {label} ({m['date']})", key=f"sv_{label}",
                  on_click=on_open_saved, args=(label,), type="primary")

    if ss.flood_msg:
        st.markdown(f"<div style='background:#EAF2FA;border-left:5px solid {BLUE};padding:10px;"
                    f"border-radius:4px;font-size:16px'><b>{ss.flood_msg}</b> That's a lot to scroll. "
                    f"What do you remember about when it was?</div>", unsafe_allow_html=True)

    f = ss.f
    if "w_stage" not in ss:
        sync_widgets()
    if NAME_STAGE[ss.w_stage] != f["stage"]:  # widgets are the source of truth for chips
        f["stage"], f["shift"] = NAME_STAGE[ss.w_stage], 0
    f["residence"], f["people"] = NAME_HOME[ss.w_res], list(ss.w_people)
    f["hide_clutter"] = ss.w_clutter
    cc = L.chip_counts(lib, f)
    order = L.chip_group_order(lib, f)

    def counts_line(names_counts):
        st.caption(" · ".join(f"{n} **{c}**" for n, c in names_counts))

    def chip_stage():
        c1, c2, c3 = st.columns([8, 1, 1])
        with c1:
            st.radio(f"How old was {CHILD}?", list(STAGE_NAME.values()), key="w_stage", horizontal=True,
                     on_change=on_chip, args=("stage",))
            counts_line((L.STAGES[k][0], cc["stage"][k]) for k in L.STAGES)
        if f["stage"] != "any":
            c2.button("◀ Earlier", on_click=on_shift, args=(-2,), help="Shift the age window 2 months earlier")
            c3.button("Later ▶", on_click=on_shift, args=(2,), help="Shift the age window 2 months later")
            if f["shift"]:
                lo, hi = L.stage_window(f)
                st.caption(f"Age window shifted: {max(lo, 0):.0f}–{hi:.0f} months")

    def chip_res():
        st.radio("Which home?", list(HOME_NAME.values()), key="w_res", horizontal=True,
                 on_change=on_chip, args=("residence",))
        counts_line((SHORT[r], cc["residence"][r]) for r in HOMES[1:])

    def chip_people():
        st.multiselect("Who else was there?", PEOPLE, key="w_people", on_change=on_chip, args=("people",))
        ranked = L.people_options(lib, f)
        if ranked:
            st.caption("Seen with " + CHILD + " in these results: " + " · ".join(
                f"{p} ({lib['people'][p]}) **{cc['people'].get(p, 0)}**" for p in ranked))

    render = {"stage": chip_stage, "residence": chip_res, "people": chip_people}
    for g in order:
        render[g]()
    c1, c2 = st.columns([3, 1])
    c1.toggle("Hide WhatsApp forwards and screenshots", key="w_clutter", on_change=on_chip,
              args=("hide_clutter",))
    c2.button("Clear filters", on_click=on_reset)

    with st.expander("Or just tell me what you remember", expanded=False):
        with st.form("ctx", clear_on_submit=False):
            st.text_input("In your own words", key="w_context",
                          placeholder="the week my in-laws stayed, before we moved, when he'd just started walking")
            st.form_submit_button("Narrow it down", on_click=on_context)
        if ss.parsed:
            p = ss.parsed
            bits = [f"age: {L.STAGES[p['stage']][0]}" if p["stage"] else None,
                    f"home: {SHORT[p['residence']]}" if p["residence"] else None,
                    f"with: {', '.join(p['people'])}" if p["people"] else None,
                    f"looking for: {', '.join(p['keywords'])}" if p["keywords"] else None,
                    f"not: {', '.join(p['exclude'])}" if p["exclude"] else None]
            st.caption(f"Understood ({p['parser']}): " + " · ".join(b for b in bits if b))

    res, relax_msg, used = L.results_no_dead_end(lib, f)
    ms = L.moments(lib, res, used)
    if relax_msg:
        st.markdown(f"<div style='border-left:5px solid {ORANGE};padding:8px;font-size:15px'>{relax_msg}</div>",
                    unsafe_allow_html=True)
    st.markdown(f"**{len(res)} photos → {len(ms)} moments.** Best matches:")

    if ss.found:
        st.success(f"✅ Shared “{ss.found['caption']}” to WhatsApp (simulated) · "
                   f"{ss.found['seconds']} s · {ss.found['taps']} taps")

    q = L.which_one(lib, ms)
    if q and not ss.found:
        st.markdown(f"<div style='border-left:5px solid {BLUE};padding:8px;font-size:16px'>"
                    f"<b>{q['question']}</b></div>", unsafe_allow_html=True)
        for i, opt in enumerate(q["options"]):
            st.button(opt["label"], key=f"w1_{i}_{opt['moment_id']}", on_click=on_which, args=(opt,))

    for rank, m in enumerate(ms[:6], 1):
        mid = m["id"]
        card(m, rank)
        b1, b2, b3, b4 = st.columns(4)
        b1.button(f"See all {len(m['items'])}", key=f"o_{mid}", on_click=on_open, args=(mid,))
        b2.button("Days around this one", key=f"a_{mid}", on_click=on_around, args=(mid,))
        b3.button("This is it → Share to WhatsApp", key=f"f_{mid}", type="primary",
                  on_click=on_found, args=(mid, m["date"].isoformat(), m["best"]["caption"]))
        with b4.expander("⭐ Save as milestone"):
            ss.setdefault(f"label_{mid}", m["best"]["event"] or "")
            st.text_input("Label", key=f"label_{mid}")
            st.button("Save", key=f"s_{mid}", on_click=on_save,
                      args=(mid, m["date"].isoformat(), m["best"]["caption"]))
        if ss.open == mid:
            frames(m["items"])
        if ss.around == mid:
            st.caption(f"3 days either side of {m['date'].strftime('%d %b %Y')}")
            for n in L.days_around(lib, m, used):
                st.markdown(f"- **{n['date'].strftime('%d %b')}** · {emoji(n['best'])} {n['best']['caption']} "
                            f"· {len(n['items'])} photo(s)")
        st.write("")


with tab_mf:
    moment_finder()

with tab_std:
    st.markdown("This is how search behaves today: a literal match on who or what is in the photo, "
                "newest first, every frame and burst shown.")
    st.text_input("Search", key="w_std", placeholder=CHILD, on_change=on_std_search)
    if ss.std_query:
        r = L.native_search(lib, ss.std_query)
        shown = r[:24 * ss.std_pages]
        st.markdown(f"**{len(r)} results**, newest first. Showing {len(shown)}.")
        cols = st.columns(6)
        for i, it in enumerate(shown):
            with cols[i % 6]:
                st.markdown(f"<div style='border:1px solid #DDD;border-radius:6px;padding:6px;font-size:14px'>"
                            f"<span style='font-size:26px'>{emoji(it)}</span><br>{it['caption']}<br>"
                            f"<span style='color:{GREY}'>{it['date']} {it['time']} · {it['source']}</span></div>",
                            unsafe_allow_html=True)
        if len(shown) < len(r):
            st.button("Show 24 more (scroll)", on_click=on_std_more)

with tab_how:
    st.markdown(f"""
**Why:** a parent's child is in almost every photo, so searching for {CHILD} returns everything.
Parents remember the *context* (how old he was, which home, who was visiting) and Google
Photos can't search by that. Full reasoning: `04_problem_definition.md` and `05_solution.md`.

**What happens here**
1. **Flood diagnosis.** A search for the child (or a milestone word, or too many or zero results)
   triggers Moment Finder instead of a wall of photos.
2. **Chips with live counts.** Age stage (computed from the birthdate, so it needs no faces
   or location), home (residence period), and people (who appears with him in these results).
   The chip group that splits the current results most evenly is shown first.
3. **Typed context (optional, the only LLM step).** Your words become the same filters.
   The output is checked against this library's people, homes and stages, so it can't
   invent any.
4. **No dead ends.** If filters leave nothing, the least certain one is relaxed and the
   app says which.
5. **Moments, not frames.** Photos are grouped by day and event, bursts collapsed, forwards
   and screenshots hidden, and each moment explains why it's shown.
6. **Save as milestone,** so next time the search is one tap.

**Honesty notes**
- The library is synthetic ({len(lib['items'])} items; a real one has 20,000+). Ranking uses only
  metadata a phone could plausibly have (vision tags, auto events such as birthdays and
  festivals, location, faces, dates), never the captions you see on the cards.
- "First steps" or "first haircut" are not labels in the data. The app gets there from age,
  home and people.
- Run `python verify.py` to check all four test tasks.
""")
