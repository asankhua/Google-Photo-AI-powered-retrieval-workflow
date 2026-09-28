"""
AI-simulated MVP test sessions (NOT real participants).

Three LLM personas from the target segment each do two tasks from the "You are Priya"
panel in both conditions, then answer the post-task and interview questions from
06_mvp_testing.md §1a. What is simulated and what is real:

  simulated  the parent's choices (what to type, which chip or photo to tap) and answers
  real       every screen: results come from the MVP's own code (mvp/library.py
             native_search / filters / moments / which_one, mvp/agent.py parse_context),
             so ranks, counts, found / wrong pick and taps are computed by the app

Each condition runs in a fresh LLM context (the persona knows only the task brief), so the
answer found in one condition can't leak into the other. This is stricter than a human
session, where the parent remembers; order still alternates (T1, T3 start standard).
Time isn't simulated: we report taps and steps, not seconds.

Run:  python run_tests.py          (all testers)
      python run_tests.py T2       (one tester)
"""
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MVP = os.path.join(HERE, "..", "mvp")
sys.path.insert(0, MVP)
from _env import load_env  # noqa: E402

load_env(os.path.join(MVP, ".env"))
import agent  # noqa: E402
import library as L  # noqa: E402
from groq import APIConnectionError, Groq, RateLimitError  # noqa: E402

MODEL = os.environ.get("SIM_MODEL", "openai/gpt-oss-120b")
MAX_STEPS = 10
PAGE = 24
lib = L.load()
CHILD = lib["child"]["name"]

BRIEF = (f"You are role-playing Priya for this test. {CHILD} was born 10 Feb 2021. You moved from the old "
         "flat (Koramangala) to the new flat (HSR Layout) in April 2023. Amma & Appa = your in-laws, "
         "Nani = your mother, Rahul = your husband, Meera = cousin.")
TASKS = {
    1: ("First steps", "2022-03-12", "His first steps, the week your in-laws stayed, before the move."),
    2: ("First haircut", "2021-10-30", "His first haircut; your mother was holding him."),
    3: ("First day at daycare", "2023-06-05", "His first day at daycare; he cried at the gate, it was raining."),
    4: ("Cake smash", "2022-02-10", "The cake smash at home on his first birthday, not the party at the hall."),
}
# id, persona, tasks, which condition first
TESTERS = [
    ("T1", "Neha, 32, Gurugram, HR manager. Daughter, 4. ~24k photos, Google One. Types full sentences "
           "into search and expects it to understand. Impatient on her phone.", [1, 4], "standard"),
    ("T2", "Vikram, 36, Chennai, bank operations. Son, 5. ~19k photos. Prefers tapping to typing, scrolls "
           "a lot, sceptical of 'AI' features and says so.", [2, 3], "moment_finder"),
    ("T3", "Ananya, 29, Kolkata, graphic designer. Son, 3. ~28k photos, Google One. Careful about privacy: "
           "uneasy when apps know her home or her family's faces.", [3, 1], "standard"),
]

PERSONA_RULES = """You are a real parent taking part in a usability test of a photo app prototype, on a call.
Who you are: {persona}
{brief}
Your task: find this photo and send it: "{task}"
Behave like this person really would on their phone. You only know what is on screen and in your brief;
don't assume the app's internals. You may pick a photo you're unsure of, keep looking, or give up.
Reply with ONLY a JSON object: {{"say": "<what you'd say aloud, 1-2 short sentences>", "action": ..., ...}}
{actions}"""

STD_ACTIONS = """Actions:
  {"action":"search","query":"<words>"}      type a new search
  {"action":"more"}                           scroll to the next 24
  {"action":"pick","n":<number on screen>}   this is the photo, send it
  {"action":"give_up"}"""

MF_ACTIONS = """Actions (one per reply):
  {"action":"search","query":"<words>"}                 type a new search
  {"action":"age","value":"<one of the age options>"}   tap an age chip
  {"action":"home","value":"<Old flat|New flat|Either>"}
  {"action":"person","name":"<name>"} / {"action":"remove_person","name":"<name>"}
  {"action":"shift","by":-2 or 2}                       move the age window earlier/later
  {"action":"tell","text":"<your own words>"}           the 'just tell me what you remember' box
  {"action":"which","option":1 or 2}                    answer the 'which one?' question
  {"action":"see_all","n":<moment number>} / {"action":"days_around","n":<moment number>}
  {"action":"pick","n":<moment number>,"save_label":"<label or empty>"}   This is it -> share; optionally save as milestone
  {"action":"give_up"}"""

client = Groq(timeout=90, max_retries=2)
tokens_used = 0
FALLBACK = "openai/gpt-oss-20b"
models_used = set()


def switch_model():
    """Daily cap on the main model: carry on with the smaller one and record it."""
    global MODEL
    print(f"daily token cap on {MODEL}; switching to {FALLBACK}", flush=True)
    MODEL = FALLBACK


def llm(messages, max_tokens=700):
    global tokens_used
    for attempt in range(8):  # free tier: 8k tokens a minute
        try:
            r = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.8,
                                               max_completion_tokens=max_tokens, reasoning_effort="low")
            tokens_used += r.usage.total_tokens if r.usage else 0
            models_used.add(MODEL)
            return (r.choices[0].message.content or "").strip()
        except (RateLimitError, APIConnectionError) as e:
            if "tokens per day" in str(e):
                if MODEL == FALLBACK:
                    raise
                switch_model()
                continue
            time.sleep(15 * (attempt + 1))
    raise RuntimeError("Groq kept failing")


def as_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        a = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}
    if isinstance(a.get("action"), dict):  # model sometimes nests: {"say":..,"action":{"action":"search",..}}
        a = {"say": a.get("say", ""), **a["action"]}
    return a


# ---------------------------------------------------------------------------
# Standard search: literal match, newest first, 24 at a time (as in the app's tab)
# ---------------------------------------------------------------------------
def run_standard(persona, task, target):
    sys_p = PERSONA_RULES.format(persona=persona, brief=BRIEF, task=task, actions=STD_ACTIONS)
    msgs = [{"role": "system", "content": sys_p},
            {"role": "user", "content": "Screen: Google Photos search, empty search box. What do you do?"}]
    res, shown, log = [], 0, []
    out = {"found": False, "wrong_picks": 0, "taps": 0, "pages": 0, "queries": [], "target_rank": None}
    for step in range(MAX_STEPS):
        a = as_json(llm(msgs))
        msgs.append({"role": "assistant", "content": json.dumps(a)})
        act = a.get("action")
        log.append({"say": a.get("say", ""), "action": {k: v for k, v in a.items() if k != "say"}})
        if act == "search":
            q = str(a.get("query", "")).strip()
            res, shown = L.native_search(lib, q), 0
            out["queries"].append(q)
            out["target_rank"] = next((i for i, it in enumerate(res, 1) if it["date"] == target), None)
            out["taps"] += 1
        elif act == "more":
            out["pages"] += 1
            out["taps"] += 1
        elif act == "pick":
            n = int(a.get("n", 0))
            it = res[n - 1] if 0 < n <= len(res) else None
            out["taps"] += 1
            if it and it["date"] == target:
                out["found"] = True
                log[-1]["result"] = f"picked #{n}: {it['caption']} ({it['date']}) = TARGET"
                break
            out["wrong_picks"] += 1
            log[-1]["result"] = f"picked #{n}: {it['caption'] if it else '?'} ({it['date'] if it else ''}) = wrong"
            msgs.append({"role": "user", "content": "Moderator: you sent that one. The family says that's not "
                         "the photo they meant. Keep going or give up."})
            continue
        elif act == "give_up":
            break
        else:
            msgs.append({"role": "user", "content": "Moderator: please reply with one action as JSON."})
            continue
        if act == "search":
            page, start = res[:PAGE], 1
            shown = len(page)
        else:
            page, start = res[shown:shown + PAGE], shown + 1
            shown += len(page)
        lines = [f"{start + i}. {it['caption']} · {it['date']} {it['time']} · {it['source']}"
                 for i, it in enumerate(page)]
        more = "End of results." if shown >= len(res) else "(more below)"
        screen = f"Screen: {len(res)} results, newest first. Showing {start}-{shown}:\n" + \
                 ("\n".join(lines) if lines else "(nothing)") + f"\n{more}"
        log[-1]["screen"] = f"{len(res)} results; showing {start}-{shown}"
        msgs.append({"role": "user", "content": screen})
    out["steps"] = len(log)
    out["ended"] = "found" if out["found"] else (
        "gave up" if log and log[-1]["action"].get("action") == "give_up" else "step limit (moderator stops)")
    return out, log


# ---------------------------------------------------------------------------
# Moment Finder: the same state and functions app.py uses
# ---------------------------------------------------------------------------
STAGE_NAMES = {"any": "Any age", **{k: v[0] for k, v in L.STAGES.items()}}
HOME_SHORT = {"any": "Either", **{r["name"]: r["short"] for r in lib["residences"]}}


def mf_screen(st):
    f = st["f"]
    res, relax, used = L.results_no_dead_end(lib, f)
    ms = L.moments(lib, res, used)
    cc = L.chip_counts(lib, f)
    lines = []
    if st["flood"]:
        lines.append(f"Banner: {st['flood']} That's a lot to scroll. What do you remember about when it was?")
    lines.append(f"Chip 'How old was {CHILD}?' (selected: {STAGE_NAMES[f['stage']]}"
                 f"{', shifted ' + str(f['shift']) + ' months' if f['shift'] else ''}): " +
                 " · ".join(f"{STAGE_NAMES[k]} {cc['stage'][k]}" for k in L.STAGES))
    lines.append(f"Chip 'Which home?' (selected: {HOME_SHORT[f['residence']]}): " +
                 " · ".join(f"{HOME_SHORT[r]} {cc['residence'][r]}" for r in cc["residence"] if r != "any"))
    lines.append(f"Chip 'Who else was there?' (selected: {', '.join(f['people']) or 'none'}): seen with {CHILD}: " +
                 " · ".join(f"{p} ({lib['people'][p]}) {cc['people'].get(p, 0)}" for p in L.people_options(lib, f)))
    if st["parsed"]:
        p = st["parsed"]
        lines.append("Understood: " + "; ".join(x for x in [
            p["stage"] and f"age {STAGE_NAMES[p['stage']]}", p["residence"] and f"home {HOME_SHORT[p['residence']]}",
            p["people"] and f"with {', '.join(p['people'])}", p["keywords"] and f"looking for {', '.join(p['keywords'])}",
            p["exclude"] and f"not {', '.join(p['exclude'])}"] if x))
    if relax:
        lines.append(relax)
    lines.append(f"{len(res)} photos -> {len(ms)} moments. Best matches:")
    q = L.which_one(lib, ms)
    if q:
        lines.append(f"Question: {q['question']}  Option 1: {q['options'][0]['label']}  |  "
                     f"Option 2: {q['options'][1]['label']}")
    for i, m in enumerate(ms[:6], 1):
        lines.append(f"Moment {i}: \"{m['best']['caption']}\" · {L.explain(lib, m)}"
                     + (f" · matches: {', '.join(m['hits'])}" if m["hits"] else ""))
    return "\n".join(lines), ms, q


def run_mf(persona, task, target):
    sys_p = PERSONA_RULES.format(persona=persona, brief=BRIEF, task=task, actions=MF_ACTIONS)
    msgs = [{"role": "system", "content": sys_p},
            {"role": "user", "content": "Screen: Moment Finder, empty search box 'Search your photos'. What do you do?"}]
    st = {"f": None, "flood": "", "parsed": None}
    ms, q, log = [], None, []
    out = {"found": False, "wrong_picks": 0, "taps": 0, "first_cue": None, "target_rank_first": None,
           "target_rank_at_pick": None, "used_which_one": False, "saved": None, "queries": []}

    def tap(kind):
        out["taps"] += 1
        out["first_cue"] = out["first_cue"] or kind

    for step in range(MAX_STEPS):
        a = as_json(llm(msgs))
        msgs.append({"role": "assistant", "content": json.dumps(a)})
        act = a.get("action")
        log.append({"say": a.get("say", ""), "action": {k: v for k, v in a.items() if k != "say"}})
        extra = ""
        if act != "search" and st["f"] is None:
            msgs.append({"role": "user", "content": "Moderator: start by typing a search."})
            continue
        if act == "search":  # same as app.on_search (not counted as a tap, like the app)
            qtext = str(a.get("query", "")).strip()
            out["queries"].append(qtext)
            f = L.default_filters()
            f["stage"] = agent._parse_rules(lib, qtext)["stage"] or "any"
            f["keywords"] = [w for w in L.terms(qtext) if w != CHILD.lower()]
            fl, _, msg = L.is_flooded(lib, qtext)
            st.update(f=f, flood=msg if fl else "", parsed=None)
        elif act == "age":
            key = next((k for k, v in STAGE_NAMES.items() if v.lower() == str(a.get("value", "")).lower()), None)
            if key:
                st["f"].update(stage=key, shift=0)
            tap("age")
        elif act == "home":
            key = next((k for k, v in HOME_SHORT.items() if v.lower() == str(a.get("value", "")).lower()), None)
            if key:
                st["f"]["residence"] = key
            tap("home")
        elif act in ("person", "remove_person"):
            name = str(a.get("name", "")).strip().title()
            if name in lib["people"]:
                ppl = set(st["f"]["people"])
                ppl = ppl | {name} if act == "person" else ppl - {name}
                st["f"]["people"] = sorted(ppl)
            tap("people")
        elif act == "shift":
            st["f"]["shift"] += 2 if int(a.get("by", 2)) > 0 else -2
            tap("age")
        elif act == "tell":  # same as app.on_context
            p = agent.parse_context(lib, str(a.get("text", "")))
            f = st["f"]
            f.update(stage=p["stage"] or f["stage"], shift=0, residence=p["residence"] or f["residence"],
                     people=p["people"] or f["people"],
                     keywords=sorted(set(f["keywords"]) | set(p["keywords"])), exclude=p["exclude"])
            st["parsed"] = p
            tap("typed context")
        elif act == "which" and q:
            opt = q["options"][0 if int(a.get("option", 1)) == 1 else 1]
            st["f"]["keywords"] = list(dict.fromkeys(st["f"]["keywords"] + opt["keywords"]))
            st["f"]["exclude"] = list(dict.fromkeys(st["f"]["exclude"] + opt["exclude"]))
            out["used_which_one"] = True
            tap("which one?")
        elif act in ("see_all", "days_around"):
            n = int(a.get("n", 0))
            m = ms[n - 1] if 0 < n <= len(ms) else None
            tap(act)
            if m and act == "see_all":
                extra = f"Moment {n}, all {len(m['items'])} photos: " + " | ".join(
                    f"{it['caption']} {it['time']}" for it in m["items"])
            elif m:
                extra = f"3 days either side of moment {n}: " + " | ".join(
                    f"{x['date']}: {x['best']['caption']} ({len(x['items'])})" for x in L.days_around(lib, m, st["f"]))
        elif act == "pick":
            n = int(a.get("n", 0))
            m = ms[n - 1] if 0 < n <= len(ms) else None
            if m and m["date"].isoformat() == target:
                out["found"] = True
                out["target_rank_at_pick"] = n
                label = str(a.get("save_label", "") or "").strip()
                out["saved"] = label or None
                log[-1]["result"] = f"picked moment {n}: {m['best']['caption']} ({m['date']}) = TARGET"
                break
            out["wrong_picks"] += 1
            log[-1]["result"] = f"picked moment {n}: {m['best']['caption'] if m else '?'} = wrong"
            msgs.append({"role": "user", "content": "Moderator: you sent that one. The family says that's not "
                         "the photo they meant. Keep going or give up."})
            continue
        elif act == "give_up":
            break
        else:
            msgs.append({"role": "user", "content": "Moderator: please reply with one available action as JSON."})
            continue
        screen, ms, q = mf_screen(st)
        rank = next((i for i, m in enumerate(ms, 1) if m["date"].isoformat() == target), None)
        if out["target_rank_first"] is None and act == "search":
            out["target_rank_first"] = rank
        log[-1]["screen"] = f"{sum(len(m['items']) for m in ms)} photos, {len(ms)} moments; target at #{rank}" + \
                            (f"; which-one asked" if q else "")
        msgs.append({"role": "user", "content": "Screen:\n" + screen + ("\n" + extra if extra else "")})
    out["steps"] = len(log)
    out["ended"] = "found" if out["found"] else (
        "gave up" if log and log[-1]["action"].get("action") == "give_up" else "step limit (moderator stops)")
    return out, log


# ---------------------------------------------------------------------------
# Post-task and interview questions (06_mvp_testing.md §1a)
# ---------------------------------------------------------------------------
QUESTIONS = [
    "Which was easier for you, the standard search or Moment Finder? Why?",
    "Would you use Moment Finder for your own child's photos? 1 to 5, and why.",
    "How sure were you that you'd picked the right photo, in each? 1 to 5.",
    "Was anything uncomfortable, like the people or home chips, or the app knowing your family's faces?",
    "Is this how you'd search for your own child? What would you have tried next?",
    "Have you ever looked for the same photo more than once? Would 'save as milestone' help?",
    "Anything that confused you, or that you'd change?",
]
INTERVIEW_RULES = """You are this real parent, just after a usability test on a call: {persona}
Stay in character: speak like a busy parent, 1-4 sentences, honest, a bit messy. Don't praise the
prototype to be nice; if something annoyed you, say it. Base every answer on what actually happened:
{summary}"""


def summarise(runs):
    out = []
    for r in runs:
        o = r["out"]
        out.append(f"- Task '{r['task']}' in {r['cond']}: {'found it' if o['found'] else 'did NOT find it'}; "
                   f"{o['taps']} taps; {o['wrong_picks']} wrong picks; searches typed: {o['queries']}. "
                   f"Your thoughts during it: " + " / ".join(s["say"] for s in r["log"] if s.get("say")))
    return "\n".join(out)


def interview(persona, runs):
    msgs = [{"role": "system", "content": INTERVIEW_RULES.format(persona=persona, summary=summarise(runs))}]
    turns = []
    for qn in QUESTIONS:
        msgs.append({"role": "user", "content": qn})
        a = llm(msgs, max_tokens=500)
        msgs.append({"role": "assistant", "content": a})
        turns.append((qn, a))
    return turns


def md_run(r):
    o = r["out"]
    lines = [f"### {r['task']} · {r['cond']}", "",
             "| Step | Said (simulated) | Action | Screen / result (from the MVP code) |", "|---|---|---|---|"]
    for i, s in enumerate(r["log"], 1):
        act = json.dumps(s["action"], ensure_ascii=False).replace("|", "/")
        lines.append(f"| {i} | {s['say'].replace('|', '/')} | `{act}` | {s.get('result') or s.get('screen', '')} |")
    lines += ["", "**Result:** " + json.dumps(o, ensure_ascii=False), ""]
    return lines


def main():
    only = set(sys.argv[1:])
    all_results = json.load(open(os.path.join(HERE, "results.json"))) if os.path.exists(
        os.path.join(HERE, "results.json")) else {}
    for tid, persona, tasks, first in TESTERS:
        if only and tid not in only:
            continue
        runs = []
        models_used.clear()
        order = ["standard", "moment_finder"] if first == "standard" else ["moment_finder", "standard"]
        for tn in tasks:
            name, target, task = TASKS[tn]
            for cond in order:
                out, log = (run_standard if cond == "standard" else run_mf)(persona, task, target)
                runs.append({"task": name, "target": target, "cond": cond, "out": out, "log": log})
                print(f"{tid} {name} {cond}: found={out['found']} taps={out['taps']} "
                      f"wrong={out['wrong_picks']} tokens so far={tokens_used}", flush=True)
        turns = interview(persona, runs)
        lines = [f"# {tid}: AI-simulated MVP test session (not a real participant)", "",
                 f"**Persona:** {persona}", "",
                 f"**Model:** {', '.join(sorted(models_used))} (persona choices and answers) · screens and results computed by the MVP code "
                 f"(`mvp/library.py`, `mvp/agent.py`) · order: {' then '.join(order)} · each condition in a fresh "
                 "context · time not simulated", "", "## Tasks", ""]
        for r in runs:
            lines += md_run(r)
        lines += ["## Post-task interview", ""]
        for qn, a in turns:
            lines += [f"**Moderator:** {qn}", "", f"**{tid}:** {a}", ""]
        open(os.path.join(HERE, f"{tid}.md"), "w").write("\n".join(lines))
        all_results[tid] = {"persona": persona, "runs": [{k: r[k] for k in ("task", "target", "cond", "out")}
                                                         for r in runs],
                            "interview": turns}
        json.dump(all_results, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=str)
        print(f"{tid} saved · tokens {tokens_used}", flush=True)


if __name__ == "__main__":
    main()
