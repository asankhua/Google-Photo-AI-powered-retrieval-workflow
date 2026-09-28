"""
Checks the four test tasks against the sample library, two ways:
  typed  - the parent types what they remember (agent.parse_context)
  chips  - the parent only taps chips (no typing, no LLM)
and compares with standard search. Pass = target moment in the top 3.

Run:  python verify.py          (uses Groq if GROQ_API_KEY is set, else rules)
      python verify.py --rules  (force the offline parser)
"""
import sys

import agent
import library as L
from _env import load_env

TASKS = [
    {"name": "First steps", "target": "2022-03-12", "query": "Arjun walking",
     "typed": "the week my in-laws stayed with us, before we moved, when he'd just started walking",
     "chips": {"stage": "first_steps", "residence": "Old flat, Koramangala", "people": ["Amma"]}},
    {"name": "First haircut", "target": "2021-10-30", "query": "Arjun haircut",
     "typed": "his first haircut, my mother was holding him, around Diwali",
     "chips": {"stage": "baby", "people": ["Nani"]}},
    {"name": "First day at daycare", "target": "2023-06-05", "query": "Arjun daycare",
     "typed": "first day at daycare, he was crying at the gate, it was raining",
     "chips": {"stage": "toddler", "residence": "New flat, HSR Layout"}},
    {"name": "Cake smash at home", "target": "2022-02-10", "query": "Arjun birthday cake",
     "typed": "first birthday cake smash at home, not the party at the hall",
     "chips": {"stage": "first_steps", "residence": "Old flat, Koramangala"}},
]


def rank_of(ms, target):
    for i, m in enumerate(ms, 1):
        if m["date"].isoformat() == target:
            return i
    return None


def run(force_rules=False):
    load_env()
    if force_rules:
        agent._groq_available = lambda: False
    lib = L.load()
    ok = True
    print(f"{'Task':22} {'Std search':>12} {'Typed':>16} {'Chips':>16} {'+which one?':>12}")
    for t in TASKS:
        native = L.native_search(lib, t["query"])
        nat_pos = next((i for i, it in enumerate(native, 1) if it["date"] == t["target"]), None)

        p = agent.parse_context(lib, t["typed"])
        f = L.default_filters()
        f.update(stage=p["stage"] or "any", residence=p["residence"] or "any",
                 people=p["people"], keywords=p["keywords"], exclude=p["exclude"])
        res, _, f = L.results_no_dead_end(lib, f)
        typed_rank = rank_of(L.moments(lib, res, f), t["target"])

        g = L.default_filters()
        g.update(t["chips"])
        g["keywords"] = [w for w in L.terms(t["query"]) if w != lib["child"]["name"].lower()]
        res2, _, g = L.results_no_dead_end(lib, g)
        chip_rank = rank_of(L.moments(lib, res2, g), t["target"])

        # if the app asks "which one?", the tester picks the option matching the target
        ms2 = L.moments(lib, res2, g)
        q = L.which_one(lib, ms2)
        after_q = "-"
        if q:
            opt = next((o for o in q["options"] if o["moment_id"].startswith(f"m-{t['target']}")), None)
            if opt:
                g2 = dict(g, keywords=g["keywords"] + opt["keywords"], exclude=g["exclude"] + opt["exclude"])
                after_q = f"#{rank_of(L.moments(lib, res2, g2), t['target'])}"
        ok &= bool(typed_rank and typed_rank <= 3 and chip_rank and chip_rank <= 3)
        print(f"{t['name']:22} {f'#{nat_pos}/{len(native)}':>12} "
              f"{f'#{typed_rank} of {len(res)}':>16} {f'#{chip_rank} of {len(res2)}':>16} {after_q:>12}"
              f"   [{p['parser']}: {p['stage']}, {p['residence'] and p['residence'][:8]}, "
              f"{p['people']}, kw={p['keywords']}, ex={p['exclude']}]")
    print("PASS" if ok else "FAIL: a target is outside the top 3")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run("--rules" in sys.argv) else 1)
