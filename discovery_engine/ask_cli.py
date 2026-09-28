"""
Ask the discovery engine a question from the command line.

    python ask_cli.py --build                                  # (re)build data/corpus.json
    python ask_cli.py "What do people remember about photos they can't find?"
    python ask_cli.py "..." --sources play_store,app_store
"""
import argparse
import textwrap

from _env import load_env
load_env()

from ask.engine import SOURCE_NAMES, Engine, _src, build_corpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--sources", default="")
    args = ap.parse_args()
    if args.build:
        docs = build_corpus()
        print(f"corpus: {len(docs)} posts with retrieval intent")
    if not args.question:
        return
    eng = Engine()
    r = eng.ask(args.question, [s for s in args.sources.split(",") if s] or None)
    print(f"\nQ: {r['question']}\nsearched: {r['searches']} facets: {r.get('facets')}\n{r['matched']} matching posts; read the top {len(r['hits'])}"
          f" ({r.get('model')}, {r.get('seconds')}s)\n")
    print(textwrap.fill(r["summary"], 100), "\n")
    for t in r["themes"]:
        print(f"■ {t['theme']}  — {t['support']} of {len(r['hits'])} posts · "
              + ", ".join(f"{s} {n}" for s, n in t["sources"]))
        print(textwrap.fill(t["explanation"], 100, initial_indent="  ", subsequent_indent="  "))
        for i in t["posts"][:2]:
            h = r["hits"][i - 1]
            print(f"    [{i}] {SOURCE_NAMES.get(_src(h))}: \"{h['text'][:180]}\"")
    print(f"\noff-topic among the {len(r['hits'])} read: {len(r.get('off_topic') or [])}")
    if r.get("not_covered"):
        print("\nNot covered:", r["not_covered"])


if __name__ == "__main__":
    main()
