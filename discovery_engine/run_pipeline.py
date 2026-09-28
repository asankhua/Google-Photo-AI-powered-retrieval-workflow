"""
Orchestrator: collect -> classify (Groq) -> aggregate -> save.

Usage:
    export GROQ_API_KEY=...
    python run_pipeline.py --play 300 --appstore 8 [--reddit] [--limit 100]
    python run_pipeline.py --play 0 --appstore 0 --hn --stackexchange --community \
        --youtube --per-source 40 --prefix public   # forums / public discussions
    python run_pipeline.py --play 8000 --countries in,us,gb --sorts newest,relevant \
        --appstore 0 --segment parents          # parent-segment run

Outputs to data/: raw.json, classified.json, summary.json
(with --segment X: X_raw.json, X_classified.json, X_summary.json, so the base run is kept)
"""
import argparse
from collections import Counter
import json
import os
import re

from _env import load_env
load_env()  # pull GROQ_API_KEY / Reddit creds from .env automatically

from analyze.filters import RETRIEVAL_KW, forum_score  # noqa: E402

from collect import play_store, app_store, reddit as reddit_mod
from collect import hackernews, stackexchange, gphotos_community, youtube
from classify.classifier import classify_batch
from analyze.aggregate import summarize
from analyze.segment import SEGMENTS, in_segment

DATA = os.path.join(os.path.dirname(__file__), "data")
FUNNEL = {}  # forum collection funnel, reported in the summary


def _save(name, obj):
    path = os.path.join(DATA, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  saved {path}")


def collect_all(args):
    raw = []
    if args.play:
        for country in args.countries.split(","):
            print(f"[collect] Play Store ({country}) x{args.play} per sort ...")
            try:
                raw += play_store.collect(count=args.play, country=country,
                                          sorts=args.sorts.split(","))
            except Exception as e:
                print(f"  play_store failed: {e}")
    if args.appstore:
        print(f"[collect] App Store x{args.appstore} pages ...")
        try:
            raw += app_store.collect(pages=args.appstore)
        except Exception as e:
            print(f"  app_store failed: {e}")
    if args.reddit:
        print("[collect] Reddit ...")
        try:
            raw += reddit_mod.collect()
        except Exception as e:
            print(f"  reddit failed (need creds?): {e}")
    forum = []
    cache = os.path.join(DATA, "public_forum_all.json")
    if args.cached and os.path.exists(cache):
        forum = json.load(open(cache))
        print(f"[collect] reusing {len(forum)} cached forum posts ({cache})")
    for flag, name, mod in [("hn", "Hacker News", hackernews), ("stackexchange", "Stack Exchange", stackexchange),
                            ("community", "Google Photos Community", gphotos_community),
                            ("youtube", "YouTube comments", youtube)]:
        if getattr(args, flag) and not (args.cached and os.path.exists(cache)):
            print(f"[collect] {name} ...")
            try:
                got = mod.collect()
                print(f"  {len(got)} items")
                forum += got
            except Exception as e:
                print(f"  {flag} failed: {e}")
    if forum and not args.cached:
        with open(cache, "w") as f:
            json.dump(forum, f, default=str)
    if forum and args.per_source:  # forums are large; keep the most on-topic posts
        FUNNEL["forum_collected"] = dict(Counter(r["source"].split("/")[0] for r in forum))
        by_src = {}
        for r in forum:
            r["relevance"] = forum_score(r.get("text", ""))
            if r["relevance"]:
                by_src.setdefault(r["source"].split("/")[0], []).append(r)
        forum = []
        for src, items in by_src.items():
            items.sort(key=lambda r: (-r["relevance"], -(r.get("upvotes") or 0)))
            print(f"[forum filter] {src}: {len(items)} on-topic, keeping {min(len(items), args.per_source)}")
            FUNNEL.setdefault("forum_on_topic", {})[src] = len(items)
            forum += items[: args.per_source]
    raw += forum
    seen, deduped = set(), []
    for r in raw:
        key = (r.get("text") or "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--play", type=int, default=300)
    ap.add_argument("--countries", default="us", help="Play Store countries, e.g. in,us,gb")
    ap.add_argument("--sorts", default="newest", help="newest and/or relevant")
    ap.add_argument("--segment", choices=sorted(SEGMENTS), default=None,
                    help="only classify feedback from this segment; outputs get a prefix")
    ap.add_argument("--appstore", type=int, default=8)
    ap.add_argument("--reddit", action="store_true")
    ap.add_argument("--hn", action="store_true", help="Hacker News comments/stories (Algolia API)")
    ap.add_argument("--stackexchange", action="store_true", help="Stack Exchange questions")
    ap.add_argument("--community", action="store_true", help="Google Photos Help Community threads")
    ap.add_argument("--youtube", action="store_true", help="YouTube comments (needs YOUTUBE_API_KEY)")
    ap.add_argument("--per-source", type=int, default=0,
                    help="keep at most N on-topic items per forum source (most upvoted first)")
    ap.add_argument("--cached", action="store_true", help="reuse data/public_forum_all.json")
    ap.add_argument("--prefix", default="", help="output file prefix, e.g. public")
    ap.add_argument("--limit", type=int, default=0, help="cap items sent to LLM")
    ap.add_argument("--filter", action="store_true",
                    help="only classify reviews matching retrieval keywords")
    args = ap.parse_args()

    prefix = f"{args.segment}_" if args.segment else (f"{args.prefix}_" if args.prefix else "")
    raw = collect_all(args)
    print(f"[collect] total deduped: {len(raw)}")
    collected = len(raw)
    if args.segment:  # keep only the segment (the full collection is large)
        raw = [r for r in raw if in_segment(r.get("text", ""), args.segment)]
        print(f"[segment] {args.segment}: {len(raw)} of {collected}")
    _save(prefix + "raw.json", raw)
    if args.filter:
        raw = [r for r in raw if RETRIEVAL_KW.search(r.get("text", ""))]
        print(f"[filter] retrieval-keyword matches: {len(raw)}")
    if args.limit:
        raw = raw[: args.limit]
    texts = [r["text"][:2000] for r in raw]  # forum posts can be long; cap LLM cost

    print(f"[classify] sending {len(texts)} items to Groq ...")
    classified = classify_batch(
        texts, on_progress=lambda i, n: print(f"  {i}/{n}", end="\r")
    )
    for c, r in zip(classified, raw):
        c["source"], c["url"], c["rating"] = r.get("source"), r.get("url"), r.get("rating")
    print()
    _save(prefix + "classified.json", classified)

    summary = summarize(classified)
    summary["collected_items"] = sum(FUNNEL["forum_collected"].values()) if FUNNEL else collected
    summary.update(FUNNEL)
    if args.segment:
        summary["segment"] = args.segment
        summary["segment_items"] = len(raw)
    _save(prefix + "summary.json", summary)

    print("\n=== OPPORTUNITY RANKING ===")
    for o in summary["opportunities_ranked"]:
        print(f"  {o['failure_mode']:<14} score={o['opportunity_score']:<7} "
              f"count={o['count']} share={o['share_pct']}% sev={o['avg_severity']}")


if __name__ == "__main__":
    main()
