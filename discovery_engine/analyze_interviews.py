"""
Interview-analysis agent (CLI). Codes transcripts, checks quotes, answers the discovery
questions, and compares the answers with the public-feedback findings.

Usage:
    python analyze_interviews.py                       # the AI-simulated P1-P6 and T1-T3
    python analyze_interviews.py notes/*.md --real     # your own real session notes
    python analyze_interviews.py a.md b.md --out data/interviews_real
    python analyze_interviews.py --reuse                # re-answer after new public data

A transcript is any .md/.txt with the participant's words; interviewer lines should start
with "Interviewer:" or "Moderator:" so quotes are checked against the participant only.
Outputs: <out>.md (report) and <out>.json (codes, counts, answers).
"""
import argparse
import glob
import json
import os
import sys

from _env import load_env
load_env()

from analyze import interviews

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT = (sorted(glob.glob(os.path.join(ROOT, "03_user_research", "synthetic_interviews", "P*.md"))) +
           sorted(glob.glob(os.path.join(ROOT, "06_simulated_tests", "T*.md"))))
PUBLIC = {"app reviews (parents)": "parents_summary.json", "app reviews (all)": "summary.json",
          "public forums": "public_summary.json"}


def load_public():
    out = {}
    for label, f in PUBLIC.items():
        p = os.path.join(HERE, "data", f)
        if os.path.exists(p):
            out[label] = json.load(open(p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--real", action="store_true", help="transcripts are real sessions (default: "
                    "files are treated as simulated if they say 'AI-simulated')")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse the coded transcripts in <out>.json; redo counts, comparison, answers")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "interviews_analysis"))
    args = ap.parse_args()
    files = args.files or DEFAULT
    transcripts = []
    for f in files:
        text = open(f).read()
        sim = not args.real and "simulated" in text[:400].lower()
        transcripts.append({"id": os.path.splitext(os.path.basename(f))[0], "text": text, "simulated": sim})
    print(f"{len(transcripts)} transcripts ({sum(t['simulated'] for t in transcripts)} simulated)")
    codes = json.load(open(args.out + ".json"))["codes"] if args.reuse else None
    res = interviews.run(transcripts, load_public(), codes=codes,
                         on_progress=lambda i, n: print(f"  coded {i}/{n}", flush=True))
    with open(args.out + ".json", "w") as fh:
        json.dump({k: v for k, v in res.items() if k != "report"}, fh, indent=2, default=str)
    with open(args.out + ".md", "w") as fh:
        fh.write(res["report"])
    print(f"saved {args.out}.md and .json")


if __name__ == "__main__":
    sys.exit(main())
