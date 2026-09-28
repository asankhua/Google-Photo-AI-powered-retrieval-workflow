"""
Validation guardrail for the discovery engine (Part 1c requirement).

Runs the live classifier over a hand-labeled gold set covering every failure
mode + non-retrieval noise, then reports accuracy on the two fields that matter:
  - is_retrieval_related  (does it correctly filter noise?)
  - failure_mode          (does it locate the right funnel stage?)

This catches hallucinated categories and gives a defensible accuracy estimate.

Run:  GROQ_MODEL=openai/gpt-oss-20b python validate.py
"""
from _env import load_env
load_env()

from classify.classifier import classify_one

# Hand-labeled gold set. Each: (text, expected_is_retrieval, expected_failure_mode)
GOLD = [
    ("I know I have a photo of a small cafe from our Goa trip a few years ago but I "
     "can't remember the name and searching 'cafe' shows nothing useful. I gave up.",
     True, "UNDERSTANDING"),
    ("I remember the feeling of that evening but I have no words to type into search — "
     "I don't know what to even search for.",
     True, "EXPRESSION"),
    ("Search returns hundreds of nearly identical beach photos and I can't tell which "
     "one is the one I want from the tiny thumbnails.",
     True, "EVALUATION"),
    ("I searched 'birthday' and got nothing useful, and there's no way to narrow it or "
     "try a different angle — it just dead-ends.",
     True, "REFINEMENT"),
    ("I can't put into words what the picture looked like, only that it made me happy.",
     True, "EXPRESSION"),
    ("Typed the restaurant name but Photos doesn't understand it and shows unrelated "
     "shots — it can't read the sign in the picture.",
     True, "UNDERSTANDING"),
    ("Too many results after I search and the previews are so small I give up picking.",
     True, "EVALUATION"),
    ("Great app, love the auto-backup, works perfectly on my new phone!",
     False, "NOT_RETRIEVAL"),
    ("The app keeps crashing when I open it and drains my battery.",
     False, "NOT_RETRIEVAL"),
    ("Please add a dark mode and let me change the grid size.",
     False, "NOT_RETRIEVAL"),
    ("Uploading is super slow on my connection and eats all my storage.",
     False, "NOT_RETRIEVAL"),
    ("I'm trying to find a receipt I photographed last year for taxes but searching "
     "the store name finds nothing and I don't remember the date.",
     True, "UNDERSTANDING"),
]


def main():
    n = len(GOLD)
    rel_correct = 0
    mode_correct = 0
    mode_denom = 0  # only score failure_mode on truly-retrieval gold items
    rows = []
    for i, (text, exp_rel, exp_mode) in enumerate(GOLD, 1):
        try:
            r = classify_one(text)
        except Exception as e:
            rows.append((i, "ERROR", str(e)[:60], exp_rel, exp_mode))
            continue
        got_rel = bool(r.get("is_retrieval_related"))
        got_mode = r.get("failure_mode")
        if got_rel == exp_rel:
            rel_correct += 1
        if exp_rel:  # score stage only where a retrieval failure truly exists
            mode_denom += 1
            if got_mode == exp_mode:
                mode_correct += 1
        rows.append((i, got_rel, got_mode, exp_rel, exp_mode))
        print(f"  {i:>2}. rel exp={exp_rel!s:<5} got={got_rel!s:<5} | "
              f"mode exp={exp_mode:<14} got={str(got_mode):<14}")

    print("\n=== VALIDATION RESULTS ===")
    print(f"Gold items:                 {n}")
    print(f"is_retrieval_related acc:   {rel_correct}/{n} = {100*rel_correct/n:.0f}%")
    if mode_denom:
        print(f"failure_mode acc (retr.):   {mode_correct}/{mode_denom} = "
              f"{100*mode_correct/mode_denom:.0f}%")


if __name__ == "__main__":
    main()
