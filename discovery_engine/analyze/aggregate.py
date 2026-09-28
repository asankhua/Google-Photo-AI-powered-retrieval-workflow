"""
Aggregate classified feedback into a comparison + opportunity ranking.

Opportunity score per failure mode = frequency (count of retrieval-related items)
weighted by average severity. Ranks where the biggest retrieval opportunity lives,
backed by real evidence.
"""
from collections import Counter, defaultdict
from typing import List, Dict, Any


def summarize(classified: List[Dict[str, Any]]) -> Dict[str, Any]:
    retrieval = [c for c in classified if c.get("is_retrieval_related")]
    total = len(classified)
    n = len(retrieval)

    mode_counts = Counter(c.get("failure_mode", "NOT_RETRIEVAL") for c in retrieval)
    mode_sev = defaultdict(list)
    for c in retrieval:
        mode_sev[c.get("failure_mode")].append(c.get("severity", 3) or 3)

    remembered, forgotten = Counter(), Counter()
    photo_types = Counter()          # Q1
    query_strategies = Counter()     # Q4
    query_examples = []              # Q4
    photo_by_mode = defaultdict(Counter)
    examples = defaultdict(list)

    for c in retrieval:
        for cue in c.get("remembered_cues", []) or []:
            remembered[cue] += 1
        for cue in c.get("forgotten_cues", []) or []:
            forgotten[cue] += 1
        photo_types[c.get("photo_type", "unknown")] += 1
        for q in c.get("query_strategies", []) or []:
            query_strategies[q] += 1
        if c.get("query_text"):
            query_examples.append(c["query_text"])
        mode = c.get("failure_mode")
        photo_by_mode[mode][c.get("photo_type", "unknown")] += 1
        if c.get("evidence_quote"):
            examples[mode].append(
                {"quote": c["evidence_quote"], "source": c.get("source", "")}
            )

    opportunities = []
    for mode, count in mode_counts.items():
        if mode == "NOT_RETRIEVAL":
            continue
        avg_sev = sum(mode_sev[mode]) / len(mode_sev[mode]) if mode_sev[mode] else 0
        opportunities.append(
            {
                "failure_mode": mode,
                "count": count,
                "share_pct": round(100 * count / n, 1) if n else 0,
                "avg_severity": round(avg_sev, 2),
                "opportunity_score": round(count * avg_sev, 1),
                "top_photo_types": photo_by_mode[mode].most_common(3),
                "examples": examples[mode][:5],
            }
        )
    opportunities.sort(key=lambda x: x["opportunity_score"], reverse=True)

    return {
        "total_items": total,
        "retrieval_related": n,
        "retrieval_share_pct": round(100 * n / total, 1) if total else 0,
        # Q1: kinds of photos users struggle to retrieve
        "photo_types_top": photo_types.most_common(),
        # Q2 / Q3: what users remember vs forget
        "remembered_cues_top": remembered.most_common(),
        "forgotten_cues_top": forgotten.most_common(),
        # Q4: how users formulate searches with incomplete memory
        "query_strategies_top": query_strategies.most_common(),
        "query_examples": query_examples[:15],
        "opportunities_ranked": opportunities,
    }
