"""
Failure-mode taxonomy for photo retrieval.

The analytical backbone of the discovery engine. Every piece of user feedback is
classified against the retrieval funnel so we can COMPARE distinct failure modes
with evidence (not just summarize sentiment).

Retrieval succeeds only if a user passes ALL FOUR stages:
    Expression -> Understanding -> Evaluation -> Refinement
"""

FAILURE_MODES = {
    "EXPRESSION": (
        "User cannot put the memory into words the system accepts. They remember "
        "associative/episodic cues (a feeling, who they were with, why they took it, "
        "'right before we left') but not searchable keywords, dates, places, or objects."
    ),
    "UNDERSTANDING": (
        "User gives a clue but Google Photos cannot interpret it into candidates "
        "(no OCR/semantic/geo/face match; natural-language query not understood; "
        "search returns nothing or irrelevant results)."
    ),
    "EVALUATION": (
        "Results exist but the user cannot recognize or pick the right one — too many "
        "results, poor previews, no way to tell candidates apart, scrolling fatigue."
    ),
    "REFINEMENT": (
        "A failed/partial search dead-ends. The user cannot iterate, narrow, combine "
        "clues, or recover — so they abandon."
    ),
    "NOT_RETRIEVAL": (
        "Feedback is unrelated to retrieving a remembered photo (e.g., storage cost, "
        "sync bugs, UI complaints, backup). Excluded from opportunity ranking."
    ),
}

PHOTO_TYPES = [
    "screenshot", "document_or_receipt", "medicine_or_health", "scenery_or_place",
    "person_or_group", "event_or_occasion", "food_or_menu", "product_or_shopping",
    "note_or_whiteboard", "pet_or_animal", "other", "unknown",
]

# Cue vocabulary — what people DO vs DON'T remember. Quantifies the
# memory-index schema mismatch (episodic cues remembered vs literal cues indexed).
CUE_TYPES = [
    "time_absolute",      # exact date/year (indexable)
    "time_relative",      # "a few years ago", "around my birthday" (fuzzy)
    "place",              # city/venue (sometimes indexable via GPS)
    "people_present",     # who was with them (indexable via faces, if tagged)
    "event_occasion",     # trip, wedding, festival (episodic)
    "activity",           # what they were doing (episodic)
    "emotion_feeling",    # the mood/why it mattered (NOT indexed)
    "visual_content",     # objects/colors visible in the photo (indexable)
    "text_in_image",      # words/labels in the photo (OCR-indexable)
    "adjacent_context",   # what happened just before/after (episodic)
]

# Q4 — how users formulate a search when memory is incomplete.
QUERY_STRATEGIES = [
    "keyword",             # typed literal words (e.g. "cafe", "medicine")
    "natural_language",    # a descriptive phrase / question
    "person_or_face",      # searched by a person
    "place_or_map",        # searched by location
    "date_or_timeline",    # navigated by date / scrolled the timeline
    "browse_scroll",       # gave up on search, scrolled manually
    "album_or_folder",     # looked in albums/folders
    "asked_someone",       # asked family/friend who had the photo
    "gave_up",             # abandoned without a strategy
    "unknown",             # not stated in the feedback
]

SEVERITY_SCALE = {
    1: "minor annoyance; user eventually succeeded",
    2: "notable friction; slow or frustrating success",
    3: "user failed this time but has workarounds",
    4: "user failed and it mattered (emotional/important photo)",
    5: "repeated failure; user gave up / lost the memory",
}

ALL_FAILURE_MODES = list(FAILURE_MODES.keys())
