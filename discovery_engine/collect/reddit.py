"""Collect Reddit discussions about finding old photos in Google Photos.

Reddit is a far richer channel than app-store reviews: threads contain the full
"I remembered it was from my sister's wedding but search showed nothing, so I
scrolled for 20 minutes" narratives — and the *comments* often hold the
workaround/give-up story. We pull the post body AND the top comments.

Uses PRAW (needs a free Reddit "script" app). Set in .env:
  REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
Create one (30 seconds, free): https://www.reddit.com/prefs/apps  -> "create app"
  -> type: script  -> redirect uri: http://localhost:8080
"""
import os
from typing import List, Dict

SUBREDDITS = ["googlephotos", "google", "android", "GooglePixel", "photography"]
QUERIES = [
    "can't find photo", "find old picture", "search doesn't work",
    "remember a photo", "locate old photo", "photos search bad",
    "looking for a photo", "how to find", "scrolling forever",
]
# how many top comments to append per post (adds the workaround/give-up story)
COMMENTS_PER_POST = 3


def _require_creds():
    missing = [k for k in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET")
               if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            "Missing Reddit creds: " + ", ".join(missing) + ". "
            "Create a free 'script' app at https://www.reddit.com/prefs/apps "
            "and add the values to discovery_engine/.env"
        )


def collect(limit_per_query: int = 25) -> List[Dict]:
    import praw

    _require_creds()
    reddit = praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ.get("REDDIT_USER_AGENT", "gphotos-discovery/0.1"),
        check_for_async=False,
    )
    reddit.read_only = True

    out, seen = [], set()
    for sub in SUBREDDITS:
        for q in QUERIES:
            try:
                results = reddit.subreddit(sub).search(
                    q, limit=limit_per_query, sort="relevance"
                )
                for post in results:
                    if post.id in seen:
                        continue
                    seen.add(post.id)

                    parts = [post.title, post.selftext or ""]
                    try:  # top few comments carry the resolution/give-up story
                        post.comments.replace_more(limit=0)
                        top = sorted(
                            post.comments[:10],
                            key=lambda c: getattr(c, "score", 0),
                            reverse=True,
                        )[:COMMENTS_PER_POST]
                        parts += [getattr(c, "body", "") for c in top]
                    except Exception:
                        pass

                    text = " ".join(p.strip() for p in parts if p and p.strip())
                    if not text:
                        continue
                    out.append(
                        {
                            "source": f"reddit/r/{sub}",
                            "url": f"https://reddit.com{post.permalink}",
                            "date": str(post.created_utc),
                            "rating": None,
                            "text": text[:4000],
                            "upvotes": post.score,
                            "num_comments": post.num_comments,
                        }
                    )
            except Exception:
                continue
    # richest (most-discussed) threads first
    out.sort(key=lambda r: r.get("num_comments") or 0, reverse=True)
    return out
