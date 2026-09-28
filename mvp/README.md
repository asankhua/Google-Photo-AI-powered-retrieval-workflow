# Moment Finder: MVP (Part 5)

A prototype that helps a parent find one milestone photo of their child when the child is
in almost every photo. Spec and reasoning: `../05_solution.md` §4–§5. Test plan: `../06_mvp_testing.md`.

**The problem it tests:** a search for "Arjun" returns all 395 photos of him. Priya remembers
*context* (how old he was, which flat, who was visiting), and search can't use that.

## What it does
1. **Flood diagnosis.** A search for the child, a milestone word, or anything returning too
   many or zero results says so ("Arjun is in 395 photos") and offers narrowing.
2. **Chips with live counts:**
   - **How old was he?** Life stage from his birthdate. Windows are wide and can be nudged
     Earlier / Later. This chip needs no faces and no location.
   - **Which home?** Old flat / new flat, by residence period.
   - **Who else was there?** People ranked by how often they appear with him in these results.
   - The chip group that splits the current results most evenly shows first.
3. **"Tell me what you remember" (optional, the only LLM step).** Groq turns
   "the week my in-laws stayed, before we moved" into the same filters. The output is
   validated against the library's own people, homes and stages. A festival name
   ("around Diwali") becomes a date anchor, and "not the party at the hall" becomes an exclusion.
4. **No dead ends.** If filters leave nothing, the least certain one is relaxed (people, then
   age window ±3 months, then home) and the app says which.
5. **Moments, not frames.** Photos are grouped by day and event, bursts collapsed, WhatsApp
   forwards and screenshots hidden. There are 6 cards, each explaining why it's shown, e.g.
   "Old flat · with Amma & Appa · 12 Mar 2022 · Arjun ~13 months · 8 photos". The count of
   frames where his face wasn't recognised is shown too.
6. **"Which one?"** When the top two moments are too close to call, one question built from
   what differs between them (place, people, visible content) settles it in a tap.
7. **Actions:** see all frames · days around this one · "This is it → Share to WhatsApp"
   (simulated) · **Save as milestone**, so the next search for that label is one tap.
8. **Time and taps** from search to found are shown when you pick a photo.
9. A **Standard search** tab shows today's behaviour (literal match, newest first, every frame,
   24 at a time) as the baseline.

## Honesty about the data
- `data/library.json` is **synthetic**: 571 items for one child over 4 years (454 camera,
  68 WhatsApp, 49 screenshots). A real library has 20,000+.
- Ranking uses **only metadata a phone could plausibly have**: vision tags ("cake",
  "umbrella", "walking"), auto-detected events (birthdays, Diwali, trips), location, face
  groups, dates. It **never reads the captions** shown on cards; they stand in for the image.
- "First steps" and "first haircut" are **not** labels in the data. The app gets there from
  age, home, people and visible content.
- Babies' faces are grouped poorly: under 9 months, 40% of frames miss Arjun's face tag.
- Each target has look-alike decoys: the salon haircut, the banquet-hall birthday party, other
  walking photos, the daycare annual day, and the first day at preschool.

## Verified (`python verify.py`)
Pass = the target moment is in the top 3. Standard search = where the target sits in today's
newest-first list.

| Task | Standard search | Typed context | Chips only | Chips + "which one?" |
|---|---|---|---|---|
| First steps | #293 of 395 | #1 of 24 | #1 of 24 | (not asked) |
| First haircut | #349 of 395 | #1 of 65 | #1 of 20 | (not asked) |
| First day at daycare | #169 of 395 | #1 of 454 | #1 of 36 | #1 |
| Cake smash at home | #314 of 395 | #1 of 66 | #2 of 66 | **#1** |

These results hold with both the Groq parser and the offline rules parser. On chips alone,
the banquet-hall party edges out the cake smash (both are birthdays in the same age
window). The app then asks **"Which one?"**, contrasting the two ("at Banquet hall ·
balloons, crowd" vs "at home · cream, high chair, messy"); one tap puts the cake smash at #1. This synthetic check
does not replace testing with real parents.

## Files
- `app.py`: Streamlit UI
- `library.py`: filters, live counts, chip order, no-dead-end relaxing, moments, ranking,
  explanations, baseline search (no LLM)
- `agent.py`: typed-context parser (Groq, with a rule-based fallback), validated
- `verify.py`: checks the 4 test tasks
- `data/generate_library.py`: builds the synthetic library deterministically (seed 7)
- `_env.py`: tiny `.env` loader

## Run
```bash
pip install -r requirements.txt
cp .env.example .env              # optional: add GROQ_API_KEY
python data/generate_library.py   # only if you change the generator
python verify.py                  # add --rules to force the offline parser
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
Push `mvp/` to a GitHub repo → share.streamlit.io → main file `app.py` → add `GROQ_API_KEY`
under *Secrets* (optional). Then put the public URL in `../deck/deck_outline.md`.
