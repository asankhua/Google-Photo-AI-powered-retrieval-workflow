# Part 6: AI-simulated MVP test sessions (n=3; not real participants)

**What's simulated and what's real.**
- **Simulated (gpt-oss-120b):** each parent's choices (what to type, which chip or photo to tap) and their interview answers.
- **Real:** every screen the persona saw. Results, ranks, counts, found or wrong pick, and taps were computed by the MVP's own code (`mvp/library.py`, `mvp/agent.py`) on the 571-item synthetic library.

**Harness.** `run_tests.py` runs the harness; the sessions are in `T1.md`–`T3.md` and the data in `results.json`.
- Each persona did 2 tasks from the "You are Priya" panel in both conditions.
- Order alternated: T1 and T3 started in standard search, T2 in Moment Finder.
- Each condition ran in a fresh context, so the answer found in one couldn't carry over to the other.
- Each attempt had a 10-step limit. Time was not simulated.

| Tester | Persona |
|---|---|
| T1 | Neha, 32, Gurugram, HR manager. Types full sentences |
| T2 | Vikram, 36, Chennai, bank operations. Prefers tapping; sceptical of AI |
| T3 | Ananya, 29, Kolkata, designer. Privacy-conscious |

## Results
| Tester | Task | Standard: queries → outcome | Moment Finder: queries / taps → outcome | Right photo's rank in MF (first search) | Wrong picks | Would use (1–5) |
|---|---|---|---|---|---|---|
| T1 | First steps | "Arjun first steps" (395), "…2023", "first steps" (26; target #19) → **gave up** | "first steps in-laws week before move" + "which one?" (1 tap) → **found** | #1 | 0 | 5 |
| T1 | Cake smash | "Arjun first birthday cake smash home" (395; target #314), 5 pages, 2 more searches → **gave up** | same words, 0 taps → **found** | #1 | 0 | |
| T2 | First haircut | "first haircut" (target #4) → **found**, 2 taps | "first haircut mother holding him" → target #1, but tapped **Amma** (then Newborn), photo filtered out → **gave up** | #1 | 0 | 2 |
| T2 | Daycare | "first day daycare rain" (target #8) → **found**, 2 taps | "first day at daycare crying gate rain", 0 taps → **found** | #1 | 0 | |
| T3 | Daycare | "first day at daycare rain gate" (target #16) → **found**, 2 taps | "first day daycare gate rain", 0 taps → **found** | #1 | 0 | 4 |
| T3 | First steps | "first steps" (26; target #19), "Amma", "first steps 2023-03" → **gave up** | "first steps", 0 taps → **found** | #1 | 0 | |

**Summary.**

| Measure | Standard search | Moment Finder |
|---|---|---|
| Found | 3 of 6 | 5 of 6 |
| Right photo in the top 3 after the first search | n/a | 6 of 6 (always #1) |
| Taps when found | median 2 | median 0 |
| Wrong picks | 0 | 0 |
| Saved as milestone | n/a | 0 of 5 (all three *said* it would help) |
| Chips used | n/a | 1 of 6 attempts, and it hurt |
| "Which one?" used | n/a | 1 of 6 |
| Would use | 5, 2, 4 | |

## What this tells us (read the caveats first)
1. **Only one of the three standard-search failures is solid.** In both "first steps" failures (T1, T3) the right photo was **on the first page (#19)** with an unmistakable caption, and the persona missed it. That's an LLM attention artefact, so it inflates Moment Finder's lead. The solid failure is the cake smash: typing the child's name returns all 395 photos, newest first, and the target is at #314.
2. **Standard search did well when parents left out the name.** "first haircut" (#4) and "first day daycare rain" (#8, #16) found the photo in 2 taps. This matches the interview finding that parents type an event word, not the name alone (H2). So Moment Finder's advantage is concentrated where the name is typed (it floods) or where look-alikes compete (the party vs the smash at home). Slide 8's "#169–#349" figure holds only for queries that include the name; say so.
3. **Moment Finder's gain came from ranking, not chips.** The typed words plus moment grouping put the target at #1 in 6 of 6 cases; the chips were tapped once. H1 ("the age chip is the key value") is **not supported** here.
4. **A wrong chip silently removes the right photo (T2).** In Tamil, "Amma" means mother, but the library labels Amma as the mother-in-law. T2 tapped it, the #1 photo vanished, and nothing said so. That's the same "a miss gives no direction" problem, inside the MVP. **Fix:** name relationships, not nicknames ("Rahul's mother"), and warn when a chip hides the current top match ("this hides your best match").
5. **Privacy discomfort with the people and home chips: 3 of 3.** They want a clear off switch and to know whether face data leaves the phone. This supports risk 4 on slide 10.
6. **Say–do gap on milestones:** 3 of 3 said "save as milestone" would help; 0 of 5 found photos were saved.
7. **T2 preferred standard search** (2 of 5) because it was faster for a keyword he knew, and the filters were "extra work". That's the disconfirming voice to keep.

## Limits
- The answers are generated, and the personas lean agreeable (T1 gave straight 5s).
- The captions stand in for thumbnails, which makes scanning easier than on a phone, except where the model misses a visible match (point 1).
- The library is synthetic.
- No time was measured.
- The first run of T2 and T3 was discarded because of a harness bug (nested JSON actions were rejected); those two were rerun in full.

**These results don't replace ≥3 real parent sessions** (`06_mvp_testing.md` §1a).
