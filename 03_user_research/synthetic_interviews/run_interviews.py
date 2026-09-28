"""
AI-simulated interviews (NOT real participants).

Each simulated parent is an LLM persona built from the target segment plus ONE real parent
review from the discovery engine's parent run (data/parents_classified.json). The interviewer
asks the fixed questions from ../interview_guide.md, in order, without leading. Transcripts are
saved as P#.md and coded afterwards against the guide's coding sheet.

Limits (state these wherever results are used): answers are generated, not observed; the
"live task" is narrated, not performed on a real library; LLM personas tend to be articulate
and agreeable. Personas P3 and P4 are built to push against the hypotheses.

Run:  python run_interviews.py            (all personas)
      python run_interviews.py P3         (one persona)
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(HERE, "..", "..", "discovery_engine")
sys.path.insert(0, ENGINE)
from _env import load_env  # noqa: E402

load_env(os.path.join(ENGINE, ".env"))
from groq import APIConnectionError, Groq, RateLimitError  # noqa: E402

MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
REVIEWS = json.load(open(os.path.join(ENGINE, "data", "parents_classified.json")))

# (id, seed review index, persona). The seed review is the persona's real lived experience.
PERSONAS = [
    ("P1", 104, "Meera, 34, Bengaluru, product manager. Daughter Anaya is 5. ~22k photos, "
                "Google One 200 GB. Loves face search and uses it for everything. Her daughter's "
                "school asked the class for a 'photo from when your child was a baby' by Friday."),
    ("P2", 125, "Rohit, 31, Pune, software engineer. Son Kabir is 3. ~18k photos, no Google One. "
                "Is the family's photo-taker. Has noticed the app mixes up his son and his nephew's "
                "faces. His mother asked on the family WhatsApp group for Kabir's first haircut photo."),
    ("P3", 151, "Sunita, 36, Mumbai, teacher. Daughter Isha is 6. ~30k photos, Google One. "
                "Is organised about dates: she usually knows the month because she links photos to "
                "school terms and birthdays, and month search mostly works for her. She sometimes "
                "finds things on the first try. Nobody asked her for a photo recently; she was "
                "looking for one herself for Isha's birthday slideshow."),
    ("P4", 21, "Arvind, 38, Delhi NCR, consultant. Son Vivaan is 4. ~15k photos on his own phone. "
               "His wife took most of the baby photos on her iPhone, and partner sharing only lets "
               "her share everything, so many early photos are not in his library at all. His "
               "father asked for a photo of Vivaan's first Holi."),
    ("P5", 76, "Farah, 33, Hyderabad, doctor. Twin boys Ayaan and Zayan, 4. ~35k photos, Google One. "
               "The app can't tell the twins apart. Her sister wants a photo of the twins' first day "
               "at playschool for a frame, by the weekend."),
    ("P6", 155, "Karthik, 29, Chennai, sales manager. Daughter Diya is 3. ~16k photos, Google One. "
                "Uses the new Gemini-style search and finds it slow and hit-or-miss. His wife wants "
                "the photo of Diya's cake smash at home (not the big party) to print for Diya's room."),
]

PERSONA_RULES = """You are playing a research participant in a 40-minute video call about finding old photos.
Stay in character. You are a real, busy parent, not a helper:
- Answer like speech: short, a bit messy, sometimes unsure. 1-4 sentences per answer.
- Only say what this person would plausibly know or do. Invent specific, believable details
  (dates you half-remember, what you typed, roughly how many results you saw) and keep them
  consistent through the call.
- Do not give product ideas unless asked. Do not agree with the interviewer to be nice.
- Your real experience with the app, in your own words from a review you once wrote:
  "{review}"
- Who you are: {persona}
During the search task, narrate what you type or tap and what you see on screen, step by step,
as it happens. You can succeed, fail, give up, or pick a close-enough photo, whatever is
realistic for you."""

QUESTIONS = [
    ("Warm-up", "Thanks for joining. Tell me about the last time you went looking for an older photo "
                "of your child. What happened? Who asked for it, and by when?"),
    ("Task", "Think of a specific moment of your child from when they were little that you'd find hard "
             "to locate. Don't describe it to me yet. Now please try to find it in Google Photos and "
             "think aloud: what are you typing or tapping, and what do you see?"),
    ("Task", "What are you seeing now?"),
    ("Task", "What would you do next?"),
    ("Task", "Keep going as you normally would. Tell me when you've found it, or when you'd stop."),
    ("Probe", "Now tell me: what was the moment you were looking for?"),
    ("Probe", "What did you remember about it before you started?"),
    ("Probe", "What did you not remember: the date, the month, the year, the place?"),
    ("Probe", "How did you know roughly when it was?"),
    ("Probe", "What made you type what you typed first? What did you expect to happen?"),
    ("Probe", "When it didn't work the way you wanted, what did you try next, and what made you stop?"),
    ("Wrap", "In the last month, how many times did you look for a specific older photo of your child? "
             "And out of those, how many times did the first search find it?"),
    ("Wrap", "When search doesn't find it, how do you usually end up getting the photo? Have you ever "
             "sent a photo that wasn't quite the one someone asked for?"),
    ("Wrap", "Could the photo be on someone else's phone, like your partner's?"),
    ("Wrap", "How did it feel when you couldn't find it? Anything else you'd like to add?"),
]


def interview(pid, seed_idx, persona, client):
    review = REVIEWS[seed_idx]["source_text"].strip()
    msgs = [{"role": "system", "content": PERSONA_RULES.format(review=review, persona=persona)}]
    turns = []
    for section, q in QUESTIONS:
        msgs.append({"role": "user", "content": q})
        for attempt in range(8):  # free tier: 8k tokens a minute, so wait and retry
            try:
                r = client.chat.completions.create(model=MODEL, messages=msgs, temperature=0.9,
                                                   max_completion_tokens=1500, reasoning_effort="low")
                break
            except (RateLimitError, APIConnectionError) as e:
                if "tokens per day" in str(e):
                    raise  # daily cap: waiting a few seconds won't help
                time.sleep(15 * (attempt + 1))
        else:
            raise RuntimeError("Groq kept failing; try again later")
        a = (r.choices[0].message.content or "").strip()
        msgs.append({"role": "assistant", "content": a})
        turns.append((section, q, a))
        print(f"  {pid} turn {len(turns)}/{len(QUESTIONS)}", flush=True)
    return review, turns


def save(pid, persona, seed_idx, review, turns):
    lines = [f"# {pid}: AI-simulated interview (not a real participant)", "",
             f"**Persona:** {persona}", "",
             f"**Seed (real parent review, parents_classified.json #{seed_idx}):** \"{review}\"", "",
             f"**Model:** {MODEL}, temperature 0.9 · questions from `interview_guide.md`", ""]
    last = None
    for section, q, a in turns:
        if section != last:
            lines += [f"## {section}", ""]
            last = section
        lines += [f"**Interviewer:** {q}", "", f"**{pid}:** {a}", ""]
    with open(os.path.join(HERE, f"{pid}.md"), "w") as f:
        f.write("\n".join(lines))


def main():
    only = set(sys.argv[1:])
    client = Groq(timeout=90, max_retries=2)
    for pid, seed_idx, persona in PERSONAS:
        if only and pid not in only:
            continue
        review, turns = interview(pid, seed_idx, persona, client)
        save(pid, persona, seed_idx, review, turns)
        print(f"{pid}: {len(turns)} turns saved")


if __name__ == "__main__":
    main()
