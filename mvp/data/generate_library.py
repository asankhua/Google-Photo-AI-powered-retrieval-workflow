"""
Generates the synthetic sample library for the Moment Finder MVP.

One child (Arjun, born 10 Feb 2021) photographed by his mother Priya over ~4 years:
two homes, grandparent visits, festivals, birthdays, a Goa trip, daycare/preschool,
bursts of every milestone, and realistic clutter (WhatsApp forwards, screenshots).
Look-alike decoys are planted next to each test-task target.

Deterministic (fixed seed). Run:  python data/generate_library.py
Writes data/library.json.
"""
import json
import os
import random
from datetime import date, timedelta

RNG = random.Random(7)
BIRTH = date(2021, 2, 10)
MOVE = date(2023, 4, 1)          # old flat until 31 Mar 2023, new flat from 1 Apr 2023
END = date(2025, 6, 30)

OLD, NEW = "Old flat, Koramangala", "New flat, HSR Layout"

# Visits: (person list, start, end)
VISITS = [
    (["Nani"], date(2021, 2, 10), date(2021, 4, 15)),
    (["Amma", "Appa"], date(2021, 6, 1), date(2021, 6, 10)),
    (["Nani"], date(2021, 10, 20), date(2021, 11, 15)),
    (["Amma", "Appa"], date(2022, 3, 6), date(2022, 3, 20)),
    (["Amma", "Appa"], date(2022, 10, 20), date(2022, 10, 30)),
    (["Amma", "Appa"], date(2023, 8, 10), date(2023, 8, 20)),
    (["Nani"], date(2024, 1, 5), date(2024, 1, 20)),
    (["Amma", "Appa"], date(2024, 10, 28), date(2024, 11, 5)),
]

EVERYDAY = [  # (min_age_months, max_age_months, captions)
    (0, 3, ["Tiny Arjun sleeping in the crib", "Arjun wrapped in a blue blanket",
            "Arjun yawning on the bed", "Arjun's tiny fingers holding Priya's thumb"]),
    (3, 10, ["Arjun sitting up with his rattle", "Arjun crawling on the rug",
             "Arjun eating mashed banana, very messy", "Arjun splashing in the bath tub",
             "Arjun chewing a teething ring"]),
    (10, 16, ["Arjun standing, holding on to the sofa", "Arjun playing with stacking blocks",
              "Arjun in his high chair with idli", "Arjun pulling himself up at the window",
              "Arjun cruising along the bed edge"]),
    (16, 30, ["Arjun running in the park", "Arjun going down the slide",
              "Arjun scribbling with crayons", "Arjun on the swing",
              "Arjun walking to the gate holding Rahul's hand"]),
    (30, 99, ["Arjun riding his tricycle", "Arjun drawing a rocket",
              "Arjun building a Lego tower", "Arjun reading a picture book",
              "Arjun playing cricket in the corridor"]),
]

# Machine-visible tags: what a vision model could plausibly detect. Search and ranking
# use ONLY tags, auto-detectable events (birthdays, festivals, trips), location, people
# and dates, never the caption. "First steps" or "first haircut" are not in the metadata;
# the app has to get there from age, home and people.
TAGS = {
    "Tiny Arjun sleeping in the crib": ["baby", "crib", "sleeping"],
    "Arjun wrapped in a blue blanket": ["baby", "blanket"],
    "Arjun yawning on the bed": ["baby", "bed"],
    "Arjun's tiny fingers holding Priya's thumb": ["baby", "hand", "close-up"],
    "Arjun sitting up with his rattle": ["baby", "toy", "floor"],
    "Arjun crawling on the rug": ["baby", "rug", "floor", "crawling"],
    "Arjun eating mashed banana, very messy": ["baby", "food", "high chair"],
    "Arjun splashing in the bath tub": ["baby", "bath", "water"],
    "Arjun chewing a teething ring": ["baby", "toy"],
    "Arjun standing, holding on to the sofa": ["baby", "sofa", "standing", "living room"],
    "Arjun playing with stacking blocks": ["baby", "toy", "blocks", "floor"],
    "Arjun in his high chair with idli": ["baby", "food", "high chair"],
    "Arjun pulling himself up at the window": ["baby", "window", "standing"],
    "Arjun cruising along the bed edge": ["baby", "bed", "standing"],
    "Arjun running in the park": ["child", "park", "grass", "running"],
    "Arjun going down the slide": ["child", "park", "slide", "playground"],
    "Arjun scribbling with crayons": ["child", "crayons", "paper", "drawing"],
    "Arjun on the swing": ["child", "park", "swing", "playground"],
    "Arjun walking to the gate holding Rahul's hand": ["child", "man", "walking", "gate"],
    "Arjun riding his tricycle": ["child", "tricycle", "outdoor"],
    "Arjun drawing a rocket": ["child", "drawing", "paper"],
    "Arjun building a Lego tower": ["child", "toy", "lego"],
    "Arjun reading a picture book": ["child", "book"],
    "Arjun playing cricket in the corridor": ["child", "cricket", "bat", "corridor"],
}

items = []
_counter = [0]


def age_months(d):
    return (d - BIRTH).days / 30.44


def residence(d):
    return OLD if d < MOVE else NEW


def visitors(d):
    out = []
    for people, s, e in VISITS:
        if s <= d <= e:
            out += people
    return out


def add(d, caption, people, event=None, location=None, n=1, source="camera",
        face_miss_rate=None, hour=None, tags=None):
    """Add one photo or a burst of n frames. Returns the burst id."""
    _counter[0] += 1
    burst = f"b{_counter[0]:04d}"
    if face_miss_rate is None:
        # Baby faces group poorly: under 9 months, Arjun is often not recognised.
        face_miss_rate = 0.4 if age_months(d) < 9 else 0.05
    hour = hour if hour is not None else RNG.choice([8, 10, 12, 17, 18, 19])
    for i in range(n):
        ppl = list(people)
        if "Arjun" in ppl and source == "camera" and RNG.random() < face_miss_rate:
            ppl.remove("Arjun")  # face not grouped for this frame
        items.append({
            "id": f"{burst}-{i + 1}",
            "burst_id": burst,
            "best_frame": i == 0,
            "date": d.isoformat(),
            "time": f"{hour:02d}:{RNG.randint(0, 59):02d}",
            "caption": caption,          # what a person sees in the photo (display only)
            "tags": tags if tags is not None else TAGS.get(caption, []),  # machine-visible labels
            "people": ppl,
            "event": event,
            "location": location or residence(d),
            "residence": residence(d),
            "source": source,
        })
    return burst


# ---------- everyday photos: ~1 per week, more during visits ----------
d = BIRTH + timedelta(days=3)
while d <= END:
    a = age_months(d)
    caps = next(c for lo, hi, c in EVERYDAY if lo <= a < hi)
    ppl = ["Arjun"] + RNG.sample(["Priya", "Rahul"], k=RNG.choice([0, 0, 1]))
    vis = visitors(d)
    if vis:
        ppl += vis
    loc = "Neighbourhood park" if "park" in caps[0] and RNG.random() < 0.5 else None
    add(d, RNG.choice(caps), ppl, location=loc, n=RNG.choice([1, 1, 1, 2]))
    if vis:  # extra photos with grandparents
        add(d + timedelta(days=1), f"Arjun playing with {' and '.join(vis)}",
            ["Arjun"] + vis, n=RNG.choice([1, 2]),
            tags=["baby" if age_months(d) < 16 else "child", "woman" if "Nani" in vis or "Amma" in vis else "man", "toy"])
    d += timedelta(days=RNG.choice([6, 7, 8]))

# ---------- events, milestones, TARGETS and DECOYS ----------
add(date(2021, 2, 10), "Newborn Arjun in the hospital, first photo", ["Arjun", "Priya", "Rahul"],
    location="Cloudnine Hospital", n=6,
    tags=["baby", "hospital", "bed"])
add(date(2021, 3, 10), "Arjun's naming ceremony, cradle decorated with flowers",
    ["Arjun", "Priya", "Rahul", "Nani", "Amma", "Appa"], n=5, tags=["baby", "cradle", "flowers", "crowd"])

# TARGET T2: first haircut, Nani holding him, near Diwali 2021, old flat
add(date(2021, 10, 30), "Nani holding Arjun during his first haircut, hair on the floor",
    ["Arjun", "Nani", "Priya"], n=5, face_miss_rate=0.3,
    tags=["baby", "hair", "scissors", "woman", "floor"])
add(date(2021, 11, 4), "Arjun in a tiny kurta with diyas", ["Arjun", "Nani", "Priya", "Rahul"],
    event="Diwali 2021", n=4, hour=19,
    tags=["baby", "diya", "lights", "night"])
# decoy: second haircut at salon
add(date(2022, 6, 15), "Arjun at the kids' salon getting a haircut", ["Arjun", "Rahul"],
    location="Kids salon, Koramangala", n=3,
    tags=["child", "hair", "salon", "chair", "mirror"])

# TARGET T4: first birthday cake smash at home (not the party at the hall)
add(date(2022, 2, 10), "Arjun smashing his first birthday cake at home, cream all over his face",
    ["Arjun", "Priya", "Rahul"], event="Birthday", n=7, hour=18,
    tags=["baby", "cake", "cream", "messy", "high chair"])
# decoy: first birthday party at banquet hall
add(date(2022, 2, 13), "Arjun's first birthday party at the banquet hall, big cake and balloons",
    ["Arjun", "Priya", "Rahul", "Amma", "Appa", "Nani", "Meera"], event="Birthday",
    location="Banquet hall, Koramangala", n=8, hour=19,
    tags=["baby", "cake", "balloons", "crowd", "stage"])

# TARGET T1: first steps, old flat, Amma & Appa visiting
add(date(2022, 3, 12), "Arjun takes his first wobbly steps towards Amma in the living room",
    ["Arjun", "Amma", "Appa"], n=8, hour=17,
    tags=["baby", "walking", "woman", "living room"])
# decoys around first steps
add(date(2022, 3, 15), "Arjun standing, holding Appa's hands", ["Arjun", "Appa", "Amma"], n=3,
    tags=["baby", "standing", "man", "living room"])
add(date(2022, 4, 20), "Arjun walking in the park holding Rahul's finger", ["Arjun", "Rahul"],
    location="Neighbourhood park", n=3,
    tags=["baby", "walking", "park", "man", "grass"])
add(date(2022, 5, 8), "Arjun walking on his own at the mall", ["Arjun", "Priya"],
    location="Forum Mall", n=2, tags=["baby", "walking", "mall"])

add(date(2022, 10, 24), "Arjun lighting diyas with Amma and Appa", ["Arjun", "Amma", "Appa", "Priya"],
    event="Diwali 2022", n=5, hour=19,
    tags=["child", "diya", "lights", "night"])
for i, cap in enumerate(["Arjun on the beach at Palolem", "Arjun building a sandcastle",
                         "Arjun asleep on the flight home"]):
    add(date(2022, 12, 10) + timedelta(days=i * 2), cap, ["Arjun", "Priya", "Rahul"],
        event="Trip to Goa", location="Palolem, Goa", n=4,
        tags=["child", "beach", "sand", "sea"] if i < 2 else ["child", "sleeping", "airplane"])
add(date(2023, 2, 10), "Arjun's second birthday, dinosaur cake at home", ["Arjun", "Priya", "Rahul", "Meera"],
    event="Birthday", n=5, hour=18,
    tags=["child", "cake", "candles", "dinosaur"])
add(date(2023, 3, 30), "Packing boxes at the old flat, Arjun hiding in a box", ["Arjun", "Rahul"],
    location=OLD, n=3, tags=["child", "boxes", "cardboard"])
add(date(2023, 4, 1), "Arjun exploring the empty new flat", ["Arjun", "Priya"], 
    location=NEW, n=3, tags=["child", "empty room", "floor"])

# TARGET T3: first day at daycare, crying at the gate, rainy week
add(date(2023, 6, 5), "Arjun crying at the daycare gate on his first day, umbrella in the rain",
    ["Arjun", "Priya"], location="Little Sprouts Daycare, HSR", n=6, hour=9,
    tags=["child", "crying", "umbrella", "rain", "gate", "woman"])
# decoys
add(date(2023, 6, 20), "Arjun happily playing with sand at daycare", ["Arjun"],
    location="Little Sprouts Daycare, HSR", n=2, tags=["child", "sand", "playground", "smiling"])
add(date(2023, 12, 15), "Arjun on stage at the daycare annual day", ["Arjun"],
    location="Little Sprouts Daycare, HSR", n=5,
    tags=["child", "stage", "costume", "crowd"])
add(date(2024, 6, 10), "Arjun in his new school uniform, first day at preschool", ["Arjun", "Priya", "Rahul"],
    location="Green Leaf Preschool, HSR", n=5, hour=8,
    tags=["child", "uniform", "school bag", "gate"])

add(date(2023, 11, 12), "Arjun with sparklers on the balcony", ["Arjun", "Priya", "Rahul"],
    event="Diwali 2023", n=4, hour=20,
    tags=["child", "sparklers", "balcony", "night"])
add(date(2024, 2, 10), "Arjun's third birthday, car-shaped cake", ["Arjun", "Priya", "Rahul", "Meera"],
    event="Birthday", n=5, hour=18,
    tags=["child", "cake", "candles", "car"])
add(date(2024, 11, 1), "Arjun in a kurta with Amma and Appa, rangoli at the door",
    ["Arjun", "Amma", "Appa", "Priya"], event="Diwali 2024", n=5, hour=19,
    tags=["child", "rangoli", "kurta", "door"])
add(date(2025, 2, 10), "Arjun's fourth birthday, rocket cake", ["Arjun", "Priya", "Rahul"],
    event="Birthday", n=5, hour=18,
    tags=["child", "cake", "candles", "rocket"])

# ---------- clutter: WhatsApp forwards and screenshots ----------
FORWARDS = ["Good morning flowers forward", "Festival greetings forward", "School circular: holiday list",
            "Birthday wishes image forward", "Funny meme forward", "Traffic alert forward"]
SCREENS = ["Screenshot: vaccination schedule", "Screenshot: UPI payment receipt",
           "Screenshot: daycare fee receipt", "Screenshot: flight booking", "Screenshot: recipe"]
d = BIRTH + timedelta(days=10)
while d <= END:
    add(d, RNG.choice(FORWARDS), [], source="whatsapp", tags=["text", "graphic"])
    if RNG.random() < 0.6:
        add(d + timedelta(days=3), RNG.choice(SCREENS), [], source="screenshot", tags=["text", "screenshot"])
    d += timedelta(days=RNG.choice([20, 25, 30]))
# a forward that IS a real photo of Arjun (sent by Nani)
add(date(2021, 11, 2), "Photo of Arjun and Nani sent by Nani on WhatsApp", ["Arjun", "Nani"],
    source="whatsapp", face_miss_rate=0, tags=["baby", "woman"])

items.sort(key=lambda x: (x["date"], x["time"], x["id"]))
out = os.path.join(os.path.dirname(__file__), "library.json")
with open(out, "w") as f:
    json.dump({"child": {"name": "Arjun", "birthdate": BIRTH.isoformat()},
               "residences": [{"name": OLD, "short": "Old flat", "until": (MOVE - timedelta(days=1)).isoformat()},
                              {"name": NEW, "short": "New flat", "from": MOVE.isoformat()}],
               "people": {"Priya": "you", "Rahul": "your husband", "Amma": "mother-in-law",
                          "Appa": "father-in-law", "Nani": "your mother", "Meera": "cousin"},
               "items": items}, f, indent=1)
print(f"wrote {len(items)} items to {out}")
