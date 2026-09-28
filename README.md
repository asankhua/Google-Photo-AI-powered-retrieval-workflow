# Google Photos retrieval: AI discovery engine + Moment Finder MVP

One Streamlit app with two pages:

| Page | What it is | Code |
|---|---|---|
| 🔎 **AI discovery engine** | **Ask the engine** any discovery question: it answers from 4,278 retrieval-related posts (Play Store, App Store, Hacker News, Stack Exchange, Google Photos Community) with themes and cited user quotes. Also: the funnel classification findings and a live classifier | `discovery_engine/` |
| 📷 **Moment Finder (MVP)** | Helps a parent find one milestone photo of their child when the child is in almost every photo: chips for age, home and who was there, plus optional "tell me what you remember" (Groq) | `mvp/` |

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp discovery_engine/.env.example discovery_engine/.env   # add your GROQ_API_KEY (free at console.groq.com)
streamlit run streamlit_app.py
```
Each app also runs on its own: `streamlit run discovery_engine/app.py` or `streamlit run mvp/app.py`.

## Deploy (Streamlit Community Cloud)
- Main file: `streamlit_app.py`
- Secrets: `GROQ_API_KEY = "..."`

Without a key the MVP still works (offline rules parser); the Ask tab and live classifier need it.

## Data
`discovery_engine/data/` holds the collected and classified posts and `corpus.json`, the search
index input. The raw store-review dump (`store_all.json`, 22 MB) is not committed; rebuild it and the
corpus with the collectors (see `discovery_engine/README.md`).
