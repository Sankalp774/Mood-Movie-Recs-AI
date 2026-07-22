# Mood Movie Recs

> **Academic stage:** 3rd year → product rebuild · **Created:** Feb 2025 · **Rebuilt:** 2026  
> **Learning focus:** Recommender systems · content-based filtering · TF-IDF · cosine similarity · mood UX · FastAPI · full-stack web · GitHub Pages hosting

**Fully hosted mood-first movie discovery website** with a real content-based AI recommender.

🎬 **Live static site (GitHub Pages):**  
`https://sankalp774.github.io/Mood-Movie-Recs-AI/`  
*(enable Pages after push — Actions workflow included)*

---

## Product features (shipped)

| Feature | Implementation |
|---------|----------------|
| **Mood-based discovery** | 12 mood channels (cozy, tense, romantic, mind-bending, …) |
| **Free-text vibe search** | “rainy night heist”, “hopeful sci-fi”, etc. |
| **AI ranking** | TF-IDF over title + genres + mood tags + overview → cosine similarity |
| **Personalization** | Likes / high ratings inject profile terms into the query |
| **Discovery feed** | High-rated catalog slice |
| **Watchlist + likes + 1–5 ratings** | Browser `localStorage` |
| **Movie detail modal** | Poster, genres, moods, overview, actions |
| **Dual host modes** | FastAPI API **or** pure on-device JS engine for static hosting |
| **Eval harness** | Self-retrieval Hit@K via Python |

Vision items like live watch-parties / social clubs remain **roadmap** — this rebuild prioritizes a polished, honest, deployable recommender product.

---

## Architecture

```text
┌─────────────────────────────────────────────┐
│  web/  cinematic SPA (HTML/CSS/JS)          │
│   ├─ GitHub Pages (static TF-IDF engine)    │
│   └─ or served by FastAPI at /              │
└──────────────────┬──────────────────────────┘
                   │ /api/* (optional)
┌──────────────────▼──────────────────────────┐
│  backend/app                                │
│   recommender.py  ·  main.py (FastAPI)      │
│   data/movies.json  (40 curated films)      │
└─────────────────────────────────────────────┘
```

---

## Quickstart — full stack (recommended locally)

```bash
git clone https://github.com/Sankalp774/Mood-Movie-Recs-AI.git
cd Mood-Movie-Recs-AI
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# from repo root
uvicorn backend.app.main:app --reload --port 8000
```

Open **http://localhost:8000**

API docs: http://localhost:8000/docs

### API examples

```bash
curl -s http://localhost:8000/api/health | jq
curl -s -X POST http://localhost:8000/api/recommend \
  -H 'Content-Type: application/json' \
  -d '{"mood":"cozy","top_k":5}' | jq
curl -s http://localhost:8000/api/eval | jq
```

### Docker

```bash
docker build -t mood-movie-recs .
docker run --rm -p 8000:8000 mood-movie-recs
```

### Offline eval

```bash
python -m eval.run_eval
```

---

## Hosting

### A) GitHub Pages (static, free, fully hosted)

1. Push this repo to GitHub  
2. **Settings → Pages → Source: GitHub Actions**  
3. Workflow `.github/workflows/pages.yml` deploys `web/`  
4. Site: `https://sankalp774.github.io/Mood-Movie-Recs-AI/`  

Static mode runs the **same ranking idea in JavaScript** (no Python server required).

### B) Single container (API + UI)

Deploy the Docker image to Render / Railway / Fly.io — one service on port 8000.

---

## Project layout

```text
web/                    # Hosted website
  index.html
  assets/styles.css
  assets/app.js
  assets/movies.json
backend/
  app/main.py           # FastAPI + static mount
  app/recommender.py    # sklearn TF-IDF engine
  data/movies.json
eval/run_eval.py
Dockerfile
.github/workflows/pages.yml
```

---

## What I learned

| Topic | In this project |
|-------|-----------------|
| Recommender systems | Content-based design with mood features |
| IR / ML | TF-IDF, cosine similarity, score blending |
| Full-stack | FastAPI + modern SPA UX |
| Progressive hosting | Static fallback when API absent |
| Product honesty | Ship core loops; mark social features as roadmap |

---

## Author

**Sankalp Sahu** · Applied AI portfolio  

License: MIT
