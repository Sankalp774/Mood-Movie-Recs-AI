# Mood Movie Recs

> **Created:** (2025-02-28)  
> **Latest update:** OpenCV facial expression mood detection · match loading bar · 4:3 liquid-glass Netflix-style card stack (linked-list navigation) · TF-IDF recommender SPA + FastAPI  

**Academic stage:** 3rd year → product rebuild  
**Learning focus:** Recommender systems · OpenCV face/expression pipeline · full-stack UX · content-based IR

🎬 **Live static site (GitHub Pages):**  
`https://sankalp774.github.io/Mood-Movie-Recs-AI/`  
*(Settings → Pages → Source: GitHub Actions)*

---

## Changelog snapshot

| When | What |
|------|------|
| **(2025-02-28)** *created* | Original Mood Movie Recs concept / Android sketch era |
| **2026 rebuild** | Full hosted website + TF-IDF content recommender + 40-film catalog |
| **Latest update** | **OpenCV** Haar face detect + **FER+ ONNX** expression → mood; **match % bar** on recommend; **4:3 pop stack** with blur liquid glass + Netflix prev/next by match rank |

---

## Features (shipped)

| Feature | Detail |
|---------|--------|
| **OpenCV face mood** | Webcam frame → `/api/emotion` → face box + expression (happiness, sadness, …) → mapped mood |
| **Match loading bar** | Glass overlay animates 0→100% while scoring recommendations |
| **4:3 stack UI** | Results as a **linked-list** of cards; navigate like Netflix (← →); match % on each card |
| **Liquid glass chrome** | Backdrop blur + frosted panels behind stack & loader |
| **Mood chips + vibe search** | Manual mood or free-text query |
| **Personalization** | Likes / ratings reshape ranking |
| **Dual host** | FastAPI (OpenCV path) or static Pages (JS TF-IDF; face needs API) |

---

## Architecture

```text
Webcam ──▶ POST /api/emotion
              OpenCV Haar face detect
              FER+ ONNX expression
              mood + query_hint
                    │
                    ▼
           TF-IDF recommender ──▶ match bar ──▶ 4:3 glass stack (linked list)
```

---

## Quickstart (full stack — required for OpenCV face)

```bash
git clone https://github.com/Sankalp774/Mood-Movie-Recs-AI.git
cd Mood-Movie-Recs-AI
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# from repo root
uvicorn backend.app.main:app --reload --port 8000
```

Open **http://localhost:8000**

1. **Enable camera** → **Read expression** (downloads FER+ ONNX on first run)  
2. **Get recommendations** → watch match bar → swipe the stack  

### API

```bash
# health
curl -s localhost:8000/api/health | jq

# recommend
curl -s -X POST localhost:8000/api/recommend \
  -H 'Content-Type: application/json' \
  -d '{"mood":"cozy","top_k":5}' | jq

# emotion (base64 image / data URL)
curl -s -X POST localhost:8000/api/emotion \
  -H 'Content-Type: application/json' \
  -d '{"image":"data:image/jpeg;base64,..."}' | jq
```

### Docker

```bash
docker build -t mood-movie-recs .
docker run --rm -p 8000:8000 mood-movie-recs
```

### Eval

```bash
python -m eval.run_eval
```

---

## Project layout

```text
web/                      # Hosted SPA
  index.html
  assets/app.js           # match bar, stack, webcam, TF-IDF fallback
  assets/styles.css       # liquid glass + 4:3 stack
  assets/movies.json
backend/
  app/main.py             # FastAPI + static host
  app/recommender.py      # sklearn TF-IDF
  app/emotion.py          # OpenCV + FER+ ONNX
  data/movies.json
  models/                 # emotion-ferplus-8.onnx (auto-download)
eval/run_eval.py
.github/workflows/pages.yml
Dockerfile
```

---

## Expression → mood map

| Expression (FER+) | Mood for ranking |
|-------------------|------------------|
| happiness | whimsical |
| surprise | curious |
| sadness | emotional |
| anger | intense |
| fear | tense |
| disgust / contempt | dark |
| neutral | reflective |

---

## Notes

- **Camera + OpenCV** need the Python API (localhost/Docker). GitHub Pages static mode still runs the site + TF-IDF; face scan prompts you to run the API.  
- First emotion call downloads `emotion-ferplus-8.onnx` (~ONCE).  
- Posters via TMDB image CDN; catalog is curated demo data.  
- Watch parties / social clubs remain roadmap — not faked.

---

## Author

**Sankalp Sahu** · Applied AI portfolio  

**Created:** (2025-02-28) · **Latest update:** OpenCV facial expression + match bar + 4:3 glass stack  

License: MIT
