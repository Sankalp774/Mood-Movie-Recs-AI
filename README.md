# Mood Movie Recs — Content-Based Recommender

**Real ML recommender** (not a vaporware pitch):

- TF-IDF over title + genres + **mood tags** + overview  
- Cosine similarity ranking  
- Offline self-retrieval metrics (Hit@1 / Hit@3 / mean rank)  
- CLI + Gradio UI  
- Kotlin stubs kept as mobile exploration notes  

```text
user mood / query → vectorize → cosine sim → top-k movies
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# CLI
python -m recommender.app --mood "tense dark" --top-k 5

# Metrics
python -m recommender.app --eval

# UI
python -m recommender.app --ui
```

## Evaluation

`python -m recommender.app --eval` writes `metrics/eval_report.json`:

| Metric | Meaning |
|--------|---------|
| hit_at_1 | Fraction of movies retrieving themselves as #1 from overview |
| hit_at_3 | Self in top-3 |
| mean_rank | Average self-rank (lower is better) |

This is a **sanity/retrieval** check on a small catalog — not offline MovieLens SOTA.

## Layout

```text
recommender/
  engine.py
  app.py
  data/movies.csv
MainActivity.kt / Movie.kt   # Android skeleton (secondary)
metrics/                     # generated
```

## Interview narrative

> Built a content-based recommender with explicit mood features, measured ranking quality with self-retrieval metrics, and shipped CLI/UI — instead of claiming an unfinished “AI social movie network.”

## Next upgrades

- Collaborative filtering on MovieLens  
- Two-tower embeddings  
- Full Android client consuming `/recommend` API  

## Author

Sankalp Sahu
