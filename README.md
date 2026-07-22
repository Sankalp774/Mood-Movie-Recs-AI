# Mood Movie Recs — Content-Based Recommender

> **Academic stage:** **3rd year** exploration · **Created:** February 2025  
> **Learning focus:** Recommendation systems · content-based filtering · TF-IDF + cosine similarity · mood features · Hit@K sanity metrics · Android/Kotlin UI sketch

## About

**Real ML recommender** (content-based) plus early Android stubs:

- TF-IDF over title + genres + mood tags + overview  
- Cosine similarity ranking  
- Offline self-retrieval metrics  
- CLI + Gradio  

## What I learned

| Topic | How this project taught it |
|-------|----------------------------|
| Recommender systems | Content-based vs collaborative ideas |
| Vector space IR | TF-IDF, cosine similarity |
| Feature design | Mood tags as explicit signals |
| Offline eval | Hit@1 / Hit@3 / mean rank |
| Mobile (intro) | Kotlin / Compose skeleton |

## Quickstart

```bash
pip install -r requirements.txt
python -m recommender.app --mood "tense dark" --top-k 5
python -m recommender.app --eval
python -m recommender.app --ui
```

## Status

If this repo is **archived** on GitHub, unarchive to receive latest pushes. Kotlin files are secondary; the recommender package is the hiring-relevant core.

## Author

Sankalp Sahu
