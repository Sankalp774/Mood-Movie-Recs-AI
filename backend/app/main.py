"""
Mood Movie Recs — FastAPI backend + static web host.

  uvicorn backend.app.main:app --reload --port 8012
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.app.emotion import analyze_image_b64
from backend.app.recommender import MOOD_CATALOG, MoodMovieRecommender

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"

engine = MoodMovieRecommender.load()

app = FastAPI(
    title="Mood Movie Recs",
    description="Mood-based content recommender + OpenCV facial expression API",
    version="2.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    mood: Optional[str] = None
    query: Optional[str] = None
    top_k: int = Field(default=8, ge=1, le=20)
    liked_ids: list[int] = Field(default_factory=list)
    exclude_ids: list[int] = Field(default_factory=list)


class EmotionRequest(BaseModel):
    image: str = Field(..., description="Base64 image or data URL from webcam")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "movies": len(engine.movies),
        "version": "2.1.0",
        "features": ["recommend", "opencv_emotion", "stack_ui"],
    }


@app.post("/api/emotion")
def emotion_from_webcam(body: EmotionRequest):
    """OpenCV face detection + FER+ expression → mood for recommendations."""
    try:
        return analyze_image_b64(body.image)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(500, f"Emotion pipeline failed: {e}") from e

@app.get("/api/moods")
def moods():
    return {"moods": MOOD_CATALOG}


@app.get("/api/movies")
def list_movies():
    return {"movies": engine.movies, "count": len(engine.movies)}


@app.get("/api/movies/{movie_id}")
def movie_detail(movie_id: int):
    m = engine.get(movie_id)
    if not m:
        raise HTTPException(404, "Movie not found")
    return m


@app.get("/api/feed")
def discovery_feed(limit: int = Query(12, ge=1, le=40)):
    return {"feed": engine.feed(limit=limit)}


@app.get("/api/search")
def search(q: str = Query(..., min_length=1), top_k: int = 12):
    return {"results": engine.search(q, top_k=top_k), "q": q}


@app.post("/api/recommend")
def recommend(body: RecommendRequest):
    recs = engine.recommend(
        mood=body.mood,
        query=body.query,
        top_k=body.top_k,
        liked_ids=body.liked_ids,
        exclude_ids=body.exclude_ids,
    )
    return {
        "recommendations": recs,
        "mood": body.mood,
        "query": body.query,
        "count": len(recs),
    }


@app.get("/api/eval")
def eval_metrics():
    return engine.evaluate_self_retrieval()


# Static site (production-style single host)
if WEB.exists():
    app.mount("/assets", StaticFiles(directory=WEB / "assets"), name="assets")

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    @app.get("/{path:path}")
    def spa_fallback(path: str):
        # API already matched; serve files or index for SPA
        candidate = WEB / path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(WEB / "index.html")
