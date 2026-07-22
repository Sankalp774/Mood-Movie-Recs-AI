"""Content-based mood movie recommender (TF-IDF + cosine similarity)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "movies.json"

MOOD_CATALOG = [
    {"id": "cozy", "label": "Cozy", "emoji": "☕", "hint": "warm comfort watches"},
    {"id": "tense", "label": "Tense", "emoji": "😰", "hint": "edge-of-seat thrill"},
    {"id": "romantic", "label": "Romantic", "emoji": "💕", "hint": "love & longing"},
    {"id": "mind-bending", "label": "Mind-bending", "emoji": "🧠", "hint": "twist your brain"},
    {"id": "adrenaline", "label": "Adrenaline", "emoji": "⚡", "hint": "high-octane action"},
    {"id": "nostalgic", "label": "Nostalgic", "emoji": "📼", "hint": "warm flashbacks"},
    {"id": "inspirational", "label": "Inspirational", "emoji": "✨", "hint": "uplift & courage"},
    {"id": "dark", "label": "Dark", "emoji": "🌑", "hint": "noir & shadows"},
    {"id": "whimsical", "label": "Whimsical", "emoji": "🎩", "hint": "playful & odd"},
    {"id": "emotional", "label": "Emotional", "emoji": "💧", "hint": "feel everything"},
    {"id": "curious", "label": "Curious", "emoji": "🔍", "hint": "mysteries & ideas"},
    {"id": "epic", "label": "Epic", "emoji": "🏔️", "hint": "grand scale"},
]


@dataclass
class MoodMovieRecommender:
    movies: list[dict[str, Any]]
    vectorizer: TfidfVectorizer
    matrix: Any

    @classmethod
    def load(cls, path: Path | None = None) -> "MoodMovieRecommender":
        path = path or DATA_PATH
        movies = json.loads(path.read_text(encoding="utf-8"))
        docs = []
        for m in movies:
            genres = " ".join(m.get("genres") or [])
            moods = " ".join(m.get("moods") or [])
            docs.append(
                f"{m['title']} {genres} {moods} {m.get('overview', '')}".lower()
            )
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        matrix = normalize(vectorizer.fit_transform(docs))
        return cls(movies=movies, vectorizer=vectorizer, matrix=matrix)

    def _query_vector(self, text: str):
        return normalize(self.vectorizer.transform([text.lower()]))

    def recommend(
        self,
        mood: str | None = None,
        query: str | None = None,
        top_k: int = 8,
        liked_ids: list[int] | None = None,
        exclude_ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        parts: list[str] = []
        if mood:
            parts.append(mood.replace("-", " "))
        if query:
            parts.append(query)
        liked_ids = liked_ids or []
        exclude = set(exclude_ids or [])

        # Blend free-text query with liked movie profiles
        if liked_ids:
            liked_docs = []
            for m in self.movies:
                if m["id"] in liked_ids:
                    liked_docs.append(
                        " ".join(m.get("moods", []))
                        + " "
                        + " ".join(m.get("genres", []))
                        + " "
                        + m.get("overview", "")
                    )
            if liked_docs:
                parts.append(" ".join(liked_docs))

        if not parts:
            parts = ["feel good warm light cozy"]

        q = " ".join(parts)
        q_vec = self._query_vector(q)
        sims = cosine_similarity(q_vec, self.matrix).ravel()

        # Slight boost for higher IMDb-style ratings
        ratings = np.array([float(m.get("rating") or 7.0) for m in self.movies])
        scores = 0.85 * sims + 0.15 * (ratings / 10.0)

        order = scores.argsort()[::-1]
        out: list[dict[str, Any]] = []
        for i in order:
            m = self.movies[int(i)]
            if m["id"] in exclude:
                continue
            if m["id"] in liked_ids and not query:
                # still allow if strong mood match, but prefer new
                continue
            item = dict(m)
            item["score"] = round(float(scores[int(i)]), 4)
            item["match_pct"] = int(round(float(scores[int(i)]) * 100))
            out.append(item)
            if len(out) >= top_k:
                break
        return out

    def feed(self, limit: int = 12) -> list[dict[str, Any]]:
        """Discovery feed: high rating + recent-ish mix."""
        ranked = sorted(
            self.movies,
            key=lambda m: (float(m.get("rating") or 0), int(m.get("year") or 0)),
            reverse=True,
        )
        return ranked[:limit]

    def get(self, movie_id: int) -> dict[str, Any] | None:
        for m in self.movies:
            if m["id"] == movie_id:
                return m
        return None

    def search(self, q: str, top_k: int = 12) -> list[dict[str, Any]]:
        return self.recommend(query=q, top_k=top_k)

    def evaluate_self_retrieval(self) -> dict[str, float]:
        ranks = []
        for i, m in enumerate(self.movies):
            q_vec = self._query_vector(m["overview"])
            sims = cosine_similarity(q_vec, self.matrix).ravel()
            order = list(sims.argsort()[::-1])
            ranks.append(order.index(i) + 1)
        ranks_a = np.array(ranks, dtype=float)
        return {
            "n": float(len(ranks)),
            "mean_rank": float(ranks_a.mean()),
            "median_rank": float(np.median(ranks_a)),
            "hit_at_1": float((ranks_a == 1).mean()),
            "hit_at_3": float((ranks_a <= 3).mean()),
        }
