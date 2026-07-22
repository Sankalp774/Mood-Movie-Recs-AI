"""
Content-based movie recommender using TF-IDF + cosine similarity
over genres, moods, and overview text.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


@dataclass
class Recommendation:
    title: str
    year: int
    score: float
    genres: str
    moods: str
    overview: str


class MoodMovieRecommender:
    def __init__(self, csv_path: str | Path):
        self.df = pd.read_csv(csv_path)
        self.df["year"] = self.df["year"].astype(int)
        self.df["document"] = (
            self.df["title"].fillna("")
            + " "
            + self.df["genres"].fillna("").str.replace("|", " ", regex=False)
            + " "
            + self.df["moods"].fillna("").str.replace("|", " ", regex=False)
            + " "
            + self.df["overview"].fillna("")
        )
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(self.df["document"])
        self.matrix = normalize(self.matrix)

    def recommend(
        self,
        mood: str | None = None,
        query: str | None = None,
        top_k: int = 5,
    ) -> list[Recommendation]:
        parts = []
        if mood:
            parts.append(mood)
        if query:
            parts.append(query)
        if not parts:
            parts = ["feel-good warm light"]
        q = " ".join(parts)
        q_vec = normalize(self.vectorizer.transform([q]))
        sims = cosine_similarity(q_vec, self.matrix).ravel()
        idx = sims.argsort()[::-1][:top_k]
        out: list[Recommendation] = []
        for i in idx:
            row = self.df.iloc[int(i)]
            out.append(
                Recommendation(
                    title=row["title"],
                    year=int(row["year"]),
                    score=float(sims[int(i)]),
                    genres=row["genres"],
                    moods=row["moods"],
                    overview=row["overview"],
                )
            )
        return out

    def evaluate_self_retrieval(self) -> dict:
        """Sanity metric: each movie's overview should rank itself highly."""
        ranks = []
        for i, row in self.df.iterrows():
            q = row["overview"]
            q_vec = normalize(self.vectorizer.transform([q]))
            sims = cosine_similarity(q_vec, self.matrix).ravel()
            order = sims.argsort()[::-1]
            rank = int(list(order).index(i)) + 1
            ranks.append(rank)
        import numpy as np

        return {
            "n": len(ranks),
            "mean_rank": float(np.mean(ranks)),
            "median_rank": float(np.median(ranks)),
            "hit_at_1": float(np.mean([r == 1 for r in ranks])),
            "hit_at_3": float(np.mean([r <= 3 for r in ranks])),
        }
