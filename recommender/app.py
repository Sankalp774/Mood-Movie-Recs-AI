"""CLI + Gradio for mood-based movie recommendations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from recommender.engine import MoodMovieRecommender

DATA = Path(__file__).resolve().parent / "data" / "movies.csv"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mood", type=str, default="cozy warm")
    p.add_argument("--query", type=str, default="")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--ui", action="store_true")
    args = p.parse_args()

    rec = MoodMovieRecommender(DATA)
    if args.eval:
        report = rec.evaluate_self_retrieval()
        out = Path(__file__).resolve().parents[1] / "metrics" / "eval_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return

    if args.ui:
        import gradio as gr

        def run(mood, query, k):
            rows = rec.recommend(mood=mood, query=query, top_k=int(k))
            return [
                [r.title, r.year, round(r.score, 3), r.moods, r.genres]
                for r in rows
            ]

        gr.Interface(
            fn=run,
            inputs=[
                gr.Textbox(value="cozy warm", label="Mood"),
                gr.Textbox(value="", label="Optional free-text query"),
                gr.Slider(1, 10, value=5, step=1, label="Top K"),
            ],
            outputs=gr.Dataframe(
                headers=["title", "year", "score", "moods", "genres"]
            ),
            title="Mood Movie Recs — content-based recommender",
        ).launch()
        return

    rows = rec.recommend(mood=args.mood, query=args.query or None, top_k=args.top_k)
    for r in rows:
        print(f"{r.score:.3f}  {r.title} ({r.year})  [{r.moods}]")
        print(f"       {r.overview}\n")


if __name__ == "__main__":
    main()
