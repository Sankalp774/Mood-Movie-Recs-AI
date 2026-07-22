"""Offline evaluation for Mood Movie Recs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.recommender import MoodMovieRecommender


def main() -> None:
    engine = MoodMovieRecommender.load()
    report = engine.evaluate_self_retrieval()
    # mood probe: does "cozy" surface warm films?
    cozy = engine.recommend(mood="cozy", top_k=5)
    report["cozy_top"] = [m["title"] for m in cozy]
    out = ROOT / "metrics" / "eval_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
