import pandas as pd

from parali.forecast.metrics import evaluate, top_share_by_day


def _day(scores, fires):
    return pd.DataFrame(
        {
            "date": ["2025-10-01"] * len(scores),
            "s": scores,
            "fire_next3": [int(f > 0) for f in fires],
            "fires_next3_count": fires,
        }
    )


def test_top_share_counts_fires_in_highest_ranked_cells():
    # 10 cells; top 10% = 1 cell, the highest score, which holds 3 of the 4 fires
    df = _day([0.9, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3], [3, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    precision, capture = top_share_by_day(df, "s", 0.10)
    assert precision == 1.0
    assert capture == 0.75


def test_perfect_ranking_scores_one():
    df = _day([0.9, 0.8, 0.1, 0.0], [1, 2, 0, 0])
    assert evaluate(df, "s")["pr_auc"] == 1.0
