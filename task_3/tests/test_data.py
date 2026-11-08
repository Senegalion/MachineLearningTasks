import pandas as pd

from movie_cf.data import load_ratings, save_submission


def test_null_ratings_round_trip(tmp_path):
    source = tmp_path / "task.csv"
    source.write_text("7;2;3;NULL\n8;2;4;5\n", encoding="utf-8")
    task = load_ratings(source)
    assert pd.isna(task.loc[0, "rating"])
    assert int(task.loc[1, "rating"]) == 5

    destination = tmp_path / "submission.csv"
    save_submission(task, pd.Series([4, 5]), destination)
    assert destination.read_text(encoding="utf-8").splitlines() == ["7;2;3;4", "8;2;4;5"]
