import pandas as pd

from movie_dt.features import TabularFeatures


def test_encodes_tmdb_columns():
    raw = pd.DataFrame(
        {
            "genres": [frozenset({"Action", "Drama"}), frozenset({"Comedy"})],
            "directors": [frozenset({"Nolan"}), frozenset({"Nolan", "Wright"})],
            "cast": [frozenset({"DiCaprio"}), frozenset({"Pegg"})],
            "year": [1999.0, 2010.0],
            "runtime": [120.0, 90.0],
            "vote_average": [8.0, 6.0],
            "vote_count": [1000.0, 50.0],
            "popularity": [20.0, 5.0],
            "in_collection": [1.0, 0.0],
            "adult": [0.0, 1.0],
        },
        index=[1, 2],
    )
    table = TabularFeatures.from_raw(raw)
    assert "genre:Action" in table.feature_names
    assert "director:Nolan" in table.feature_names
    assert "cast:DiCaprio" in table.feature_names
    assert "vote_count" in table.feature_names
    assert "popularity" in table.feature_names
    assert "adult" in table.feature_names
    assert table.values.shape == (2, len(table.feature_names))
    nolan = table.feature_names.index("director:Nolan")
    assert list(table.values[:, nolan]) == [1.0, 1.0]
