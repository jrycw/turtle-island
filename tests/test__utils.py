import datetime

import polars as pl
import pytest

from turtle_island._utils import _cast_datatype, _get_unique_name, _litify


@pytest.mark.parametrize("items", [(1, 2), (3.3, 4.4), ("x", "y")])
def test__litify(items):
    litified = _litify(items)
    assert all(isinstance(lit, pl.Expr) for lit in litified)


@pytest.mark.parametrize("n", [10, 11, 12])
def test__get_unique_name(n):
    name1 = _get_unique_name(n)
    name2 = _get_unique_name(n)
    assert name1 != name2
    assert len(name1) == len(name2) == n


def test__get_unique_name_raise():
    with pytest.raises(ValueError) as exc_info:
        _get_unique_name(7)

    assert (
        "`n` must be at least 8 to ensure uniqueness of the name."
        in exc_info.value.args[0]
    )


@pytest.mark.parametrize(
    "item, expected",
    [
        (True, pl.Boolean),
        (False, pl.Boolean),
        (datetime.datetime(2025, 1, 1), pl.Datetime),
        (datetime.date(2025, 1, 1), pl.Date),
        (datetime.time(13, 0), pl.Time),
        (datetime.timedelta(hours=1), pl.Duration),
        (1, pl.Int64),
        (1.0, pl.Float64),
        ("1.0", pl.String),
        ([1, 2, 3], pl.List),
        ((1, 2, 3), pl.List),
        # the type of `pl.col("a")` is `pl.Int64` => return self without casting
        (object(), pl.Int64),
    ],
)
def test__cast_datatype(df_abcd, item, expected):
    new_df = df_abcd.select(_cast_datatype(pl.col("a"), item))
    assert new_df.dtypes[0] == expected
