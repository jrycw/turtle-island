import polars as pl
from turtle_island._utils import _litify, _cast_datatype
import pytest


@pytest.mark.parametrize("items", [(1, 2), (3.3, 4.4), ("x", "y")])
def test__litify(items):
    litified = _litify(items)
    assert all(isinstance(lit, pl.Expr) for lit in litified)


@pytest.mark.parametrize(
    "item, expected",
    [
        (1, pl.Int64),
        (1.0, pl.Float64),
        ("1.0", pl.String),
        (object(), pl.Int64),  # the type of `pl.col("a")` is `pl.Int64``
    ],
)
def test__cast_datatype(df_abcd, item, expected):
    new_df = df_abcd.select(_cast_datatype(pl.col("a"), item))
    assert new_df.dtypes[0] == expected
