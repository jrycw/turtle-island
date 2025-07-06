import polars as pl
from turtle_island._utils import _litify, _cast_datatype, _concat_str
import pytest


@pytest.mark.parametrize("items", [(1, 2), (3.3, 4.4), ("x", "y")])
def test__litify(items):
    litified = _litify(items)
    assert all(isinstance(lit, pl.Expr) for lit in litified)


@pytest.mark.parametrize(
    "item, expected",
    [
        (True, pl.Boolean),
        (False, pl.Boolean),
        (1, pl.Int64),
        (1.0, pl.Float64),
        ("1.0", pl.String),
        (object(), pl.Int64),  # the type of `pl.col("a")` is `pl.Int64``
    ],
)
def test__cast_datatype(df_abcd, item, expected):
    new_df = df_abcd.select(_cast_datatype(pl.col("a"), item))
    assert new_df.dtypes[0] == expected


def test__concat_str():
    quick, lazy = "quick", "lazy"
    fox, dog = "fox", "dog"
    concat_str_expr = _concat_str(
        f"The {quick} brown <<X>> jumps over the {lazy} <<X>>.",
        fox,
        dog,
    )
    expected = pl.concat_str(
        [
            pl.lit(f"The {quick} brown "),
            fox,
            pl.lit(f" jumps over the {lazy} "),
            dog,
            pl.lit("."),
        ]
    )

    assert concat_str_expr.meta.eq(expected)


def test__concat_str_sep():
    quick, lazy = "quick", "lazy"
    fox, dog = "fox", "dog"
    concat_str_expr = _concat_str(
        f"The {quick} brown ##<X>## jumps over the {lazy} ##<X>##.",
        fox,
        dog,
        sep="##<X>##",
    )
    expected = pl.concat_str(
        [
            pl.lit(f"The {quick} brown "),
            fox,
            pl.lit(f" jumps over the {lazy} "),
            dog,
            pl.lit("."),
        ]
    )

    assert concat_str_expr.meta.eq(expected)


def test__concat_str_fail_col_names_not_all_str():
    fox = "fox"
    with pytest.raises(ValueError) as exc_info:
        assert _concat_str(
            "The quick brown <<X>> jumps over the lazy <<X>>.", fox, 123
        )  # 123 is int type

    assert "All column names must be of type string." in exc_info.value.args[0]


def test__concat_str_fail_params_not_match():
    fox = "fox"
    with pytest.raises(ValueError) as exc_info:
        assert _concat_str(
            "The quick brown <<X>> jumps over the lazy <<X>>.", fox
        )  # `dog` is missed

    assert "which does not match the number of column names" in exc_info.value.args[0]
