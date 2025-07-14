import polars as pl
import pytest
from polars.testing import assert_frame_equal

import turtle_island as ti


def test_case_when(df_abcd):
    expr_ti = ti.case_when(
        caselist=[
            (pl.col("a") < 2, pl.col("a")),
            (pl.col("a") < 3, pl.col("a") * 2),
        ],
        otherwise=pl.col("b"),
    )
    expr_pl = (
        pl.when(pl.col("a") < 2)
        .then(pl.col("a"))
        .when(pl.col("a") < 3)
        .then(pl.col("a") * 2)
        .otherwise(pl.col("b"))
    )
    assert expr_ti.meta.eq(expr_pl)

    # test the outcome
    df_ti = df_abcd.select(expr_ti)
    df_pl = df_abcd.select(expr_pl)

    assert_frame_equal(df_ti, df_pl)


def test_case_when_lit(df_x):
    # test the expression itself
    expr_ti = ti.case_when(
        caselist=[
            (pl.col("x") < 2, pl.lit("small")),
            (pl.col("x") < 4, pl.lit("medium")),
        ],
        otherwise=pl.lit("large"),
    )
    expr_pl = (
        pl.when(pl.col("x") < 2)
        .then(pl.lit("small"))
        .when(pl.col("x") < 4)
        .then(pl.lit("medium"))
        .otherwise(pl.lit("large"))
    )

    assert expr_ti.meta.eq(expr_pl)

    # test the outcome
    df_ti = df_x.select(expr_ti)
    df_pl = df_x.select(expr_pl)

    assert_frame_equal(df_ti, df_pl)


def test_bulk_append(df_abcd):
    exprs = [pl.all().last(), pl.all().first()]
    new_df = df_abcd.select(ti.bulk_append(*exprs))
    expected = pl.DataFrame(
        {
            "a": [3, 1],
            "b": [3.33, 1.11],
            "c": [6, 4],
            "d": ["z", "x"],
        }
    )

    assert_frame_equal(new_df, expected)


def test_bulk_append_raise():
    with pytest.raises(ValueError) as exc_info:
        ti.bulk_append(pl.lit(1))

    assert (
        "At least two Polars expressions must be provided."
        in exc_info.value.args[0]
    )
