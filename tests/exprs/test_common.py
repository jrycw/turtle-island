import polars as pl
import pytest
from polars.testing import assert_frame_equal

import turtle_island as ti


def test_case_when(df_abcd):
    expr_ti = ti.case_when(
        case_list=[
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
        case_list=[
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


def test_case_when_all_forms(df_xy):
    expr1 = ti.case_when(
        case_list=[
            (pl.col("x") < 2, pl.lit("small")),
            (pl.col("x") < 4, pl.lit("medium")),
        ],
        otherwise=pl.lit("large"),
    ).alias("size1")

    expr2 = ti.case_when(
        case_list=[
            (pl.col("x") < 3, pl.col("y") < 6, pl.lit("small")),
            (pl.col("x") < 4, pl.col("y") < 8, pl.lit("medium")),
        ],
        otherwise=pl.lit("large"),
    ).alias("size2")

    expr3 = ti.case_when(
        case_list=[
            ((pl.col("x") < 3, pl.col("y") < 6), pl.lit("small")),
            ((pl.col("x") < 4, pl.col("y") < 8), pl.lit("medium")),
        ],
        otherwise=pl.lit("large"),
    ).alias("size3")

    new_df = df_xy.select(expr1, expr2, expr3)
    expected = pl.DataFrame(
        {
            "size1": ["small", "medium", "medium", "large"],
            "size2": ["small", "medium", "medium", "large"],
            "size3": ["small", "medium", "medium", "large"],
        }
    )

    assert_frame_equal(new_df, expected)


@pytest.mark.parametrize(
    "exprs",
    [
        (pl.col("a").last().cast(pl.Float64), pl.col("b").first()),
        ([pl.col("a").last().cast(pl.Float64), pl.col("b").first()]),
    ],
)
def test_bulk_append(df_abcd, exprs):
    new_df = df_abcd.select(ti.bulk_append(exprs))
    expected = pl.DataFrame({"a": [3.0, 1.11]})

    assert_frame_equal(new_df, expected)


def test_bulk_append_pl_all(df_abcd):
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


def test_bulk_append_list_eval(df_xy_list):
    new_df = df_xy_list.select(
        pl.all().list.eval(
            ti.bulk_append(pl.element().first(), pl.element().last())
        )
    )
    expected = pl.DataFrame(
        {
            "x": [[1, 4], [5, 8]],
            "y": [[9, 12], [13, 16]],
        }
    )

    assert_frame_equal(new_df, expected)


@pytest.mark.parametrize("exprs", [pl.lit(1), (pl.lit(1),)])
def test_bulk_append_raise_one_element(exprs):
    with pytest.raises(ValueError) as exc_info:
        ti.bulk_append(exprs)

    assert (
        "At least two Polars expressions must be provided."
        in exc_info.value.args[0]
    )


def test_shift_pre_fill(df_x):
    new_df = df_x.select(
        ti.shift(pl.col("x"), 2, fill_expr=pl.col("x").add(100))
    )
    expected = pl.DataFrame({"x": [101, 102, 1, 2]})
    assert_frame_equal(new_df, expected)


def test_shift_back_fill(df_x):
    new_df = df_x.select(
        ti.shift(pl.col("x"), -2, fill_expr=pl.col("x").add(100))
    )
    expected = pl.DataFrame({"x": [3, 4, 103, 104]})
    assert_frame_equal(new_df, expected)


def test_shift_default(df_x):
    new_df = df_x.select(ti.shift(pl.col("x"), fill_expr=pl.col("x").add(100)))
    expected = pl.DataFrame({"x": [101, 1, 2, 3]})
    assert_frame_equal(new_df, expected)


def test_shift_pl_all(df_xy):
    new_df = df_xy.with_columns(
        ti.shift(pl.all(), fill_expr=pl.col("y").alias("z").add(100))
    )
    expected = pl.DataFrame({"x": [105, 1, 2, 3], "y": [105, 5, 6, 7]})
    assert_frame_equal(new_df, expected)


def test_shift_offset_zero_return_self():
    expr = pl.col("x")
    expected = ti.shift(expr, 0, fill_expr=pl.col("x").add(100))

    assert expr is expected


def test_shift_list_eval(df_xy_list):
    new_df = df_xy_list.select(
        pl.all().list.eval(
            ti.shift(pl.element(), 2, fill_expr=pl.element().add(10))
        )
    )
    expected = pl.DataFrame(
        {
            "x": [[11, 12, 1, 2], [15, 16, 5, 6]],
            "y": [[19, 20, 9, 10], [23, 24, 13, 14]],
        }
    )

    assert_frame_equal(new_df, expected)


def test_shift_raise_offset_not_integer():
    with pytest.raises(ValueError) as exc_info:
        ti.shift(pl.col("x"), 1.1, fill_expr=pl.col("x").add(100))

    assert "`offset=` must be an integer." in exc_info.value.args[0]


def test_prepend(df_x):
    new_df = df_x.select(ti.prepend(pl.col("x")))
    expected = pl.DataFrame({"x": [1, 1, 2, 3, 4]})
    assert_frame_equal(new_df, expected)


def test_prepend_default(df_x):
    new_df = df_x.select(
        ti.prepend(pl.col("x"), prepend_expr=pl.col("x").add(100))
    )
    expected = pl.DataFrame({"x": [101, 1, 2, 3, 4]})
    assert_frame_equal(new_df, expected)


@pytest.mark.parametrize(
    "offset, result",
    [
        (1, [1, 1, 2, 3, 4]),
        (2, [1, 2, 1, 2, 3, 4]),
        (3, [1, 2, 3, 1, 2, 3, 4]),
        (4, [1, 2, 3, 4, 1, 2, 3, 4]),
        (5, [1, 2, 3, 4, 1, 2, 3, 4]),  # current limitation
    ],
)
def test_prepend_offset(df_x, offset, result):
    new_df = df_x.select(ti.prepend(pl.col("x"), offset=offset))
    expected = pl.DataFrame({"x": result})
    assert_frame_equal(new_df, expected)


def test_prepend_pl_all(df_xy):
    # can not use `df_xy.with_columns()`
    new_df = df_xy.select(ti.prepend(pl.all()))
    expected = pl.DataFrame({"x": [1, 1, 2, 3, 4], "y": [5, 5, 6, 7, 8]})
    assert_frame_equal(new_df, expected)


def test_prepend_offset_zero_return_self():
    expr = pl.col("x")
    expected = ti.prepend(expr, 0)

    assert expr is expected


def test_prepend_list_eval(df_xy_list):
    new_df = df_xy_list.select(
        pl.all().list.eval(
            ti.prepend(
                pl.element().add(1), offset=2, prepend_expr=pl.element().mul(2)
            )
        )
    )
    expected = pl.DataFrame(
        {
            "x": [[2, 4, 2, 3, 4, 5], [10, 12, 6, 7, 8, 9]],
            "y": [[18, 20, 10, 11, 12, 13], [26, 28, 14, 15, 16, 17]],
        }
    )

    assert_frame_equal(new_df, expected)


@pytest.mark.parametrize("prepend_expr", [0, pl.lit(0)])
def test_prepend_list_eval_lit(df_xy_list, prepend_expr):
    new_df = df_xy_list.select(
        pl.all().list.eval(
            ti.prepend(
                pl.element().add(1), offset=2, prepend_expr=prepend_expr
            )
        )
    )
    expected = pl.DataFrame(
        {
            "x": [[0, 0, 2, 3, 4, 5], [0, 0, 6, 7, 8, 9]],
            "y": [[0, 0, 10, 11, 12, 13], [0, 0, 14, 15, 16, 17]],
        }
    )

    assert_frame_equal(new_df, expected)


def test_prepend_raise_offset_negative():
    with pytest.raises(ValueError) as exc_info:
        ti.prepend(pl.col("x"), -1)

    assert "`offset=` cannot be negative." in exc_info.value.args[0]
