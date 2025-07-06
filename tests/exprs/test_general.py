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


def test_create_index(df_x):
    df_ti = df_x.select(ti.create_index(), pl.all())
    df_pl = df_x.with_row_index()

    assert_frame_equal(df_ti, df_pl)


@pytest.mark.parametrize(
    "columns, result",
    [
        (["a"], ["a", "b", "c", "d"]),
        (["b"], ["b", "a", "c", "d"]),
        (["c"], ["c", "a", "b", "d"]),
        (["d"], ["d", "a", "b", "c"]),
        (["c", "a"], ["c", "a", "b", "d"]),
        (["d", "c", "b"], ["d", "c", "b", "a"]),
        (pl.String, ["d", "a", "b", "c"]),
        ([pl.Int64, pl.String], ["a", "c", "d", "b"]),
        ([pl.String, pl.Int64], ["a", "c", "d", "b"]),
    ],
)
def test_move_cols_to_start(df_abcd, columns, result):
    new_df = df_abcd.select(ti.move_cols_to_start(columns))

    assert new_df.columns == result


@pytest.mark.parametrize(
    "columns, result",
    [
        (["a"], ["b", "c", "d", "a"]),
        (["b"], ["a", "c", "d", "b"]),
        (["c"], ["a", "b", "d", "c"]),
        (["d"], ["a", "b", "c", "d"]),
        (["c", "a"], ["b", "d", "c", "a"]),
        (["d", "c", "b"], ["a", "d", "c", "b"]),
        (pl.String, ["a", "b", "c", "d"]),
        ([pl.Int64, pl.String], ["b", "a", "c", "d"]),
        ([pl.String, pl.Int64], ["b", "a", "c", "d"]),
    ],
)
def test_move_cols_to_end(df_abcd, columns, result):
    new_df = df_abcd.select(ti.move_cols_to_end(columns))

    assert new_df.columns == result


def test_bucketize_int(df_n):
    new_df = df_n.select(ti.bucketize(1, 2))
    expected = pl.DataFrame({"bucketized": [1, 2, 1, 2, 1, 2, 1, 2, 1]})

    assert_frame_equal(new_df, expected)


def test_bucketize_float(df_n):
    new_df = df_n.select(ti.bucketize(1.1, 2.2, 3.3))
    expected = pl.DataFrame(
        {"bucketized": [1.1, 2.2, 3.3, 1.1, 2.2, 3.3, 1.1, 2.2, 3.3]}
    )

    assert_frame_equal(new_df, expected)


def test_bucketize_str(df_n):
    new_df = df_n.select(ti.bucketize("one", "two", "three", "four"))
    expected = pl.DataFrame(
        {
            "bucketized": [
                "one",
                "two",
                "three",
                "four",
                "one",
                "two",
                "three",
                "four",
                "one",
            ]
        }
    )

    assert_frame_equal(new_df, expected)


@pytest.mark.parametrize("name", ["cool_name"])
def test_bucketize_alias(df_n, name):
    new_df = df_n.select(ti.bucketize(1, 2, name=name))
    expected = pl.DataFrame({name: [1, 2, 1, 2, 1, 2, 1, 2, 1]})

    assert_frame_equal(new_df, expected)


def test_bucketize_same_item(df_n):
    new_df = df_n.select(ti.bucketize(1, 1, 1))
    expected = pl.DataFrame({"bucketized": [1, 1, 1, 1, 1, 1, 1, 1, 1]})

    assert_frame_equal(new_df, expected)


def test_bucketize_multicols(df_n):
    new_df = df_n.select(
        ti.bucketize(1, 2, name="binarized"),
        ti.bucketize(1.1, 2.2, 3.3, name="trinarized"),
        ti.bucketize("one", "two", "three", "four", name="bucketized"),
    )
    expected = pl.DataFrame(
        {
            "binarized": [1, 2, 1, 2, 1, 2, 1, 2, 1],
            "trinarized": [1.1, 2.2, 3.3, 1.1, 2.2, 3.3, 1.1, 2.2, 3.3],
            "bucketized": [
                "one",
                "two",
                "three",
                "four",
                "one",
                "two",
                "three",
                "four",
                "one",
            ],
        }
    )

    assert_frame_equal(new_df, expected)


def test_bucketize_fail_one_item():
    with pytest.raises(ValueError) as exc_info:
        assert ti.bucketize(1)

    assert "must contain a minimum of two items." in exc_info.value.args[0]


def test_bucketize_fail_not_the_same_type():
    with pytest.raises(ValueError) as exc_info:
        assert ti.bucketize(1, "1")

    assert "must contain only one unique type." in exc_info.value.args[0]


def test_is_nth_row(df_n):
    new_df = df_n.select(ti.is_nth_row(3))
    expected = pl.DataFrame(
        {"bool_nth_row": [True, False, False, True, False, False, True, False, False]}
    )

    assert_frame_equal(new_df, expected)


def test_is_nth_row_ne(df_n):
    new_df = df_n.select(~ti.is_nth_row(3))
    expected = pl.DataFrame(
        {"bool_nth_row": [False, True, True, False, True, True, False, True, True]}
    )

    assert_frame_equal(new_df, expected)
