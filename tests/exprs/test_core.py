import polars as pl
import pytest
from polars.testing import assert_frame_equal

import turtle_island as ti


def test_make_index(df_x):
    df_ti = df_x.select(ti.make_index(), pl.all())
    df_pl = df_x.with_row_index()

    assert_frame_equal(df_ti, df_pl)


@pytest.mark.parametrize("offset", [0, 1, 10, 100])
def test_make_index_offset(df_x, offset):
    df_ti = df_x.select(ti.make_index(offset=offset), pl.all())
    df_pl = df_x.with_row_index(offset=offset)

    assert_frame_equal(df_ti, df_pl)


# ===
# === Test selected functions that internally use `_make_index()` if an index column exists ===
# ===


@pytest.mark.parametrize("offset", [0, 1, 10, 100])
def test_make_index_index_column_exist(df_x, offset):
    # intentionally use `with_columns()`
    assert_frame_equal(
        df_x.with_row_index().with_columns(ti.make_index(offset=offset)),
        pl.concat(
            [
                pl.DataFrame(
                    {"index": [offset, offset + 1, offset + 2, offset + 3]},
                    schema={"index": pl.UInt32},
                ),
                df_x,
            ],
            how="horizontal",
        ),
    )


@pytest.mark.parametrize(
    "exprs, result",
    [
        (
            (pl.lit("one"), pl.lit("two"), pl.lit("three"), pl.lit("four")),
            [
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
        ),
        (
            (pl.col("n").cast(pl.String), pl.col("n").add(10).cast(pl.String)),
            ["1", "12", "3", "14", "5", "16", "7", "18", "9"],
        ),
    ],
)
def test_bucketize_index_column_exist(df_n, exprs, result):
    _df = df_n.with_row_index()
    # intentionally use `with_columns()`
    new_df = _df.with_columns(ti.bucketize(*exprs))
    expected = pl.concat(
        [_df, pl.DataFrame({"bucketized": result})], how="horizontal"
    )
    assert_frame_equal(new_df, expected)


@pytest.mark.parametrize(
    "n,  s_bool",
    [
        (1, [True, True, True, True, True, True, True, True, True]),
        (2, [True, False, True, False, True, False, True, False, True]),
        (3, [True, False, False, True, False, False, True, False, False]),
        (4, [True, False, False, False, True, False, False, False, True]),
        (5, [True, False, False, False, False, True, False, False, False]),
        (6, [True, False, False, False, False, False, True, False, False]),
        (7, [True, False, False, False, False, False, False, True, False]),
        (8, [True, False, False, False, False, False, False, False, True]),
        (9, [True, False, False, False, False, False, False, False, False]),
        (10, [True, False, False, False, False, False, False, False, False]),
    ],
)
def test_is_every_nth_row_index_column_exist(df_n, n, s_bool):
    _df = df_n.with_row_index()
    expr = ti.is_every_nth_row(n)
    # intentionally use `with_columns()`
    new_df = _df.with_columns(expr)
    expected = pl.concat(
        [_df, pl.DataFrame({"bool_nth_row": s_bool})], how="horizontal"
    )

    assert_frame_equal(new_df, expected)

    return pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [1.11, 2.22, 3.33],
            "c": [4, 5, 6],
            "d": ["x", "y", "z"],
        }
    )
