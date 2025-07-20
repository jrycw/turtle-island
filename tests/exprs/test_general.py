import polars as pl
import pytest
from polars.testing import assert_frame_equal

import turtle_island as ti


@pytest.mark.parametrize(
    "columns, result",
    [
        ("a", ["a", "b", "c", "d"]),
        ("b", ["b", "a", "c", "d"]),
        ("c", ["c", "a", "b", "d"]),
        ("d", ["d", "a", "b", "c"]),
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
        ("a", ["b", "c", "d", "a"]),
        ("b", ["a", "c", "d", "b"]),
        ("c", ["a", "b", "d", "c"]),
        ("d", ["a", "b", "c", "d"]),
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


@pytest.mark.parametrize(
    "items, result",
    [
        ((1, 2), [1, 2, 1, 2, 1, 2, 1, 2, 1]),
        ((1.1, 2.2, 3.3), [1.1, 2.2, 3.3, 1.1, 2.2, 3.3, 1.1, 2.2, 3.3]),
        (
            ("one", "two", "three", "four"),
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
        ((1, 1, 1), [1, 1, 1, 1, 1, 1, 1, 1, 1]),  # test same item
    ],
)
def test_bucketize_lit(df_n, items, result):
    new_df = df_n.select(ti.bucketize_lit(*items))
    expected = pl.DataFrame({"bucketized": result})

    assert_frame_equal(new_df, expected)


@pytest.mark.parametrize("name", ["cool_name"])
def test_bucketize_lit_alias(df_n, name):
    new_df = df_n.select(ti.bucketize_lit(1, 2, name=name))
    expected = pl.DataFrame({name: [1, 2, 1, 2, 1, 2, 1, 2, 1]})

    assert_frame_equal(new_df, expected)


@pytest.mark.parametrize(
    "items, result, coalesce_to",
    [
        ((1, 2), ["1", "2", "1", "2", "1", "2", "1", "2", "1"], pl.String),
        ((True, False), [1, 0, 1, 0, 1, 0, 1, 0, 1], pl.Int64),
        (
            (0, 1),
            [False, True, False, True, False, True, False, True, False],
            pl.Boolean,
        ),
    ],
)
def test_bucketize_lit_coalesce_to(df_n, items, result, coalesce_to):
    new_df = df_n.select(ti.bucketize_lit(*items, coalesce_to=coalesce_to))
    expected = pl.DataFrame({"bucketized": result})

    assert_frame_equal(new_df, expected)


def test_bucketize_lit_multicols(df_n):
    new_df = df_n.select(
        ti.bucketize_lit(1, 2, name="binarized"),
        ti.bucketize_lit(1.1, 2.2, 3.3, name="trinarized"),
        ti.bucketize_lit("one", "two", "three", "four", name="bucketized"),
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


def test_bucketize_lit_raise_one_item():
    with pytest.raises(ValueError) as exc_info:
        ti.bucketize_lit(1)

    assert (
        "`items=` must contain a minimum of two items."
        in exc_info.value.args[0]
    )


def test_bucketize_lit_raise_not_the_same_type():
    with pytest.raises(ValueError) as exc_info:
        ti.bucketize_lit(1, "1")

    assert (
        "`items=` must contain only one unique type." in exc_info.value.args[0]
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
def test_bucketize(df_n, exprs, result):
    new_df = df_n.select(ti.bucketize(*exprs))
    expected = pl.DataFrame({"bucketized": result})
    assert_frame_equal(new_df, expected)


@pytest.mark.parametrize("name", ["cool_name"])
def test_bucketize_alias(df_n, name):
    new_df = df_n.select(ti.bucketize(pl.col("n"), pl.col("n"), name=name))
    expected = pl.DataFrame({name: [1, 2, 3, 4, 5, 6, 7, 8, 9]})

    assert_frame_equal(new_df, expected)


@pytest.mark.parametrize(
    "exprs, result, coalesce_to",
    [
        (
            (pl.col("n"), pl.col("n").add(10)),
            ["1", "12", "3", "14", "5", "16", "7", "18", "9"],
            pl.String,
        ),
        (
            (pl.lit(True), pl.lit(False)),
            [1, 0, 1, 0, 1, 0, 1, 0, 1],
            pl.Int64,
        ),
    ],
)
def test_bucketize_coalesce_to(df_n, exprs, result, coalesce_to):
    new_df = df_n.select(ti.bucketize(*exprs, coalesce_to=coalesce_to))
    expected = pl.DataFrame({"bucketized": result})

    assert_frame_equal(new_df, expected)


def test_bucketize_raise_one_item():
    with pytest.raises(ValueError) as exc_info:
        ti.bucketize(pl.lit(1))

    assert (
        "`exprs=` must contain a minimum of two expressions."
        in exc_info.value.args[0]
    )


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
def test_is_every_nth_row(df_n, n, s_bool):
    expr = ti.is_every_nth_row(n)
    new_df = df_n.select(expr)
    expected = pl.DataFrame({"bool_nth_row": s_bool})

    assert_frame_equal(new_df, expected)

    # https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.gather_every.html
    assert_frame_equal(
        df_n.filter(expr), df_n.select(pl.col("n").gather_every(n))
    )


@pytest.mark.parametrize(
    "n,  s_bool",
    [
        (1, [False, False, False, False, False, False, False, False, False]),
        (2, [False, True, False, True, False, True, False, True, False]),
        (3, [False, True, True, False, True, True, False, True, True]),
        (4, [False, True, True, True, False, True, True, True, False]),
        (5, [False, True, True, True, True, False, True, True, True]),
        (6, [False, True, True, True, True, True, False, True, True]),
        (7, [False, True, True, True, True, True, True, False, True]),
        (8, [False, True, True, True, True, True, True, True, False]),
        (9, [False, True, True, True, True, True, True, True, True]),
        (10, [False, True, True, True, True, True, True, True, True]),
    ],
)
def test_is_every_nth_row_ne(df_n, n, s_bool):
    expr = ~ti.is_every_nth_row(n)
    new_df = df_n.select(expr)
    expected = pl.DataFrame({"bool_nth_row": s_bool})

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
def test_is_every_nth_row_ne_twice(df_n, n, s_bool):
    expr = ~(~ti.is_every_nth_row(n))
    new_df = df_n.select(expr)
    expected = pl.DataFrame({"bool_nth_row": s_bool})

    assert_frame_equal(new_df, expected)

    # Verify that the results are equal
    assert_frame_equal(
        df_n.filter(expr), df_n.select(pl.col("n").gather_every(n))
    )


@pytest.mark.parametrize(
    "n, offset, s_bool",
    [
        (3, 0, [True, False, False, True, False, False, True, False, False]),
        (3, 1, [False, True, False, False, True, False, False, True, False]),
        (3, 2, [False, False, True, False, False, True, False, False, True]),
        (3, 3, [False, False, False, True, False, False, True, False, False]),
        (3, 4, [False, False, False, False, True, False, False, True, False]),
        (3, 5, [False, False, False, False, False, True, False, False, True]),
        (3, 6, [False, False, False, False, False, False, True, False, False]),
        (3, 7, [False, False, False, False, False, False, False, True, False]),
        (3, 8, [False, False, False, False, False, False, False, False, True]),
        (
            3,
            9,
            [False, False, False, False, False, False, False, False, False],
        ),
        (
            3,
            10,
            [False, False, False, False, False, False, False, False, False],
        ),
    ],
)
def test_is_every_nth_row_offset(df_n, n, offset, s_bool):
    new_df = df_n.select(ti.is_every_nth_row(n, offset))
    expected = pl.DataFrame({"bool_nth_row": s_bool})

    assert_frame_equal(new_df, expected)


@pytest.mark.parametrize("n", [0, -1, -10, -100])
def test_is_every_nth_row_raise_neg_n(n):
    with pytest.raises(ValueError) as exc_info:
        ti.is_every_nth_row(n)

    assert "`n=` should be positive." in exc_info.value.args[0]


@pytest.mark.parametrize("offset", [-1, -10, -100])
def test_is_every_nth_row_raise_neg_offset(offset):
    with pytest.raises(ValueError) as exc_info:
        ti.is_every_nth_row(999, offset=offset)

    assert "`offset=` cannot be negative." in exc_info.value.args[0]


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


def test_shift_n_zero_return_self():
    expr = pl.col("x")
    expected = ti.shift(expr, 0, fill_expr=pl.col("x").add(100))

    assert expr.meta.eq(expected)


def test_shift_raise_offset_not_integer():
    with pytest.raises(ValueError) as exc_info:
        ti.shift(pl.col("x"), 1.1, fill_expr=pl.col("x").add(100))

    assert "`offset=` must be an integer." in exc_info.value.args[0]


@pytest.mark.parametrize(
    "offset, result",
    [
        (-5, [2, 3, 4, 1]),
        (-4, [1, 2, 3, 4]),
        (-3, [4, 1, 2, 3]),
        (-2, [3, 4, 1, 2]),
        (-1, [2, 3, 4, 1]),
        (0, [1, 2, 3, 4]),
        (1, [4, 1, 2, 3]),
        (2, [3, 4, 1, 2]),
        (3, [2, 3, 4, 1]),
        (4, [1, 2, 3, 4]),
        (5, [4, 1, 2, 3]),
    ],
)
def test_cycle(df_x, offset, result):
    new_df = df_x.select(ti.cycle(pl.col("x"), offset=offset))
    expected = pl.DataFrame({"x": result})

    assert_frame_equal(new_df, expected)


def test_cycle_default(df_x):
    new_df = df_x.select(ti.cycle(pl.col("x")))
    expected = pl.DataFrame({"x": [4, 1, 2, 3]})

    assert_frame_equal(new_df, expected)


def test_cycle_raise_offset_not_integer():
    with pytest.raises(ValueError) as exc_info:
        ti.cycle(pl.col("x"), 1.1)

    assert "`offset=` must be an integer." in exc_info.value.args[0]
