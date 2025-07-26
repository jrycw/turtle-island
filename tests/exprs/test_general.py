import polars as pl
import pytest
from polars.testing import assert_frame_equal

import turtle_island as ti
from turtle_island.exprs.general import _get_move_cols, _make_concat_str


def test__get_move_cols():
    assert _get_move_cols(["col1"]) == ["col1"]
    assert _get_move_cols("col1", "col2") == ["col1", "col2"]
    assert _get_move_cols(["col1"], "col2") == ["col1", "col2"]


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
    name = "bucketized"
    new_df = df_n.select(ti.bucketize_lit(*items).alias(name))
    expected = pl.DataFrame({name: result})

    assert_frame_equal(new_df, expected)


@pytest.mark.parametrize(
    "items, result, return_dtype",
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
def test_bucketize_lit_return_dtype(df_n, items, result, return_dtype):
    name = "bucketized"
    new_df = df_n.select(
        ti.bucketize_lit(*items, return_dtype=return_dtype).alias(name)
    )
    expected = pl.DataFrame({name: result})

    assert_frame_equal(new_df, expected)


def test_bucketize_lit_multicols(df_n):
    binarized, trinarized, bucketized = "binarized", "trinarized", "bucketized"
    new_df = df_n.select(
        ti.bucketize_lit(1, 2).alias(binarized),
        ti.bucketize_lit(1.1, 2.2, 3.3).alias(trinarized),
        ti.bucketize_lit("one", "two", "three", "four").alias(bucketized),
    )
    expected = pl.DataFrame(
        {
            binarized: [1, 2, 1, 2, 1, 2, 1, 2, 1],
            trinarized: [1.1, 2.2, 3.3, 1.1, 2.2, 3.3, 1.1, 2.2, 3.3],
            bucketized: [
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


def test_bucketize_lit_list_eval(df_xy_list):
    new_df = df_xy_list.select(pl.all().list.eval(ti.bucketize_lit(1, 0)))
    expected = pl.DataFrame(
        {
            "x": [[1, 0, 1, 0], [1, 0, 1, 0]],
            "y": [[1, 0, 1, 0], [1, 0, 1, 0]],
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
    name = "bucketized"
    new_df = df_n.select(ti.bucketize(*exprs).alias(name))
    expected = pl.DataFrame({name: result})
    assert_frame_equal(new_df, expected)


def test_bucketize_list_eval(df_xy_list):
    new_df = df_xy_list.select(
        pl.all().list.eval(ti.bucketize(pl.element().add(10), pl.lit(100)))
    )
    expected = pl.DataFrame(
        {
            "x": [[11, 100, 13, 100], [15, 100, 17, 100]],
            "y": [[19, 100, 21, 100], [23, 100, 25, 100]],
        }
    )

    assert_frame_equal(new_df, expected)


@pytest.mark.parametrize(
    "exprs, result, return_dtype",
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
def test_bucketize_return_dtype(df_n, exprs, result, return_dtype):
    name = "bucketized"
    new_df = df_n.select(
        ti.bucketize(*exprs, return_dtype=return_dtype).alias(name)
    )
    expected = pl.DataFrame({name: result})

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
    name = "bool_nth_row"
    expr = ti.is_every_nth_row(n)
    assert expr.meta.output_name() == name

    new_df = df_n.select(expr)
    expected = pl.DataFrame({name: s_bool})

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


def test_is_every_nth_row_list_eval(df_xy_list):
    new_df = df_xy_list.select(pl.all().list.eval(ti.is_every_nth_row(2)))
    expected = pl.DataFrame(
        {
            "x": [[True, False, True, False], [True, False, True, False]],
            "y": [[True, False, True, False], [True, False, True, False]],
        }
    )

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


def test_cycle_pl_all(df_xy):
    new_df = df_xy.with_columns(ti.cycle(pl.all()))
    expected = pl.DataFrame({"x": [4, 1, 2, 3], "y": [8, 5, 6, 7]})
    assert_frame_equal(new_df, expected)


def test_cycle_list_eval(df_xy_list):
    new_df = df_xy_list.select(pl.all().list.eval(ti.cycle(pl.element(), 2)))
    expected = pl.DataFrame(
        {
            "x": [[3, 4, 1, 2], [7, 8, 5, 6]],
            "y": [[11, 12, 9, 10], [15, 16, 13, 14]],
        }
    )

    assert_frame_equal(new_df, expected)


def test__make_concat_str():
    """
    The quick brown fox jumps over the lazy dog.
    """
    fox, dog = "fox", "dog"
    df = pl.DataFrame({"fox": [fox], "dog": [dog]})
    quick, lazy = "quick", "lazy"
    concat_str_expr = _make_concat_str(
        f"The {quick} brown [$X] jumps over the {lazy} [$X].",
        fox,
        dog,
        sep="[$X]",
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

    assert_frame_equal(df.select(concat_str_expr), df.select(expected))


def test_make_concat_str():
    """
    The quick brown fox jumps over the lazy dog.
    """
    fox, dog = "fox", "dog"
    df = pl.DataFrame({"fox": [fox], "dog": [dog]})
    quick, lazy = "quick", "lazy"
    concat_str_expr = ti.make_concat_str(
        f"The {quick} brown [$X] jumps over the {lazy} [$X].",
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

    # Since `.alias()` is now added at the end of `make_concat_str()`,
    # the following assertion will no longer hold(checked by `test__make_concat_str()`):
    # assert concat_str_expr.meta.eq(expected)

    # Instead, verify the actual result
    assert_frame_equal(df.select(concat_str_expr), df.select(expected))


def test_make_concat_str_sep():
    """
    The quick brown fox jumps over the lazy dog.
    """
    name = "literal"
    fox, dog = "fox", "dog"
    df = pl.DataFrame({"fox": [fox], "dog": [dog]})
    quick, lazy = "quick", "lazy"
    concat_str_expr = ti.make_concat_str(
        f"The {quick} brown ##<X>## jumps over the {lazy} ##<X>##.",
        fox,
        dog,
        sep="##<X>##",
    )
    assert concat_str_expr.meta.output_name() == name

    expected = pl.concat_str(
        [
            pl.lit(f"The {quick} brown "),
            fox,
            pl.lit(f" jumps over the {lazy} "),
            dog,
            pl.lit("."),
        ]
    )

    assert_frame_equal(df.select(concat_str_expr), df.select(expected))


@pytest.mark.parametrize("name", [("name")])
def test_make_concat_alias(name):
    """
    The quick brown fox jumps over the lazy dog.
    """
    fox, dog = "fox", "dog"
    quick, lazy = "quick", "lazy"
    concat_str_expr = ti.make_concat_str(
        f"The {quick} brown [$X] jumps over the {lazy} [$X].",
        fox,
        dog,
        name=name,
    )

    assert concat_str_expr.meta.output_name() == name


def test_make_concat_str_complex():
    """
    Turtle floats gently across the sunlit bay.
    1. Test for column names located at the beginning and end.
    2. Test two consecutive column names.
    """
    name = "turtle"
    df = pl.DataFrame(
        {"animal": ["turtle"], "location": ["sunlit bay"], "period": ["."]}
    )
    concat_str_expr = ti.make_concat_str(
        "[$X] floats gently across the [$X][$X]",
        "animal",
        "location",
        "period",
        name=name,
    )
    expected = pl.concat_str(
        "animal", pl.lit(" floats gently across the "), "location", "period"
    ).alias(name)

    assert concat_str_expr.meta.eq(expected)

    assert_frame_equal(df.select(concat_str_expr), df.select(expected))


def test_make_concat_str_raise_col_names_not_all_str():
    fox = "fox"
    with pytest.raises(ValueError) as exc_info:
        ti.make_concat_str(
            "The quick brown [$X] jumps over the lazy [$X].", fox, 123
        )  # 123 is int type

    assert "All column names must be of type string." in exc_info.value.args[0]


def test_make_concat_str_raise_params_not_match():
    fox = "fox"
    with pytest.raises(ValueError) as exc_info:
        ti.make_concat_str(
            "The quick brown [$X] jumps over the lazy [$X].",
            fox,
        )  # `dog` is missed

    assert (
        "which does not match the number of column names"
        in exc_info.value.args[0]
    )
