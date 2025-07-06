from collections.abc import Collection
from typing import TypeVar

import polars as pl
from polars._typing import PolarsDataType

from .._utils import _cast_datatype, _litify

T = TypeVar("T")


def case_when(
    caselist: list[tuple[pl.Expr, pl.Expr]], otherwise: pl.Expr | None = None
) -> pl.Expr:
    """
    Simplifies conditional logic in Polars by chaining multiple `when-then-otherwise` expressions.

    Inspired by [pandas.Series.case_when](https://pandas.pydata.org/docs/reference/api/pandas.Series.case_when.html),
    this provides a more ergonomic way to express chained conditional logic in Polars.

    Parameters
    ----------
    caselist
        A list of (condition, value) expression pairs. Conditions are evaluated in order.
    otherwise
        Fallback expression used when no conditions match.

    Returns
    -------
    pl.Expr
        A single Polars expression that can be used in transformation contexts.

    Examples
    -------
    `expr_ti` is constructed by `case_when()`, which should be equivalent to `expr_pl`.
    When evaluated in a Polars context (e.g., `with_columns()`), both expressions will produce the same result.
    ```{python}
    import polars as pl
    import turtle_island as ti

    df = pl.DataFrame({"x": [1, 2, 3, 4]})

    expr_ti = ti.case_when(
        caselist=[(pl.col("x") < 2, pl.lit("small")),
                  (pl.col("x") < 4, pl.lit("medium"))],
        otherwise=pl.lit("large"),
    ).alias("size_ti")

    expr_pl = (
        pl.when(pl.col("x") < 2)
        .then(pl.lit("small"))
        .when(pl.col("x") < 4)
        .then(pl.lit("medium"))
        .otherwise(pl.lit("large"))
        .alias("size_pl")
    )

    df.with_columns(expr_ti, expr_pl)
    ```
    ```
    shape: (4, 3)
    ┌─────┬─────────┬─────────┐
    │ x   ┆ size_ti ┆ size_pl │
    │ --- ┆ ---     ┆ ---     │
    │ i64 ┆ str     ┆ str     │
    ╞═════╪═════════╪═════════╡
    │ 1   ┆ small   ┆ small   │
    │ 2   ┆ medium  ┆ medium  │
    │ 3   ┆ medium  ┆ medium  │
    │ 4   ┆ large   ┆ large   │
    └─────┴─────────┴─────────┘
    ```
    """
    (first_when, first_then), *cases = caselist

    # first
    expr = pl.when(first_when).then(first_then)

    # middles
    for when, then in cases:
        expr = expr.when(when).then(then)

    # last
    expr = expr.otherwise(otherwise)

    return expr


def create_index(name: str = "index") -> pl.Expr:
    """
    Returns a Polars expression that creates a virtual row index.

    Borrowed from the [Polars documentation](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.with_row_index.html)
    and adapted for expression-level use.

    Unlike `with_row_index()`, which works at the DataFrame level,
    this expression can be composed inline and reused without materializing an actual column.

    Parameters
    ----------
    name
        The name to assign to the generated index column.

    Returns
    -------
    pl.Expr
        A Polars expression that yields a sequential integer index starting from 0.

    Examples
    -------
    Adds a sequential index column to the DataFrame:
    ```{python}
    import polars as pl
    import turtle_island as ti

    df = pl.DataFrame({"a": [1, 3, 5], "b": [2, 4, 6]})
    df.select(ti.create_index(), pl.all())
    ```
    ```
    shape: (3, 3)
    ┌───────┬─────┬─────┐
    │ index ┆ a   ┆ b   │
    │ ---   ┆ --- ┆ --- │
    │ u32   ┆ i64 ┆ i64 │
    ╞═══════╪═════╪═════╡
    │ 0     ┆ 1   ┆ 2   │
    │ 1     ┆ 3   ┆ 4   │
    │ 2     ┆ 5   ┆ 6   │
    └───────┴─────┴─────┘
    ```
    """
    return pl.int_range(pl.len(), dtype=pl.UInt32).alias(name)


def bucketize(*items: T, name: str = "bucketized") -> pl.Expr:
    """
    Returns a Polars expression that assigns a label to each row based on its index,
    cycling through the provided items in a round-robin fashion.

    Parameters
    ----------
    items
        The values to cycle through. All items must be of the same type, and at least two must be given.
    name
        The name of the resulting column. Defaults to "bucketized".

    Returns
    -------
    pl.Expr
        A Polars expression that cycles through the values based on the row index modulo.

    Examples
    -------
    Useful for alternating values across rows by index:
    ```{python}
    import polars as pl
    import turtle_island as ti

    df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
    df.with_columns(ti.bucketize(True, False))
    ```
    ```
    shape: (5, 2)
    ┌─────┬────────────┐
    │ x   ┆ bucketized │
    │ --- ┆ ---        │
    │ i64 ┆ bool       │
    ╞═════╪════════════╡
    │ 1   ┆ true       │
    │ 2   ┆ false      │
    │ 3   ┆ true       │
    │ 4   ┆ false      │
    │ 5   ┆ true       │
    └─────┴────────────┘
    ```
    Here, rows are alternately labeled `True` and `False`.
    """
    if len(items) <= 1:
        raise ValueError(f"{items} must contain a minimum of two items.")
    if len(set(type(item) for item in items)) != 1:
        raise ValueError(f"{items} must contain only one unique type.")
    n = len(items)
    mod_expr = create_index().mod(n)
    *litified, litified_otherwise = _litify(items)
    caselist = [(mod_expr.eq(i), lit) for i, lit in enumerate(litified)]
    expr = case_when(caselist, litified_otherwise).alias(name)
    return _cast_datatype(expr, items[0])


def is_nth_row(n: int, *, name: str = "bool_nth_row") -> pl.Expr:
    """
    Returns a Polars expression that is `True` for every `n`-th row (index modulo `n` equals 0).

    Parameters
    ----------
    n
        The interval to use for row selection.
    name
        The alias name for the resulting column.

    Returns
    -------
    pl.Expr
        A boolean Polars expression. Defaults to "bool_nth_row".

    Examples
    -------
    Mark every second row:
    ```{python}
    import polars as pl
    import turtle_island as ti

    df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
    df.with_columns(ti.is_nth_row(2))
    ```
    ```
    shape: (5, 1)
    ┌──────────────┐
    │ bool_nth_row │
    │ ---          │
    │ bool         │
    ╞══════════════╡
    │ true         │
    │ false        │
    │ true         │
    │ false        │
    │ true         │
    └──────────────┘
    ```
    To invert the result:
    ```{python}
    df.with_columns(~ti.is_nth_row(2))
    ```
    ```
    shape: (5, 1)
    ┌──────────────┐
    │ bool_nth_row │
    │ ---          │
    │ bool         │
    ╞══════════════╡
    │ false        │
    │ true         │
    │ false        │
    │ true         │
    │ false        │
    └──────────────┘
    ```
    """
    return create_index(name=name).mod(n).eq(0)


def move_cols_to_start(
    columns: str | PolarsDataType | Collection[str] | Collection[PolarsDataType],
) -> list[pl.Expr]:
    """
    Returns a list of Polars expressions that reorders columns to place the specified columns first.

    Parameters
    ----------
    columns
        The name or datatype of the column(s) to exclude. Accepts regular expression input.
        Regular expressions should start with `^` and end with `$`.

    Returns
    -------
    list[pl.Expr]
        A list of expressions to reorder columns.

    Examples
    -------
    Reorder columns so that selected columns appear first:
    ```{python}
    import polars as pl
    import turtle_island as ti

    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [4.4, 5.5, 6.6]})
    df.select(ti.move_cols_to_start(["b", "c"]))
    ```
    Or by data type:
    ```{python}
    df.select(ti.move_cols_to_start([pl.Float64, pl.String]))
    ```
    ```
    shape: (3, 3)
    ┌─────┬─────┬─────┐
    │ b   ┆ c   ┆ a   │
    │ --- ┆ --- ┆ --- │
    │ str ┆ f64 ┆ i64 │
    ╞═════╪═════╪═════╡
    │ x   ┆ 4.4 ┆ 1   │
    │ y   ┆ 5.5 ┆ 2   │
    │ z   ┆ 6.6 ┆ 3   │
    └─────┴─────┴─────┘
    ```
    """
    return [pl.col(columns), pl.all().exclude(columns)]


def move_cols_to_end(
    columns: str | PolarsDataType | Collection[str] | Collection[PolarsDataType],
) -> list[pl.Expr]:
    """
    Returns a list of Polars expressions that reorders columns to place the specified columns last.

    Parameters
    ----------
    columns
        The name or datatype of the column(s) to exclude. Accepts regular expression input.
        Regular expressions should start with `^` and end with `$`.

    Returns
    -------
    list[pl.Expr]
        A list of expressions to reorder columns.

    Examples
    -------
    Reorder columns so that selected columns appear last:
    ```{python}
    import polars as pl
    import turtle_island as ti

    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [4.4, 5.5, 6.6]})
    df.select(ti.move_cols_to_end(["a", "b"]))
    ```
    Or by data type:
    ```{python}
    df.select(ti.move_cols_to_end([pl.String, pl.Int64]))
    ```
    ```
    shape: (3, 3)
    ┌─────┬─────┬─────┐
    │ c   ┆ a   ┆ b   │
    │ --- ┆ --- ┆ --- │
    │ f64 ┆ i64 ┆ str │
    ╞═════╪═════╪═════╡
    │ 4.4 ┆ 1   ┆ x   │
    │ 5.5 ┆ 2   ┆ y   │
    │ 6.6 ┆ 3   ┆ z   │
    └─────┴─────┴─────┘
    ```
    """
    return [pl.all().exclude(columns), pl.col(columns)]
