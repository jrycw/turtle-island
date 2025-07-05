from collections.abc import Collection
from typing import TypeVar

import polars as pl
from polars._typing import PolarsDataType

from ._utils import _cast_datatype, _litify

T = TypeVar("T")


def case_when(
    caselist: list[tuple[pl.Expr, pl.Expr]], otherwise: pl.Expr | None = None
) -> pl.Expr:
    """
    Simplifies conditional logic in Polars by chaining multiple `when-then` expressions.

    Parameters
    ----------
    caselist
        A list of (condition, value) pairs. Each condition is evaluated in order,
        and the corresponding value is returned when a condition is met.
    otherwise
        The fallback value to use if none of the conditions match.

    Returns
    -------
    pl.Expr

    Examples
    -------
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
    Parameters
    ----------
    name
        Name of the index column.

    Returns
    -------
    pl.Expr

    Examples
    -------
    ```{python}
    import polars as pl
    import turtle_island as ti

    df = pl.DataFrame({"a": [1, 3, 5], "b": [2, 4, 6]})
    df.select(ti.create_index(), pl.all())
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

    References
    -------
    https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.with_row_index.html
    """
    return pl.int_range(pl.len(), dtype=pl.UInt32).alias(name)


def bucketize(*items: T, name: str = "bucketized") -> pl.Expr:
    """
    Return a Polars expression that bucketizes rows by index, cycling through the given items.

    Parameters
    ----------
    items
        Values to assign in a round-robin fashion based on the row index.
        All items must be of the same type, and at least two must be provided.
    name
        The name of the resulting column. Defaults to "bucketized".

    Returns
    -------
    pl.Expr

    Examples
    -------
    ```{python}
    import polars as pl
    import turtle_island as ti

    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    df.with_columns(ti.bucketize("a", "b"))
    ```
    shape: (5, 2)
    ┌─────┬────────────┐
    │ a   ┆ bucketized │
    │ --- ┆ ---        │
    │ i64 ┆ str        │
    ╞═════╪════════════╡
    │ 1   ┆ a          │
    │ 2   ┆ b          │
    │ 3   ┆ a          │
    │ 4   ┆ b          │
    │ 5   ┆ a          │
    └─────┴────────────┘
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


def move_cols_to_start(
    columns: str | PolarsDataType | Collection[str] | Collection[PolarsDataType],
) -> list[pl.Expr]:
    """
    Returns a Polars expression that reorders the DataFrame by moving the selected columns to the start.

    Parameters
    ----------
    columns
        The name or datatype of the column(s) to exclude. Accepts regular expression input.
        Regular expressions should start with `^` and end with `$`.

    Returns
    -------
    list[pl.Expr]

    Examples
    -------
    ```{python}
    import polars as pl
    import turtle_island as ti

    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [4.4, 5.5, 6.6]})
    df.select(ti.move_cols_to_start(["b", "c"]))
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

    ```{python}
    df.select(ti.move_cols_to_start([pl.Float64, pl.String]))
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
    """
    return [pl.col(columns), pl.all().exclude(columns)]


def move_cols_to_end(
    columns: str | PolarsDataType | Collection[str] | Collection[PolarsDataType],
) -> list[pl.Expr]:
    """
    Returns a Polars expression that reorders the DataFrame by moving the selected columns to the end.

    Parameters
    ----------
    columns
        The name or datatype of the column(s) to exclude. Accepts regular expression input.
        Regular expressions should start with `^` and end with `$`.

    Returns
    -------
    list[pl.Expr]

    Examples
    -------
    ```{python}
    import polars as pl
    import turtle_island as ti

    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [4.4, 5.5, 6.6]})
    df.select(ti.move_cols_to_end(["a", "b"]))
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

    ```{python}
    df.select(ti.move_cols_to_end([pl.String, pl.Int64]))
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
    """
    return [pl.all().exclude(columns), pl.col(columns)]


def is_nth_row(n: int, *, name: str = "is_nth_row") -> pl.Expr:
    return create_index(name=name).mod(n).eq(0)


def is_not_nth_row(n: int, *, name: str = "is_not_nth_row") -> pl.Expr:
    return is_nth_row(n, name=name).not_()
