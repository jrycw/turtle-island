from collections.abc import Collection
from typing import Any

import polars as pl
from polars._typing import PolarsDataType

from .._utils import _cast_datatype, _litify, _get_unique_name

__all__ = [
    "bucketize",
    "bucketize_lit",
    "case_when",
    "make_index",
    "is_every_nth_row",
    "move_cols_to_end",
    "move_cols_to_start",
]


def case_when(
    caselist: list[tuple[pl.Expr, pl.Expr]],
    otherwise: pl.Expr | None = None,
) -> pl.Expr:
    """
    Simplifies conditional logic in Polars by chaining multiple `when-then-otherwise` expressions.

    Inspired by [pd.Series.case_when()](https://pandas.pydata.org/docs/reference/api/pandas.Series.case_when.html),
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
    """
    from polars.expr.whenthen import Then

    (first_when, first_then), *cases = caselist

    # first
    expr: Then = pl.when(first_when).then(first_then)

    # middles
    for when, then in cases:
        expr: Then = expr.when(when).then(then)  # type: ignore[no-redef]

    # last
    expr: pl.Expr = expr.otherwise(otherwise)  # type: ignore[no-redef]

    return expr


def _make_index(start: int, end: int | pl.Expr, *, name: str) -> pl.Expr:
    return pl.int_range(start, end, dtype=pl.UInt32).alias(name)


def make_index(name: str = "index", offset: int = 0) -> pl.Expr:
    """
    Returns a Polars expression that creates a virtual row index.

    Borrowed from the [Polars documentation](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.with_row_index.html)
    and adapted for expression-level use.

    Unlike [pl.DataFrame.with_row_index()](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.with_row_index.html), which works at the DataFrame level,
    this expression can be composed inline and reused without materializing an actual column.

    Parameters
    ----------
    name
        The name to assign to the generated index column.
    offset
        Start the index at this offset. Cannot be negative.

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
    df.select(ti.make_index(), pl.all())
    ```
    """
    return _make_index(offset, pl.len() + offset, name=name)


def _make_bucketize_casewhen(
    exprs: Collection[Any], *, is_litify: bool, name: str
) -> pl.Expr:
    if is_litify:
        # turn items into exprs
        exprs: list[pl.Expr] = _litify(exprs)  # type: ignore[no-redef]
    n = len(exprs)
    mod_expr = make_index(name=_get_unique_name()).mod(n)
    *whenthen_exprs, otherwise_expr = exprs
    caselist: list[tuple[pl.Expr, pl.Expr]] = [
        (mod_expr.eq(i), expr) for i, expr in enumerate(whenthen_exprs)
    ]
    return case_when(caselist, otherwise_expr).alias(name)


def bucketize_lit(
    *items: Any,
    coalesce_to: pl.DataType | None = None,
    name: str = "bucketized",
) -> pl.Expr:
    """
    Returns a Polars expression that assigns a label to each row based on its index, cycling through the provided items in a round-robin fashion.

    `bucketize_lit()` is a simplified version of
    [bucketize()](bucketize.html#turtle_island.bucketize), designed for common
    use cases involving literal values. For more advanced scenarios, consider using
    `bucketize()` directly.

    Parameters
    ----------
    items
        Literal values to cycle through. All items must be of the same type,
        and at least two must be provided. See the table below for supported
        types and their conversions.
    coalesce_to
        An optional Polars data type to cast the resulting expression to.
    name
        The name of the resulting column. Defaults to "bucketized".

    Returns
    -------
    pl.Expr
        A Polars expression that cycles through the provided values based on the row index modulo.

    Supported Type Conversions
    --------------------------
    | Python Type          | Converted To       |
    |----------------------|--------------------|
    | `bool`               | `pl.Boolean`       |
    | `datetime.datetime`  | `pl.Datetime`      |
    | `datetime.date`      | `pl.Date`          |
    | `datetime.time`      | `pl.Time`          |
    | `datetime.timedelta` | `pl.Duration`      |
    | `int`                | `pl.Int64`         |
    | `float`              | `pl.Float64`       |
    | `str`                | `pl.String`        |
    | `list`, `tuple`      | `pl.List`          |
    | Others               | no cast involved   |

    Examples
    -------
    Cycle through boolean values to mark alternating rows:
    ```{python}
    import polars as pl
    import turtle_island as ti

    df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
    df.with_columns(ti.bucketize_lit(True, False))
    ```
    Cast the result to a specific data type using `coalesce_to=`:
    ```{python}
    df.with_columns(ti.bucketize_lit(True, False, coalesce_to=pl.Int64))
    ```
    """
    if len(items) <= 1:
        raise ValueError("`items=` must contain a minimum of two items.")
    if len(set(type(item) for item in items)) != 1:
        raise ValueError("`items=` must contain only one unique type.")
    expr = _make_bucketize_casewhen(items, is_litify=True, name=name)
    if coalesce_to is not None:
        return expr.cast(coalesce_to)
    return _cast_datatype(expr, items[0])


def bucketize(
    *exprs: pl.Expr,
    coalesce_to: pl.DataType | None = None,
    name: str = "bucketized",
) -> pl.Expr:
    """
    Returns a Polars expression that assigns a label to each row based on its index, cycling through the provided expressions in a round-robin fashion.

    `bucketize()` is the more general form of
    [bucketize_lit()](bucketize_lit.html#turtle_island.bucketize_lit),
    allowing you to pass Polars expressions instead of just literal values.
    This enables advanced use cases such as referencing or transforming
    existing column values.

    ::: {.callout-warning}
    ### Be cautious when using `pl.lit()` as the first expression

    Polars will automatically infer the data type of `pl.lit()`. For example, `pl.lit(1)` is inferred as `pl.Int32`.

    To avoid unexpected type mismatches, it's recommended to explicitly set the desired data type using `coalesce_to=`.
    :::

    Parameters
    ----------
    exprs
        A list of Polars expressions to cycle through. All expressions must resolve
        to the same data type. At least two expressions must be provided.
    coalesce_to
        An optional Polars data type to cast the resulting expression to.
    name
        The name of the resulting column. Defaults to "bucketized".

    Returns
    -------
    pl.Expr
        A Polars expression that cycles through the input expressions based on the row index modulo.

    Examples
    -------
    Alternate between a column expression and a literal value:
    ```{python}
    import polars as pl
    import turtle_island as ti

    df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
    df.with_columns(ti.bucketize(pl.col("x").add(10), pl.lit(100)))
    ```
    This alternates between the values of `x + 10` and the literal `100`.
    Make sure all expressions resolve to the same type—in this case, integers.

    You can also cast the result to a specific type using `coalesce_to=`:
    ```{python}
    df.with_columns(
        ti.bucketize(pl.col("x").add(10), pl.lit(100), coalesce_to=pl.String)
    )
    ```
    """
    if len(exprs) <= 1:
        raise ValueError("`exprs=` must contain a minimum of two expressions.")
    expr = _make_bucketize_casewhen(exprs, is_litify=False, name=name)
    if coalesce_to is not None:
        return expr.cast(coalesce_to)
    return expr


def is_every_nth_row(
    n: int, offset: int = 0, *, name: str = "bool_nth_row"
) -> pl.Expr:
    """
    Returns a Polars expression that is `True` for every `n`-th row (index modulo `n` equals 0).

    `is_every_nth_row()` can be seen as the complement of [pl.Expr.gather_every()](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.gather_every.html).

    While `pl.Expr.gather_every()` is typically used in a `select()` context and may return a
    DataFrame with fewer rows, `is_every_nth_row()` produces a predicate expression
    that can be used with `select()` or `with_columns()` to preserve the original row structure for
    further processing, or with `filter()` to achieve the same result as
    `pl.Expr.gather_every()`.

    ::: {.callout-warning}
    ### Ensure `offset=` does not exceed the total number of rows

    Since expressions are only evaluated at runtime, their validity cannot be
    checked until execution. If `offset=` is greater than the number of rows
    in the DataFrame, it may result in unexpected behavior.
    :::

    Parameters
    ----------
    n
        The interval to use for row selection. Should be positive.
    offset
        Start the index at this offset. Cannot be negative.
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
    df.with_columns(ti.is_every_nth_row(2))
    ```
    To invert the result, use either the `~` operator or `pl.Expr.not_()`:
    ```{python}
    df.with_columns(
        ~ti.is_every_nth_row(2).alias("~2"),
        ti.is_every_nth_row(2).not_().alias("not_2"),
    )
    ```
    Use `offset=` to shift the starting index:
    ```{python}
    df.with_columns(ti.is_every_nth_row(3, 1))
    ```
    For reference, here’s the output using `pl.Expr.gather_every()`:
    ```{python}
    df.select(pl.col("x").gather_every(3, 1))
    ```
    You can also combine multiple `is_every_nth_row()` expressions to construct more complex row selections.
    For example, to select rows that are part of every second **or** every third row:
    ```{python}
    df.select(
        ti.is_every_nth_row(2).alias("2"),
        ti.is_every_nth_row(3).alias("3"),
        ti.is_every_nth_row(2).or_(ti.is_every_nth_row(3)).alias("2_or_3")
    )
    ```
    """
    if n <= 0:
        raise ValueError("`n=` should be positive.")
    if offset < 0:
        raise ValueError("`offset=` cannot be negative.")

    offset_rows = pl.repeat(False, n=offset, dtype=pl.Boolean)
    rest_rows = (
        _make_index(0, pl.len() - offset, name=_get_unique_name()).mod(n).eq(0)
    )
    return offset_rows.append(rest_rows).alias(name)


def move_cols_to_start(
    columns: str
    | PolarsDataType
    | Collection[str]
    | Collection[PolarsDataType],
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

    df = pl.DataFrame(
        {"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [4.4, 5.5, 6.6]}
    )
    df.select(ti.move_cols_to_start(["b", "c"]))
    ```
    Or by data type:
    ```{python}
    df.select(ti.move_cols_to_start([pl.Float64, pl.String]))
    ```
    """
    return [pl.col(columns), pl.all().exclude(columns)]


def move_cols_to_end(
    columns: str
    | PolarsDataType
    | Collection[str]
    | Collection[PolarsDataType],
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
    """
    return [pl.all().exclude(columns), pl.col(columns)]
