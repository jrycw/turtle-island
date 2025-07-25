from collections.abc import Collection
from typing import Any

import polars as pl
from polars._typing import PolarsDataType

from .._utils import _cast_datatype, _get_unique_name, _litify
from .common import case_when, shift
from .core import make_index

__all__ = [
    "bucketize",
    "bucketize_lit",
    "cycle",
    "is_every_nth_row",
    "make_concat_str",
    "move_cols_to_end",
    "move_cols_to_start",
]


def _make_bucketize_casewhen(
    exprs: Collection[Any], *, is_litify: bool
) -> pl.Expr:
    if is_litify:
        # turn items into exprs
        exprs: list[pl.Expr] = _litify(exprs)  # type: ignore[no-redef]
    n = len(exprs)
    mod_expr = make_index(name=_get_unique_name()).mod(n)
    *whenthen_exprs, otherwise_expr = exprs
    case_list: list[tuple[pl.Expr, pl.Expr]] = [
        (mod_expr.eq(i), expr) for i, expr in enumerate(whenthen_exprs)
    ]
    return case_when(case_list, otherwise_expr)


def _get_move_cols(
    columns: str
    | PolarsDataType
    | Collection[str]
    | Collection[PolarsDataType],
    *more_columns: str | PolarsDataType,
) -> list[str] | list[PolarsDataType]:
    if not isinstance(columns, str) and isinstance(columns, Collection):
        _columns = [*columns, *more_columns]
    else:
        _columns = [columns, *more_columns]
    return _columns


def _make_concat_str(template: str, *col_names: str, sep: str) -> pl.Expr:
    if not all(isinstance(col_name, str) for col_name in col_names):
        raise ValueError("All column names must be of type string.")
    splitted = template.split(sep)
    len_splitted, len_col_names = len(splitted), len(col_names)
    if len_splitted != (len_col_names + 1):
        raise ValueError(
            f"The number of placeholders in the template is {len_splitted}, "
            f"which does not match the number of column names ({len_col_names})."
        )
    col_names_iter = iter(col_names)
    concat_str_list: list[pl.Expr | str] = []
    for s in splitted:
        if s:
            concat_str_list.append(pl.lit(s))
        if col_name := next(col_names_iter, None):
            concat_str_list.append(col_name)
    return pl.concat_str(concat_str_list)


def bucketize_lit(
    *items: Any, return_dtype: pl.DataType | pl.DataTypeExpr | None = None
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

    return_dtype
        An optional Polars data type to cast the resulting expression to.

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
    ### DataFrame Context

    Cycle through boolean values to mark alternating rows:
    ```{python}
    import polars as pl
    import turtle_island as ti

    pl.Config.set_fmt_table_cell_list_len(10)
    df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
    df.with_columns(ti.bucketize_lit(True, False).alias("bucketized"))
    ```
    Cast the result to a specific data type using `return_dtype=`:
    ```{python}
    df.with_columns(
        ti.bucketize_lit(True, False, return_dtype=pl.Int64).alias("bucketized")
    )
    ```
    ### List Namespace Context
    ::: {.callout-tip collapse="true"}
    ### Working with Lists as Series

    In the list namespace, it may be easier to think of each row as an
    element in a list. Conceptually, you're working with a `pl.Series`, where
    each row corresponds to one item in the list.
    :::

    Cycle through boolean values to mark alternating elements:
    ```{python}
    df2 = pl.DataFrame(
        {
            "x": [[1, 2, 3, 4], [5, 6, 7, 8]],
            "y": [[9, 10, 11, 12], [13, 14, 15, 16]],
        }
    )
    df2.with_columns(
        pl.all().list.eval(ti.bucketize_lit(True, False))
    )
    ```
    """
    if len(items) <= 1:
        raise ValueError("`items=` must contain a minimum of two items.")
    if len(set(type(item) for item in items)) != 1:
        raise ValueError("`items=` must contain only one unique type.")
    expr = _make_bucketize_casewhen(items, is_litify=True)
    if return_dtype is not None:
        return expr.cast(return_dtype)
    return _cast_datatype(expr, items[0])


def bucketize(
    *exprs: pl.Expr, return_dtype: pl.DataType | pl.DataTypeExpr | None = None
) -> pl.Expr:
    """
    Returns a Polars expression that assigns a label to each row based on its index, cycling through the provided expressions in a round-robin fashion.

    `bucketize()` is the more general form of
    [bucketize_lit()](bucketize_lit.html#turtle_island.bucketize_lit),
    allowing you to pass Polars expressions instead of just literal values.
    This enables advanced use cases such as referencing or transforming
    existing column values.

    ::: {.callout-warning collapse="true"}
    ### Be cautious when using `pl.lit()` as the first expression

    Polars will automatically infer the data type of `pl.lit()`. For example, `pl.lit(1)` is inferred as `pl.Int32`.

    To avoid unexpected type mismatches, it's recommended to explicitly set the desired data type using `return_dtype=`.
    :::

    Parameters
    ----------
    exprs
        Two or more Polars expressions to cycle through.
        All expressions must resolve to the same data type.

    return_dtype
        An optional Polars data type to cast the resulting expression to.

    Returns
    -------
    pl.Expr
        A Polars expression that cycles through the input expressions based on the row index modulo.

    Examples
    -------
    ### DataFrame Context

    Alternate between a column expression and a literal value:
    ```{python}
    import polars as pl
    import turtle_island as ti

    pl.Config.set_fmt_table_cell_list_len(10)
    df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
    df.with_columns(
        ti.bucketize(pl.col("x").add(10), pl.lit(100)).alias("bucketized")
    )
    ```
    This alternates between the values of `x + 10` and the literal `100`.
    Make sure all expressions resolve to the same type—in this case, integers.

    You can also cast the result to a specific type using `return_dtype=`:
    ```{python}
    df.with_columns(
        ti.bucketize(
            pl.col("x").add(10), pl.lit(100), return_dtype=pl.String
        ).alias("bucketized")
    )
    ```
    ### List Namespace Context
    ::: {.callout-tip collapse="true"}
    ### Working with Lists as Series

    In the list namespace, it may be easier to think of each row as an
    element in a list. Conceptually, you're working with a `pl.Series`, where
    each row corresponds to one item in the list.
    :::

    Alternate between a column expression and a literal value for each element:
    ```{python}
    df2 = pl.DataFrame(
        {
            "x": [[1, 2, 3, 4], [5, 6, 7, 8]],
            "y": [[9, 10, 11, 12], [13, 14, 15, 16]],
        }
    )
    (
    df2.with_columns(
            pl.all().list.eval(ti.bucketize(pl.element().add(10), pl.lit(100)))
        )
    )
    ```
    """
    if len(exprs) <= 1:
        raise ValueError("`exprs=` must contain a minimum of two expressions.")
    expr = _make_bucketize_casewhen(exprs, is_litify=False)
    if return_dtype is not None:
        return expr.cast(return_dtype)
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

    ::: {.callout-warning collapse="true"}
    ### Ensure `offset=` does not exceed the total number of rows

    Since expressions are only evaluated at runtime, their validity cannot be
    checked until execution. If `offset=` is greater than the number of rows
    in the DataFrame, the result will be a column filled with `False`.
    :::

    Parameters
    ----------
    n
        The interval to use for row selection. Should be positive.

    offset
        Start the index at this offset. Cannot be negative.

    name
        The name of the resulting column.

    Returns
    -------
    pl.Expr
        A boolean Polars expression.

    Examples
    -------
    ### DataFrame Context

    Mark every second row:
    ```{python}
    import polars as pl
    import turtle_island as ti

    pl.Config.set_fmt_table_cell_list_len(10)
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
    ### List Namespace Context
    ::: {.callout-tip collapse="true"}
    ### Working with Lists as Series

    In the list namespace, it may be easier to think of each row as an
    element in a list. Conceptually, you're working with a `pl.Series`, where
    each row corresponds to one item in the list.
    :::

    Mark every second element:
    ```{python}
    df2 = pl.DataFrame(
        {
            "x": [[1, 2, 3, 4], [5, 6, 7, 8]],
            "y": [[9, 10, 11, 12], [13, 14, 15, 16]],
        }
    )
    df2.with_columns(pl.all().list.eval(ti.is_every_nth_row(2)))
    ```
    """
    if n <= 0:
        raise ValueError("`n=` should be positive.")
    if offset < 0:
        raise ValueError("`offset=` cannot be negative.")

    return shift(
        make_index(name=_get_unique_name()).mod(n).eq(0),
        offset,
        fill_expr=pl.lit(False),
    ).alias(name)


def move_cols_to_start(
    columns: str
    | PolarsDataType
    | Collection[str]
    | Collection[PolarsDataType],
    *more_columns: str | PolarsDataType,
) -> list[pl.Expr]:
    """
    Returns a list of Polars expressions that reorder columns so the specified columns appear first.

    ::: {.callout-warning collapse="true"}
    ### Column type restriction

    You may specify either column names or data types, but not a combination of both.
    :::

    Parameters
    ----------
    columns
        The name or datatype of the column(s) to move. Accepts regular expression input.
        Regular expressions should start with `^` and end with `$`.

    *more_columns
        Additional names or datatypes of columns to move, specified as positional arguments.

    Returns
    -------
    list[pl.Expr]
        A list of expressions to reorder columns.

    Examples
    -------
    ### DataFrame Context

    ::: {.callout-caution collapse="true"}
    ### Works Only in `select()` Context

    The list of expressions returned by `move_cols_to_start()` take effect only within
    the `select()` context. Using them in `with_columns()` will have no effect,
    and the result will remain unchanged.
    :::

    ```{python}
    import polars as pl
    import turtle_island as ti

    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [4.4, 5.5, 6.6]})
    df
    ```
    Reorder columns so that selected columns appear first:
    ```{python}
    df.select(ti.move_cols_to_start("c", "b"))
    ```
    Reorder by data type:
    ```{python}
    df.select(ti.move_cols_to_start([pl.Float64, pl.String]))
    ```
    Note that when selecting by data type, the moved columns will follow the
    original order in the DataFrame schema.
    ```{python}
    df.select(ti.move_cols_to_start([pl.String, pl.Float64]))
    ```
    """
    _columns = _get_move_cols(columns, *more_columns)
    return [pl.col(_columns), pl.exclude(_columns)]


def move_cols_to_end(
    columns: str
    | PolarsDataType
    | Collection[str]
    | Collection[PolarsDataType],
    *more_columns: str | PolarsDataType,
) -> list[pl.Expr]:
    """
    Returns a list of Polars expressions that reorder columns so the specified columns appear last.

    ::: {.callout-warning collapse="true"}
    ### Column type restriction

    You may specify either column names or data types, but not a combination of both.
    :::

    Parameters
    ----------
    columns
        The name or datatype of the column(s) to move. Accepts regular expression input.
        Regular expressions should start with `^` and end with `$`.

    *more_columns
        Additional names or datatypes of columns to move, specified as positional arguments.

    Returns
    -------
    list[pl.Expr]
        A list of expressions to reorder columns.

    Examples
    -------
    ### DataFrame Context

    ::: {.callout-caution collapse="true"}
    ### Works Only in `select()` Context

    The list of expressions returned by `move_cols_to_end()` take effect only within
    the `select()` context. Using them in `with_columns()` will have no effect,
    and the result will remain unchanged.
    :::

    ```{python}
    import polars as pl
    import turtle_island as ti

    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [4.4, 5.5, 6.6]})
    df
    ```
    Reorder columns so that selected columns appear last:
    ```{python}
    df.select(ti.move_cols_to_end("b", "a"))
    ```
    Reorder by data type:
    ```{python}
    df.select(ti.move_cols_to_end([pl.String, pl.Int64]))
    ```
    Note that when selecting by data type, the moved columns will follow the
    original order in the DataFrame schema.
    ```{python}
    df.select(ti.move_cols_to_end([pl.Int64, pl.String]))
    ```
    """
    _columns = _get_move_cols(columns, *more_columns)
    return [pl.exclude(_columns), pl.col(_columns)]


def cycle(expr, offset: int = 1) -> pl.Expr:
    """
    Return a Polars expression that cycles the rows by a given offset.

    ::: {.callout-tip collapse="true"}
    ### Rechunk

    You may consider rechunking the result of `cycle()` using
    [pl.Expr.rechunk()](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.rechunk.html#polars.Expr.rechunk)
    for better performance.
    :::

    Parameters
    ----------
    expr
        A single Polars expression to apply the cycling operation on.

    offset
        The number of rows to cycle by. Positive values shift rows downward,
        and negative values shift rows upward.

    Returns
    -------
    pl.Expr
        A Polars expression with values cyclically shifted.

    Examples
    -------
    ### DataFrame Context

    Cycle downward by 2 rows:
    ```{python}
    import polars as pl
    import turtle_island as ti

    pl.Config.set_fmt_table_cell_list_len(10)
    df = pl.DataFrame({"x": [1, 2, 3, 4]})
    df.with_columns(ti.cycle(pl.col("x"), 2).alias("cycle"))
    ```
    Cycle upward by 4 rows (no visible change due to full cycle):
    ```{python}
    df.with_columns(ti.cycle(pl.col("x"), -4).alias("cycle"))
    ```
    ### List Namespace Context
    ::: {.callout-tip collapse="true"}
    ### Working with Lists as Series

    In the list namespace, it may be easier to think of each row as an
    element in a list. Conceptually, you're working with a `pl.Series`, where
    each row corresponds to one item in the list.
    :::

    Cycle downward by 2 elements:
    ```{python}
    df2 = pl.DataFrame(
        {
            "x": [[1, 2, 3, 4], [5, 6, 7, 8]],
            "y": [[9, 10, 11, 12], [13, 14, 15, 16]],
        }
    )
    df2.with_columns(pl.all().list.eval(ti.cycle(pl.element(), 2)))
    ```
    """
    if not isinstance(offset, int):
        raise ValueError("`offset=` must be an integer.")
    if offset == 0:
        return expr
    n = pl.len() - (offset % pl.len())
    return expr.slice(n).append(expr.head(n))


def make_concat_str(
    template: str, *col_names: str, sep: str = "[$X]", name: str = "literal"
) -> pl.Expr:
    """
    Construct a concatenated string expression by treating column names as placeholders within a template string.

    This function simplifies string concatenation by allowing users to insert column values into a string template using
    a custom separator (`sep=`). It’s particularly useful when constructing long strings like HTML content.

    Internally, `make_concat_str()` splits the `template=` string based on the `sep=` value, then interleaves the literals
    with the specified column names using [pl.concat_str()](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.concat_str.html).

    ::: {.callout-important collapse="false"}
    ### Unstable
    This function was originally intended for internal use and is now promoted to a public API. However, it is still
    experimental and may change in future versions.
    :::

    Parameters
    ----------
    template
        A string template where column placeholders are defined using the `sep=` string.
        Users are responsible for choosing a **safe** separator that won’t conflict with existing text.
        Common separators like "," or ";" may cause issues if they appear elsewhere in the template.

    col_names
        One or more column names to inject into the template string. These will be inserted at positions marked by `sep=`.

    sep
        The placeholder used to indicate where column names should be inserted within the template.

    name
        The name of the resulting column.

    Returns
    -------
    pl.Expr
        A Polars expression representing the final concatenated string.

    Examples
    --------
    ### DataFrame Context

    Here’s an example that builds an HTML `<p>` tag from a DataFrame:
    ```{python}
    import polars as pl
    import turtle_island as ti

    pl.Config.set_fmt_str_lengths(200)
    df = pl.DataFrame({"text": ["This is a simple paragraph of text."]})
    style = 'style="color: steelblue;"'
    new_df = df.with_columns(
        ti.make_concat_str(f"<p {style}>[$X]</p>", "text", name="p_tag")
    )
    new_df
    ```
    ```{python}
    new_df.style
    ```
    Did you notice that `style` is just a regular Python variable?

    We’re dynamically injecting it with an f-string before passing it to `make_concat_str()`.
    """
    return _make_concat_str(template, *col_names, sep=sep).alias(name)
