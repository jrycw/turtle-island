from typing import Sequence

import polars as pl

from .._utils import _get_unique_name
from .core import make_index

__all__ = ["bulk_append", "case_when", "prepend", "shift"]


def case_when(
    case_list: Sequence[tuple[pl.Expr | tuple[pl.Expr], pl.Expr]],
    otherwise: pl.Expr | None = None,
) -> pl.Expr:
    """
    Simplifies conditional logic in Polars by chaining multiple `when‑then‑otherwise` expressions.

    Inspired by [pd.Series.case_when()](https://pandas.pydata.org/docs/reference/api/pandas.Series.case_when.html), this function offers a more ergonomic way to express chained
    conditional logic with Polars expressions.

    ::: {.callout-warning}
    ### Keyword shortcut is not supported
    Passing multiple keyword arguments as equality conditions—such as `x=123` in
    `pl.when()`—is not supported in this function.
    :::

    Parameters
    ----------
    case_list
        A sequence of tuples where each tuple represents a `when` and `then`
        branch. This function accepts three input forms (see examples below).
        Each tuple is evaluated in order from top to bottom. For each tuple, the
        expressions before the final element are treated as `when` conditions and
        combined with `&`. If the combined condition evaluates
        to `True`, the corresponding `then` expression (the last element) is returned
        and the evaluation stops. If no condition matches any tuple, the
        `otherwise` expression is used as the fallback.

    otherwise
        Fallback expression used when no conditions match.

    Returns
    -------
    pl.Expr
        A single Polars expression suitable for use in transformations.

    Examples
    --------
    The example below demonstrates all three supported input forms.

    `expr1` uses the simplest form, where each tuple contains a single `when`
    condition followed by its corresponding `then` expression.

    `expr2` shows tuples with multiple `when` conditions listed before the final
    `then` expression. These conditions are implicitly combined with `&`.

    `expr3` uses a tuple as the first element of each tuple, containing multiple
    `when` conditions which are also combined with `&` before evaluation.
    ```{python}
    import polars as pl
    import turtle_island as ti

    df = pl.DataFrame({"x": [1, 2, 3, 4], "y": [5, 6, 7, 8]})

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

    df.with_columns(expr1, expr2, expr3)
    ```
    """

    from polars.expr.whenthen import Then

    first_case, *cases = case_list

    # first
    *first_whens, first_then = first_case
    expr: Then = pl.when(*first_whens).then(first_then)

    # middles
    for case in cases:
        *whens, then = case
        expr: Then = expr.when(*whens).then(then)  # type: ignore[no-redef]

    # last
    expr: pl.Expr = expr.otherwise(otherwise)  # type: ignore[no-redef]

    return expr


def bulk_append(*exprs: pl.Expr) -> pl.Expr:
    """
    Combines multiple Polars expressions using [pl.Expr.append()](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.append.html#polars-expr-append) internally.

    ::: {.callout-tip}
    ### Rechunk

    You may consider rechunking the result of `bulk_append()` using
    [pl.Expr.rechunk()](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.rechunk.html#polars.Expr.rechunk)
    for better performance.
    :::

    Parameters
    ----------
    exprs
        One or more Polars expressions to be appended in sequence.

    Returns
    -------
    pl.Expr
        A single Polars expression resulting from appending all input expressions.

    Examples
    -------
    ```{python}
    import polars as pl
    import turtle_island as ti

    df = pl.DataFrame(
        {"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [4.4, 5.5, 6.6]}
    )
    df.select(ti.bulk_append(pl.all().last(), pl.all().first()))
    ```
    """
    if len(exprs) <= 1:
        raise ValueError("At least two Polars expressions must be provided.")
    expr, *rest_exprs = exprs
    for _expr in rest_exprs:
        expr = expr.append(_expr)
    return expr


def _get_case_list(
    expr: pl.Expr, offset: int
) -> list[tuple[pl.Expr, pl.Expr]]:
    shifted_expr = expr.shift(offset)
    index_expr = make_index(name=_get_unique_name())
    if offset > 0:
        # n is positive => pre_filled
        case_list = [(index_expr.ge(offset), shifted_expr)]
    else:
        # n is negative => back_filled
        case_list = [(index_expr.lt(pl.len() + offset), shifted_expr)]
    return case_list


def shift(expr: pl.Expr, offset: int = 1, *, fill_expr: pl.Expr) -> pl.Expr:
    """
    A variant of [pl.Expr.shift()](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.shift.html#polars.Expr.shift) that allows filling shifted values using another Polars expression.

    ::: {.callout-warning}
    ### Note: When `abs(offset)` exceeds the total number of rows

    Since expressions are evaluated lazily at runtime, their validity cannot be
    verified during construction. If `abs(offset)` equals or exceeds the total row count, the result
    may behave like a full-column replacement using `fill_expr=`.
    :::

    Parameters
    ----------
    expr
        A single Polars expression to shift.

    offset
        The number of rows to shift. It must be a non-zero integer.
        A positive value shifts the column downward (forward), while a negative value shifts it upward (backward).

    fill_expr
        Expression used to fill the shifted positions.

    Returns
    -------
    pl.Expr
        A Polars expression with shifted values and custom fill logic.

    Examples
    -------
    Shift values downward by 2:
    ```{python}
    import polars as pl
    import turtle_island as ti

    df = pl.DataFrame({"x": [1, 2, 3, 4], "y": [5, 6, 7, 8]})
    df.with_columns(
        ti.shift(pl.col("x"), 2, fill_expr=pl.col("y")).alias("shifted")
    )
    ```
    Shift values upward by 3:
    ```{python}
    df.with_columns(
        ti.shift(pl.col("x"), -3, fill_expr=pl.col("y")).alias("shifted")
    )
    ```
    """
    if not isinstance(offset, int):
        raise ValueError("`offset=` must be an integer.")
    if offset == 0:
        return expr
    case_list = _get_case_list(expr, offset)
    return case_when(case_list, fill_expr)


def prepend(
    expr: pl.Expr, offset: int = 1, *, prepend_expr: pl.Expr | None = None
) -> pl.Expr:
    """
    Returns a Polars expression that prepends rows using the `prepend_expr=`
    parameter. If `prepend_expr=` is not provided, the first row(s) of the
    current DataFrame—or Series if used within the `pl.List` namespace—will be
    used by default, based on the `offset=` value.

    ::: {.callout-caution}
    ### Unstable

    `prepend()` is an unconventional operation in Polars. In most cases, similar
    behavior should be achieved using `expr1.append(expr2)`, or via
    [pl.concat()](https://docs.pola.rs/api/python/stable/reference/api/polars.concat.html),
    [pl.DataFrame.vstack()](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.vstack.html#polars.DataFrame.vstack),
    or [pl.DataFrame.extend()](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.extend.html#polars.DataFrame.extend).

    This function was originally introduced to support quick insertion of
    elements in the `pl.List` namespace.

    Keep in mind that `prepend()` dynamically depends on the total number of
    rows, which sets the upper limit on how many rows can be prepended. If
    `offset=` exceeds the current row count, the result may resemble a
    duplication of the original column.

    If you need to call `prepend()` multiple times, consider doing so manually for now.
    :::

    Parameters
    ----------
    expr
        A Polars expression to which rows will be prepended.

    offset
        Number of rows to prepend. Must be a positive integer.

    prepend_expr
        The expression to prepend.

    Returns
    -------
    pl.Expr
        A Polars expression with prepended values.

    Examples
    -------
    Prepend one row using the default behavior:
    ```{python}
    import polars as pl
    import turtle_island as ti

    pl.Config.set_fmt_table_cell_list_len(10)
    df = pl.DataFrame({"x": [1, 2, 3, 4], "y": [5, 6, 7, 8]})
    df.select(ti.prepend(pl.all()))
    ```
    Prepend two rows using a custom expression:
    ```{python}
    df.select(
        ti.prepend(pl.all(), offset=2, prepend_expr=pl.col("x").mul(pl.col("y")))
    )
    ```
    `prepend()` can also be used inside `.list.eval()`. For example, prepend two elements inside a list column:
    ```{python}
    df2 = pl.DataFrame(
        {
            "x": [[1, 2, 3, 4], [5, 6, 7, 8]],
            "y": [[9, 10, 11, 12], [13, 14, 15, 16]],
        }
    )
    df2.select(pl.all().list.eval(ti.prepend(pl.element(), offset=2)))
    ```
    Prepend three elements using a custom expression:
    ```{python}
    df2.select(
        pl.all().list.eval(
            ti.prepend(pl.element(), offset=3, prepend_expr=pl.element().add(10))
        )
    )
    ```
    """
    if offset < 0:
        raise ValueError("`offset=` cannot be negative.")
    if offset == 0:
        return expr
    if prepend_expr is None:
        prepend_expr: pl.Expr = expr  # type: ignore[no-redef]
    case_list = _get_case_list(expr, offset)
    return case_when(case_list, prepend_expr).head(offset).append(expr)
