from collections.abc import Iterable
from typing import Sequence, overload

import polars as pl

from .._utils import _flatten_elems, _get_unique_name
from .core import make_index

__all__ = ["bulk_append", "case_when", "prepend", "shift"]


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


def case_when(
    case_list: Sequence[tuple[pl.Expr | tuple[pl.Expr], pl.Expr]],
    otherwise: pl.Expr | None = None,
) -> pl.Expr:
    """
    Simplifies conditional logic in Polars by chaining multiple `when-then-otherwise` expressions.

    Inspired by [pd.Series.case_when()](https://pandas.pydata.org/docs/reference/api/pandas.Series.case_when.html), this function offers a more ergonomic way to express chained
    conditional logic with Polars expressions.

    ::: {.callout-warning collapse="true"}
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
    ### DataFrame Context
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
    ### List Namespace Context
    ::: {.callout-tip collapse="true"}
    ### Working with Lists as Series

    In the list namespace, it may be easier to think of each row as an
    element in a list. Conceptually, you're working with a `pl.Series`, where
    each row corresponds to one item in the list.
    :::

    Check whether each string in the list starts with the letter "a" or "A":
    ```{python}
    df2 = pl.DataFrame(
        {
            "col1": [
                ["orange", "Lemon", "Kiwi"],
                ["Acerola", "Cherry", "Papaya"],
            ],
            "col2": [
                ["Grape", "Avocado", "apricot"],
                ["Banana", "apple", "Mango"],
            ],
        }
    )

    case_list = [
        (pl.element().str.to_lowercase().str.starts_with("a"), pl.lit("Y"))
    ]
    otherwise = pl.lit("N")

    (df2.with_columns(pl.all().list.eval(ti.case_when(case_list, otherwise))))
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


@overload
def bulk_append(*exprs: pl.Expr) -> pl.Expr: ...


@overload
def bulk_append(*exprs: Iterable[pl.Expr]) -> pl.Expr: ...


def bulk_append(*exprs: pl.Expr | Iterable[pl.Expr]) -> pl.Expr:
    """
    Combines multiple Polars expressions using [pl.Expr.append()](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.append.html#polars-expr-append) internally.

    ::: {.callout-tip collapse="true"}
    ### Rechunk

    You may consider rechunking the result of `bulk_append()` using
    [pl.Expr.rechunk()](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.rechunk.html#polars.Expr.rechunk)
    for better performance.
    :::

    Parameters
    ----------
    exprs
        One or more `pl.Expr` objects passed as separate arguments, or a single
        iterable containing multiple `pl.Expr` objects.

    Returns
    -------
    pl.Expr
        A single Polars expression resulting from appending all input expressions.

    Examples
    -------
    ### DataFrame Context
    ::: {.callout-caution collapse="true"}
    ### Caution When Used in `with_columns()` Context

    Because `bulk_append()` may change the total number of rows, use it with
    caution inside `with_columns()`.
    :::

    Append the last value to the first:
    ```{python}
    import polars as pl
    import turtle_island as ti

    df = pl.DataFrame(
        {"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [4.4, 5.5, 6.6]}
    )
    df.select(ti.bulk_append(pl.all().first(), pl.all().last()))
    ```
    ### List Namespace Context
    ::: {.callout-tip collapse="true"}
    ### Working with Lists as Series

    In the list namespace, it may be easier to think of each row as an
    element in a list. Conceptually, you're working with a `pl.Series`, where
    each row corresponds to one item in the list.
    :::

    A similar operation applies to lists, where the last element is appended to the first.
    ```{python}
    df2 = pl.DataFrame(
        {
            "x": [[1, 2, 3, 4], [5, 6, 7, 8]],
            "y": [[9, 10, 11, 12], [13, 14, 15, 16]],
        }
    )
    df2.select(
        pl.all().list.eval(
            ti.bulk_append(pl.element().first(), pl.element().last())
        )
    )
    ```
    """
    flatten_exprs = _flatten_elems(exprs)
    if len(flatten_exprs) <= 1:
        raise ValueError("At least two Polars expressions must be provided.")
    expr, *rest_exprs = flatten_exprs
    for _expr in rest_exprs:
        expr = expr.append(_expr)
    return expr


def shift(expr: pl.Expr, offset: int = 1, *, fill_expr: pl.Expr) -> pl.Expr:
    """
    A variant of [pl.Expr.shift()](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.shift.html#polars.Expr.shift) that allows filling shifted values using another Polars expression.

    ::: {.callout-warning collapse="true"}
    ### When `abs(offset)` exceeds the total number of rows

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
    ### DataFrame Context

    Shift values downward by 2:
    ```{python}
    import polars as pl
    import turtle_island as ti

    pl.Config.set_fmt_table_cell_list_len(10)
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
    ### List Namespace Context
    ::: {.callout-tip collapse="true"}
    ### Working with Lists as Series

    In the list namespace, it may be easier to think of each row as an
    element in a list. Conceptually, you're working with a `pl.Series`, where
    each row corresponds to one item in the list.
    :::

    Shift values downward by 2:
    ```{python}
    df2 = pl.DataFrame(
        {
            "x": [[1, 2, 3, 4], [5, 6, 7, 8]],
            "y": [[9, 10, 11, 12], [13, 14, 15, 16]],
        }
    )
    df2.with_columns(
        pl.all().list.eval(
            ti.shift(pl.element(), 2, fill_expr=pl.element().add(10))
        )
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
    Returns a Polars expression that prepends rows using the `prepend_expr=` parameter.

    If `prepend_expr=` is not provided, the first row(s) of the
    current DataFrame—or Series if used within the list namespace—will be
    used by default, based on the `offset=` value.

    ::: {.callout-important collapse="false"}
    ### Unconventional Operation

    `prepend()` is an unconventional operation in Polars. In most cases, similar
    behavior can be achieved using `expr1.append(expr2)`, or through
    [pl.concat()](https://docs.pola.rs/api/python/stable/reference/api/polars.concat.html),
    [pl.DataFrame.vstack()](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.vstack.html#polars.DataFrame.vstack),
    or
    [pl.DataFrame.extend()](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.extend.html#polars.DataFrame.extend).

    However, it provides a convenient way to insert elements quickly
    when working within the list namespace.
    :::

    ::: {.callout-warning collapse="true"}
    ### When `offset=` exceeds the total number of rows

    Keep in mind that `prepend()` dynamically depends on the total number of
    rows, which sets the upper limit on how many rows can be prepended. If
    `offset=` exceeds the current row count, the result may resemble a
    duplication of the original column.

    To prepend more rows than the total number of existing rows, you currently
    need to call `prepend()` multiple times manually.
    :::

    ::: {.callout-tip collapse="true"}
    ### Rechunk

    You may consider rechunking the result of `prepend()` using
    [pl.Expr.rechunk()](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.rechunk.html#polars.Expr.rechunk)
    for better performance.
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
        A Polars expression with the specified values prepended.

    Examples
    -------
    ### DataFrame Context

    ::: {.callout-caution collapse="true"}
    ### Cannot be used in the `with_columns()` context

    Because `prepend()` modifies the total number of rows, it cannot be used
    inside `with_columns()`.
    :::

    Prepend one row using the default behavior:
    ```{python}
    import polars as pl
    import turtle_island as ti

    pl.Config.set_fmt_table_cell_list_len(10)
    df = pl.DataFrame({"x": [1, 2, 3, 4], "y": [5, 6, 7, 8]})
    df.select(ti.prepend(pl.all()))
    ```
    Prepend two rows using a literal value:
    ```{python}
    df.select(
        ti.prepend(pl.all(), offset=2, prepend_expr=pl.lit(0))
    )
    ```
    Prepend two rows using a custom expression:
    ```{python}
    df.select(
        ti.prepend(pl.all(), offset=3, prepend_expr=pl.col("x").mul(pl.col("y")))
    )
    ```
    ### List Namespace Context
    ::: {.callout-tip collapse="true"}
    ### Working with Lists as Series

    In the list namespace, it may be easier to think of each row as an
    element in a list. Conceptually, you're working with a `pl.Series`, where
    each row corresponds to one item in the list.
    :::

    Prepend one element to each list:
    ```{python}
    df2 = pl.DataFrame(
        {
            "x": [[1, 2, 3, 4], [5, 6, 7, 8]],
            "y": [[9, 10, 11, 12], [13, 14, 15, 16]],
        }
    )
    df2.select(pl.all().list.eval(ti.prepend(pl.element())))
    ```
    Prepend two elements using a literal value:
    ```{python}
    df2.select(
        pl.all().list.eval(
            ti.prepend(pl.element(), offset=2, prepend_expr=pl.lit(0))
        )
    )
    ```
    Prepend three elements using a custom expression:
    ```{python}
    df2.select(
        pl.all().list.eval(
            ti.prepend(pl.element(), offset=3, prepend_expr=pl.element().add(10))
        )
    )
    ```
    Prepend five elements using a literal value.
    ```{python}
    df2.select(
        pl.all()
        .list.eval(ti.prepend(pl.element(), offset=4, prepend_expr=pl.lit(0)))
        .list.eval(ti.prepend(pl.element(), offset=1, prepend_expr=pl.lit(0)))
    )
    ```
    In this case, the number of elements to prepend (5) exceeds the number of
    elements in each list (4), so you need to call `prepend()` twice in separate
    `.list.eval()` steps to achieve the desired result.

    You can also approach this by using `bulk_append()` to achieve the same result:
    ```{python}
    df2.select(pl.all().list.eval(ti.bulk_append(pl.repeat(0, 5), pl.element())))
    ```
    ::: {.callout-note collapse="true"}
    ### Why `bulk_append()` is not used to implement `prepend()`

    1. If `prepend_value=` is an expression rather than a literal, there's no
    reliable way to determine how many times the expression should be prepended
    dynamically.

    2. This approach only works within the list namespace. Aliases appear to have
    no effect in this context. For example, `pl.repeat()` typically carries a
    "literal" alias in the DataFrame context, which makes it difficult to
    programmatically substitute it with the original column name—especially
    when supporting wildcards like `pl.all()` or `pl.col("*")`.
    :::
    """
    if offset < 0:
        raise ValueError("`offset=` cannot be negative.")
    if offset == 0:
        return expr
    if prepend_expr is None:
        prepend_expr: pl.Expr = expr  # type: ignore[no-redef]
    case_list = _get_case_list(expr, offset)
    return case_when(case_list, prepend_expr).head(offset).append(expr)
