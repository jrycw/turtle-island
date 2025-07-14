from typing import Iterable, Sequence

import polars as pl


__all__ = ["bulk_append", "case_when"]


def case_when(
    caselist: Sequence[tuple[pl.Expr | Iterable[pl.Expr], pl.Expr]],
    otherwise: pl.Expr | None = None,
) -> pl.Expr:
    """
    Simplifies conditional logic in Polars by chaining multiple `when‑then‑otherwise` expressions.

    Inspired by [pd.Series.case_when()](https://pandas.pydata.org/docs/reference/api/pandas.Series.case_when.html), this function offers a more ergonomic way to express chained
    conditional logic with Polars expressions.

    ::: {.callout-warning}
    ### Keyword shortcut is not supported
    Passing multiple keyword arguments to `pl.when()` as equality matches—for example, `x=123`—is not supported.
    :::

    Parameters
    ----------
    caselist
        A sequence of tuples where each tuple represents a `when` and `then`
        branch. This function accepts three input styles (see examples below). In
        every case, the `when` condition is evaluated first; if it evaluates to
        true, the corresponding `then` expression is returned. If none of the
        conditions match, the `otherwise` expression is used.

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

    `expr3` uses an iterable as the first element of each tuple, containing multiple
    `when` conditions which are also combined with `&` before evaluation.
    ```{python}
    import polars as pl
    import turtle_island as ti

    df = pl.DataFrame({"x": [1, 2, 3, 4], "y": [5, 6, 7, 8]})

    expr1 = ti.case_when(
        caselist=[
            (pl.col("x") < 2, pl.lit("small")),
            (pl.col("x") < 4, pl.lit("medium")),
        ],
        otherwise=pl.lit("large"),
    ).alias("size1")

    expr2 = ti.case_when(
        caselist=[
            (pl.col("x") < 3, pl.col("y") < 6, pl.lit("small")),
            (pl.col("x") < 4, pl.col("y") < 8, pl.lit("medium")),
        ],
        otherwise=pl.lit("large"),
    ).alias("size2")

    expr3 = ti.case_when(
        caselist=[
            ((pl.col("x") < 3, pl.col("y") < 6), pl.lit("small")),
            ((pl.col("x") < 4, pl.col("y") < 8), pl.lit("medium")),
        ],
        otherwise=pl.lit("large"),
    ).alias("size3")

    df.with_columns(expr1, expr2, expr3)
    ```
    """

    from polars.expr.whenthen import Then

    first_case, *cases = caselist

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
    Combine multiple Polars expressions using [pl.Expr.append()](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.append.html#polars-expr-append) internally.

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
