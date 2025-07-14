import polars as pl


__all__ = ["bulk_append", "case_when"]


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
