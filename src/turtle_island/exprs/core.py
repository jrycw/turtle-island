import polars as pl


__all__ = [
    "make_index",
]


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
    return _make_index(0, pl.len(), name=name).add(offset)
