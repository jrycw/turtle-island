import datetime
from typing import Any, Collection

import polars as pl


def _litify(items: Collection[Any]) -> list[pl.Expr]:
    return [pl.lit(item) for item in items]


def _cast_datatype(expr: pl.Expr, item: Any) -> pl.Expr:
    """
    Cast a Polars expression to match the type of the given item.

    This utility is used internally by `bucketize()` to ensure
    the expression is cast to the appropriate type based on the
    provided value.
    """
    # order matters
    if item is True or item is False:
        return expr.cast(pl.Boolean)
    elif isinstance(item, datetime.datetime):
        return expr.cast(pl.Datetime)
    elif isinstance(item, datetime.date):
        return expr.cast(pl.Date)
    elif isinstance(item, datetime.time):
        return expr.cast(pl.Time)
    elif isinstance(item, datetime.timedelta):
        return expr.cast(pl.Duration)
    elif isinstance(item, int):
        return expr.cast(pl.Int64)
    elif isinstance(item, float):
        return expr.cast(pl.Float64)
    elif isinstance(item, str):
        return expr.cast(pl.String)
    # TODO: Is it possible to cast dict -> pl.Struct here?
    elif isinstance(item, (list, tuple)):
        return expr.cast(pl.List)
    return expr


def _concat_str(
    template: str, *col_names: str, name: str, sep: str = "**X**"
) -> pl.Expr:
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
    for lit in _litify(splitted):
        concat_str_list.append(lit)
        if col_name := next(col_names_iter, None):
            concat_str_list.append(col_name)
    return pl.concat_str(concat_str_list).alias(name)
