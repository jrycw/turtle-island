import datetime
import uuid
from typing import Any, Collection

import polars as pl


def _litify(items: Collection[Any]) -> list[pl.Expr]:
    return [pl.lit(item) for item in items]


def _get_unique_name(n: int = 10) -> str:
    if n < 8:
        raise ValueError(
            "`n` must be at least 8 to ensure uniqueness of the name."
        )
    return uuid.uuid4().hex[:n]


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
