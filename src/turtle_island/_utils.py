import uuid
from typing import Any, Collection

import polars as pl


def _create_uuid4_hex() -> str:
    return uuid.uuid4().hex


def _litify(items: Collection[Any]) -> list[pl.lit]:
    return [pl.lit(item) for item in items]


def _cast_datatype(expr: pl.Expr, item: Any) -> pl.Expr:
    """
    Cast a Polars expression to match the type of the given item.

    This utility is used internally by `bucketize()` to ensure
    the expression is cast to the appropriate type based on the
    provided value.
    """
    if isinstance(item, int):
        return expr.cast(pl.Int64)
    elif isinstance(item, float):
        return expr.cast(pl.Float64)
    elif isinstance(item, str):
        return expr.cast(pl.String)
    return expr
