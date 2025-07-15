from collections.abc import Collection
from typing import Any

import polars as pl
from polars._typing import PolarsDataType

from .._utils import _get_unique_name, _litify
from .common import case_when
from .core import make_index


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
