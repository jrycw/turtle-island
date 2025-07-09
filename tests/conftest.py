import datetime

import polars as pl
import pytest


@pytest.fixture(scope="module")
def df_x():
    return pl.DataFrame({"x": [1, 2, 3, 4]})


@pytest.fixture(scope="module")
def df_n():
    """
    n_row = 9
    """
    return pl.DataFrame({"n": [1, 2, 3, 4, 5, 6, 7, 8, 9]})


@pytest.fixture(scope="module")
def df_abcd():
    return pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [1.11, 2.22, 3.33],
            "c": [4, 5, 6],
            "d": ["x", "y", "z"],
        }
    )


@pytest.fixture(scope="module")
def df_full():
    return pl.DataFrame(
        {
            "int": [1, 2, 3],
            "float": [1.11, 2.22, 3.33],
            "bool": [True, False, True],
            "string": ["x", "y", "z"],
            "datetimee": pl.datetime_range(
                datetime.datetime(2025, 1, 1),
                datetime.datetime(2025, 1, 3),
                eager=True,
            ),
            "date": pl.date_range(
                datetime.date(2025, 1, 1),
                datetime.date(2025, 1, 3),
                eager=True,
            ),
            "time": pl.time_range(
                datetime.time(13, 0), datetime.time(15, 0), eager=True
            ),
            "duration": [
                datetime.timedelta(seconds=1),
                datetime.timedelta(minutes=1),
                datetime.timedelta(hours=1),
            ],
            "list": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "tuple": [(1, 2, 3), (4, 5, 6), (7, 8, 9)],
        }
    )
