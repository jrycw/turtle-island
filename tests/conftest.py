import polars as pl
import pytest


@pytest.fixture(scope="module")
def df_x():
    return pl.DataFrame({"x": [1, 2, 3, 4]})


@pytest.fixture(scope="module")
def df_abcd():
    return pl.DataFrame(
        {"a": [1, 2, 3], "b": [1.11, 2.22, 3.33], "c": [4, 5, 6], "d": ["x", "y", "z"]}
    )


@pytest.fixture(scope="module")
def df_n():
    """
    n_row = 9
    """
    return pl.DataFrame({"n": [100, 50, 72, 83, 97, 42, 20, 51, 77]})
