import polars as pl
import pytest

import turtle_island as ti


@pytest.fixture
def df_html():
    return pl.DataFrame(
        {
            "name": ["Turtle Island"],
            "url": ["https://github.com/jrycw/turtle-island"],
        }
    )


@pytest.mark.parametrize(
    "expr1, expr2",
    [
        (pl.col("name"), pl.col("url")),
        (pl.col("name"), "url"),
        ("name", pl.col("url")),
        ("name", "url"),
    ],
)
def test_with_hyperlink(df_html, expr1, expr2):
    new_df = df_html.select(ti.with_hyperlink(expr1, expr2))
    result = new_df.item()
    expected = '<a href="https://github.com/jrycw/turtle-island" target="_blank">Turtle Island</a>'

    assert result == expected


@pytest.mark.parametrize("new_tab, expected", [(True, "_blank"), (False, "_self")])
def test_with_hyperlink_newtab(df_html, new_tab, expected):
    new_df = df_html.select(ti.with_hyperlink("name", "url", new_tab=new_tab))

    assert expected in new_df.item()
