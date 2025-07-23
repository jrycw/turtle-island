import polars as pl
import pytest

import turtle_island as ti


@pytest.fixture
def df_html():
    return pl.DataFrame(
        {
            "text": ["Turtle Island"],
            "url": ["https://github.com/jrycw/turtle-island"],
            "description": ["A Utility Kit for Polars Expressions"],
        }
    )


@pytest.mark.parametrize("expr1, expr2", [("text", "url")])
def test_make_hyperlink(df_html, expr1, expr2):
    name = "hyperlink"
    expr = ti.make_hyperlink(expr1, expr2)

    assert expr.meta.output_name() == name

    new_df = df_html.select(expr)

    assert name in new_df.columns

    result = new_df.item()
    expected = '<a href="https://github.com/jrycw/turtle-island" target="_blank">Turtle Island</a>'

    assert result == expected


@pytest.mark.parametrize(
    "new_tab, expected", [(True, "_blank"), (False, "_self")]
)
def test_make_hyperlink_newtab(df_html, new_tab, expected):
    new_df = df_html.select(ti.make_hyperlink("text", "url", new_tab=new_tab))

    assert expected in new_df.item()


@pytest.mark.parametrize("name", [("name")])
def test_make_hyperlink_alias(name):
    expr = ti.make_hyperlink("text", "url", name=name)

    assert expr.meta.output_name() == name


@pytest.mark.parametrize("expr1, expr2", [("text", "description")])
def test_make_tooltip(df_html, expr1, expr2):
    name = "tooltip"
    expr = ti.make_tooltip(expr1, expr2)

    assert expr.meta.output_name() == name

    new_df = df_html.select(expr)
    assert name in new_df.columns

    result = new_df.item()
    expected = '<abbr style="cursor: help; text-decoration: underline; text-decoration-style: dotted; color: blue; " title="A Utility Kit for Polars Expressions">Turtle Island</abbr>'

    assert result == expected


@pytest.mark.parametrize(
    "expr1, expr2, text_decoration_style, color",
    [
        ("text", "description", "dotted", "blue"),
        ("text", "description", "none", "none"),
    ],
)
def test_make_tooltip_options(
    df_html, expr1, expr2, text_decoration_style, color
):
    new_df = df_html.select(
        ti.make_tooltip(
            expr1,
            expr2,
            text_decoration_style=text_decoration_style,
            color=color,
        )
    )
    result = new_df.item()

    assert text_decoration_style in result
    assert color in result


@pytest.mark.parametrize("name", [("name")])
def test_make_tooltip_alias(name):
    expr = ti.make_tooltip("label", "tooltip", name=name)

    assert expr.meta.output_name() == name


@pytest.mark.parametrize(
    "expr1, expr2, text_decoration_style",
    [("name", "description", "text_decoration_style")],
)
def test_make_tooltip_raise_text_decoration_style(
    expr1, expr2, text_decoration_style
):
    with pytest.raises(ValueError) as exc_info:
        ti.make_tooltip(
            expr1, expr2, text_decoration_style=text_decoration_style
        )

    assert (
        "`text_decoration_style=` must be one of 'none', 'solid', or 'dotted'"
        in exc_info.value.args[0]
    )


@pytest.mark.parametrize(
    "expr1, expr2, color", [("name", "description", None)]
)
def test_make_tooltip_raise_color(expr1, expr2, color):
    with pytest.raises(ValueError) as exc_info:
        ti.make_tooltip(expr1, expr2, color=color)

    assert (
        "`color=` must be a string or 'none', not None."
        in exc_info.value.args[0]
    )
