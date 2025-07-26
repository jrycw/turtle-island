from typing import Literal

import polars as pl

from .general import make_concat_str

__all__ = ["make_tooltip", "make_hyperlink"]


def make_hyperlink(
    text: str, url: str, new_tab: bool = True, *, name: str = "hyperlink"
) -> pl.Expr:
    """
    Returns a Polars expression that generates an HTML hyperlink (`<a>` tag) for each row.

    ::: {.callout-tip}
    # Credit: gt-extras
    This function is heavily inspired by the [gt-extras](https://github.com/posit-dev/gt-extras) package.
    :::

    Parameters
    ----------
    text
        Column name containing the display text for the hyperlink.

    url
        Column name containing the destination URL.

    new_tab
        Whether the link opens in a new browser tab (`target="_blank"`) or the current tab.

    name
        The name of the resulting column.

    Returns
    -------
    pl.Expr
        A Polars expression generating the HTML anchor tag.

    Examples
    --------
    ### DataFrame Context

    Create an HTML anchor tag (`<a>`) combining link text and URL from two columns:
    ```{python}
    import polars as pl
    import turtle_island as ti

    pl.Config.set_fmt_str_lengths(200)
    df = pl.DataFrame(
        {
            "name": ["Turtle Island"],
            "url": ["https://github.com/jrycw/turtle-island"],
        }
    )
    new_df = df.with_columns(ti.make_hyperlink("name", "url"))
    new_df
    ```
    ```{python}
    new_df.style
    ```
    """
    target = "_blank" if new_tab else "_self"
    return make_concat_str(
        f'<a href="[$X]" target="{target}">[$X]</a>', url, text, name=name
    )


def make_tooltip(
    label: str,
    tooltip: str,
    text_decoration_style: Literal["solid", "dotted", "none"] = "dotted",
    color: str | Literal["none"] = "blue",
    *,
    name: str = "tooltip",
) -> pl.Expr:
    """
    Returns a Polars expression that generates an HTML tooltip for each row.

    ::: {.callout-tip}
    # Credit: gt-extras
    This function is heavily inspired by the [gt-extras](https://github.com/posit-dev/gt-extras) package.
    :::

    Parameters
    ----------
    label
        Column name containing the main text to display.

    tooltip
        Column name containing containing the text shown when hovering over the label.

    text_decoration_style
        A string indicating the style of underline decoration. Options are `"solid"`,
        `"dotted"`, or `"none"`.

    color
        A string indicating the text color. If "none", no color styling is applied.

    name
        The name of the resulting column.

    Returns
    -------
    pl.Expr
        A Polars expression that creates an HTML string with tooltip functionality.

    Examples
    -------
    ### DataFrame Context

    ```{python}
    import polars as pl
    import turtle_island as ti

    pl.Config.set_fmt_str_lengths(200)
    df = pl.DataFrame(
        {
            "name": ["Turtle Island"],
            "description": ["A Utility Kit for Polars Expressions"],
        }
    )
    new_df = df.with_columns(ti.make_tooltip("name", "description"))
    new_df
    ```
    ```{python}
    new_df.style
    ```
    """

    # Throw if `text_decoration_style` is not one of the allowed values
    if text_decoration_style not in ["none", "solid", "dotted"]:
        raise ValueError(
            "`text_decoration_style=` must be one of 'none', 'solid', or 'dotted'"
        )

    if color is None:
        raise ValueError("`color=` must be a string or 'none', not None.")

    style = "cursor: help; "

    if text_decoration_style != "none":
        style += "text-decoration: underline; "
        style += f"text-decoration-style: {text_decoration_style}; "
    else:
        style += "text-decoration: none; "

    if color != "none":
        style += f"color: {color}; "

    return make_concat_str(
        f'<abbr style="{style}" title="[$X]">[$X]</abbr>',
        tooltip,
        label,
        name=name,
    )
