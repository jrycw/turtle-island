import polars as pl
from .._utils import _concat_str


def make_hyperlink(text: str, url: str, new_tab: bool = True) -> pl.Expr:
    """
    Returns a Polars expression that generates an HTML hyperlink (`<a>` tag) for each row.

    Parameters
    ----------
    text
        Column name containing the display text for the hyperlink.
    url
        Column name containing the destination URL.
    new_tab
        Whether the link opens in a new browser tab (`target="_blank"`) or the current tab.
        Defaults to `True`.

    Returns
    -------
    pl.Expr
        A Polars expression generating the HTML anchor tag.

    Examples
    --------
    Create an HTML anchor tag (`<a>`) combining link text and URL from two columns:
    ```{python}
    import polars as pl
    import turtle_island as ti

    df = pl.DataFrame(
        {
            "name": ["Turtle Island"],
            "url": ["https://github.com/jrycw/turtle-island"],
        }
    )
    df.with_columns(ti.make_hyperlink("name", "url").alias("link")).style
    ```
    """
    target: str = "_blank" if new_tab else "_self"
    return _concat_str(
        f'<a href="<<X>>" target="{target}"><<X>></a>', url, text
    )
