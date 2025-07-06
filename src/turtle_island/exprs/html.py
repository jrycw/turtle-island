import polars as pl


def with_hyperlink(text: str, url: str, new_tab: bool = True) -> pl.Expr:
    """
    Returns a Polars expression that generates an HTML hyperlink (<a> tag) for each row.

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
    Create an HTML anchor tag (<a>) combining link text and URL from two columns:
    ```{python}
    import polars as pl
    import turtle_island as ti

    df = pl.DataFrame({"name": ["GitHub"], "url": ["https://github.com/"]})
    df.select(ti.with_hyperlink("name", "url").alias("link"))
    ```
    shape: (1, 1)
    ┌──────────────────────────────────────────────────────────┐
    │ link                                                     │
    │ ---                                                      │
    │ str                                                      │
    ╞══════════════════════════════════════════════════════════╡
    │ <a href="https://github.com/" target="_blank">GitHub</a> │
    └──────────────────────────────────────────────────────────┘
    """
    target: str = "_blank" if new_tab else "_self"
    return pl.concat_str(
        [
            pl.lit('<a href="'),
            url,
            pl.lit(f'" target="{target}">'),
            text,
            pl.lit("</a>"),
        ]
    )
