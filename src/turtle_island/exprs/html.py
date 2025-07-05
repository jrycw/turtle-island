import polars as pl


def with_hyperlink(text: str, url: str, new_tab: bool = True) -> pl.Expr:
    """
    Return a Polars expression that generates an HTML hyperlink (<a> tag) for each row.

    Parameters
    ----------
    text
        The name of the column containing the display text for the hyperlink.
    url
        The name of the column containing the destination URL for the hyperlink.
    new_tab
        If True, the link opens in a new browser tab (`target="_blank"`). If False,
        it opens in the current tab. Default is True.

    Returns
    -------
    pl.Expr

    Examples
    --------
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
