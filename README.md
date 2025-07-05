# 🐢 Turtle Island
**Turtle Island** is a lightweight utility library that provides helper functions to reduce boilerplate when writing [Polars](https://pola.rs) expressions. It aims to simplify common expression patterns and improve developer productivity when working with the Polars API.

> ⚠️ **Disclaimer**: This project is in early development. The API is still evolving and may change without notice. Use with caution in production environments.


## 📦 Recommended Import
To keep your code clean and idiomatic, it's recommended to import **Turtle Island** as a top-level module:

```python
import turtle_island as ti
```

## ✨ Selected Functions

### `case_when()`
A more ergonomic way to write chained `when-then-otherwise` logic in Polars:
```python
df = pl.DataFrame({"x": [1, 2, 3, 4]})

expr_ti = ti.case_when(
    caselist=[(pl.col("x") < 2, pl.lit("small")),
              (pl.col("x") < 4, pl.lit("medium"))],
    otherwise=pl.lit("large"),
).alias("size_ti")

expr_pl = (
    pl.when(pl.col("x") < 2)
    .then(pl.lit("small"))
    .when(pl.col("x") < 4)
    .then(pl.lit("medium"))
    .otherwise(pl.lit("large"))
    .alias("size_pl")
)

df.with_columns(expr_ti, expr_pl)
```
```
shape: (4, 3)
┌─────┬─────────┬─────────┐
│ x   ┆ size_ti ┆ size_pl │
│ --- ┆ ---     ┆ ---     │
│ i64 ┆ str     ┆ str     │
╞═════╪═════════╪═════════╡
│ 1   ┆ small   ┆ small   │
│ 2   ┆ medium  ┆ medium  │
│ 3   ┆ medium  ┆ medium  │
│ 4   ┆ large   ┆ large   │
└─────┴─────────┴─────────┘
```

### `create_index()`
Adds a sequential index column to the DataFrame:
```python
df = pl.DataFrame({"a": [1, 3, 5], "b": [2, 4, 6]})
df.select(ti.create_index(), pl.all())
```
```
shape: (3, 3)
┌───────┬─────┬─────┐
│ index ┆ a   ┆ b   │
│ ---   ┆ --- ┆ --- │
│ u32   ┆ i64 ┆ i64 │
╞═══════╪═════╪═════╡
│ 0     ┆ 1   ┆ 2   │
│ 1     ┆ 3   ┆ 4   │
│ 2     ┆ 5   ┆ 6   │
└───────┴─────┴─────┘
```

### `bucketize()`
Assign values to rows based on index in a round-robin fashion:
```python
df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
df.with_columns(ti.bucketize("a", "b"))
```
```
shape: (5, 2)
┌─────┬────────────┐
│ a   ┆ bucketized │
│ --- ┆ ---        │
│ i64 ┆ str        │
╞═════╪════════════╡
│ 1   ┆ a          │
│ 2   ┆ b          │
│ 3   ┆ a          │
│ 4   ┆ b          │
│ 5   ┆ a          │
└─────┴────────────┘
```

### `move_cols_to_start()`
Reorder columns so that selected columns appear first:
```python
df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [4.4, 5.5, 6.6]})
df.select(ti.move_cols_to_start(["b", "c"]))
```
Or by data type:
```python
df.select(ti.move_cols_to_start([pl.Float64, pl.String]))
```
```
shape: (3, 3)
┌─────┬─────┬─────┐
│ b   ┆ c   ┆ a   │
│ --- ┆ --- ┆ --- │
│ str ┆ f64 ┆ i64 │
╞═════╪═════╪═════╡
│ x   ┆ 4.4 ┆ 1   │
│ y   ┆ 5.5 ┆ 2   │
│ z   ┆ 6.6 ┆ 3   │
└─────┴─────┴─────┘
```
### `move_cols_to_end()`
Reorder columns so that selected columns appear last:
```python
df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [4.4, 5.5, 6.6]})
df.select(ti.move_cols_to_end(["a", "b"]))
```
Or by data type:
```python
df.select(ti.move_cols_to_end([pl.String, pl.Int64]))
```
```
shape: (3, 3)
┌─────┬─────┬─────┐
│ c   ┆ a   ┆ b   │
│ --- ┆ --- ┆ --- │
│ f64 ┆ i64 ┆ str │
╞═════╪═════╪═════╡
│ 4.4 ┆ 1   ┆ x   │
│ 5.5 ┆ 2   ┆ y   │
│ 6.6 ┆ 3   ┆ z   │
└─────┴─────┴─────┘
```

### `with_hyperlink()`
Create an HTML anchor tag (`<a>`) from two columns — link text and URL:
```python
df = pl.DataFrame({"name": ["GitHub"], "url": ["https://github.com/"]})
df.select(ti.with_hyperlink("name", "url").alias("link"))
```
```
shape: (1, 1)
┌──────────────────────────────────────────────────────────┐
│ link                                                     │
│ ---                                                      │
│ str                                                      │
╞══════════════════════════════════════════════════════════╡
│ <a href="https://github.com/" target="_blank">GitHub</a> │
└──────────────────────────────────────────────────────────┘
```
