from turtle_island.exprs.general import _get_move_cols


def test__get_move_cols():
    assert _get_move_cols(["col1"]) == ["col1"]
    assert _get_move_cols("col1", "col2") == ["col1", "col2"]
    assert _get_move_cols(["col1"], "col2") == ["col1", "col2"]
