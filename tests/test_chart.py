"""The charts: what they must not hide, and what they must not draw outside.

Two rules are being protected. A track longer than the chart is wide is
reduced by column, and the reduction keeps the lowest and highest value in
each column, so a single low-complexity window in a long sequence cannot be
averaged out of sight. And every mark a chart draws stays inside the viewBox
it declares, since a bar that runs past its own frame is invisible in exactly
the cases where it matters most.

Charts are checked as XML rather than by eye: every fragment has to parse, and
its coordinates are read back out of the parse rather than out of the string.
"""

import xml.etree.ElementTree as ET

import pytest

from kmer_counter import chart

_SVG = "{http://www.w3.org/2000/svg}"


def _parse(fragment):
    """Parse a chart and return its root and its viewBox as four floats."""
    root = ET.fromstring(fragment)
    return root, [float(value) for value in root.get("viewBox").split()]


def _points(root, tag):
    for element in root.iter(f"{_SVG}{tag}"):
        for point in element.get("points").split():
            x, y = point.split(",")
            yield float(x), float(y)


def test_a_track_shorter_than_the_chart_keeps_one_column_per_window():
    points = [(float(index), float(index)) for index in range(1, 11)]
    columns = chart.reduce_track(points, 10)
    assert len(columns) == 10
    assert all(column.count == 1 for column in columns)
    assert [column.mean for column in columns] == [float(index) for index in range(1, 11)]


def test_reducing_a_track_keeps_the_extremes_of_each_column():
    # One window out of a hundred is low. Reduced to four columns it must
    # still be the low of its column: a mean-only reduction would bury it.
    points = [(float(index), 30.0) for index in range(1, 101)]
    points[72] = (73.0, 2.0)
    columns = chart.reduce_track(points, 4)

    assert len(columns) == 4
    assert min(column.low for column in columns) == 2.0
    assert max(column.high for column in columns) == 30.0
    holding = [column for column in columns if column.low == 2.0]
    assert len(holding) == 1
    assert holding[0].mean < 30.0


def test_a_track_is_reduced_by_position_not_by_sample_number():
    # A profile sampled with a step has gaps in it; a column must still stand
    # for a stretch of sequence rather than for a count of samples.
    points = [(1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (100.0, 9.0)]
    columns = chart.reduce_track(points, 2)
    assert len(columns) == 2
    assert columns[0].count == 3
    assert columns[1].count == 1
    assert columns[1].high == 9.0


def test_a_track_of_one_position_reduces_to_one_column():
    columns = chart.reduce_track([(5.0, 2.0), (5.0, 4.0)], 8)
    assert len(columns) == 1
    assert (columns[0].low, columns[0].high, columns[0].count) == (2.0, 4.0, 2)


def test_reducing_an_empty_track_gives_no_columns():
    assert chart.reduce_track([], 10) == []


def test_a_track_needs_at_least_one_column():
    with pytest.raises(ValueError, match="at least one column"):
        chart.reduce_track([(1.0, 1.0)], 0)


def test_bars_need_a_value_for_every_label():
    with pytest.raises(ValueError, match="2 labels but 1 values"):
        chart.bars(["AC", "GT"], [1.0])


def test_no_labels_draws_nothing():
    assert chart.bars([], []) == ""
    assert chart.track([], "score") == ""
    assert chart.grid([], [], []) == ""


def test_every_bar_stays_inside_the_chart():
    root, box = _parse(chart.bars(["AAAA", "CCCC", "GGGG"], [100.0, 50.0, 0.0]))
    for rect in root.iter(f"{_SVG}rect"):
        right = float(rect.get("x")) + float(rect.get("width"))
        bottom = float(rect.get("y")) + float(rect.get("height"))
        assert right <= box[0] + box[2] + 0.01
        assert bottom <= box[1] + box[3] + 0.01


def test_a_bar_of_zero_has_no_width_and_the_longest_fills_the_plot():
    root, _ = _parse(chart.bars(["one", "zero"], [8.0, 0.0]))
    widths = [float(rect.get("width")) for rect in root.iter(f"{_SVG}rect")]
    assert widths[1] == 0.0
    assert widths[0] > widths[1]


def test_each_bar_names_itself_for_a_reader_who_cannot_see_it():
    root, _ = _parse(chart.bars(["AAAA"], [3.0], unit="windows"))
    titles = [element.text for element in root.iter(f"{_SVG}title")]
    assert titles == ["AAAA: 3 windows"]


def test_a_given_note_replaces_the_generated_one():
    root, _ = _parse(chart.bars(["AAAA"], [3.0], notes=["said so"]))
    assert [element.text for element in root.iter(f"{_SVG}title")] == ["said so"]


def test_every_point_of_a_track_stays_inside_the_chart():
    points = [(float(index), float(index % 7)) for index in range(1, 400)]
    root, box = _parse(chart.track(points, "score", threshold=3.0))
    drawn = list(_points(root, "polyline")) + list(_points(root, "polygon"))
    assert drawn
    for x, y in drawn:
        assert box[0] <= x <= box[0] + box[2] + 0.01
        assert box[1] <= y <= box[1] + box[3] + 0.01


def test_a_threshold_outside_the_axis_is_not_drawn():
    points = [(float(index), 1.0) for index in range(1, 20)]
    inside, _ = _parse(chart.track(points, "score", threshold=1.0, ceiling=2.0))
    outside, _ = _parse(chart.track(points, "score", threshold=9.0, ceiling=2.0))
    assert any(
        "threshold" in (line.get("class") or "") for line in inside.iter(f"{_SVG}line")
    )
    assert not any(
        "threshold" in (line.get("class") or "") for line in outside.iter(f"{_SVG}line")
    )


def test_a_region_is_shaded_and_names_its_bounds():
    points = [(float(index), 1.0) for index in range(1, 50)]
    root, _ = _parse(chart.track(points, "score", regions=[(10, 20)]))
    shaded = [
        rect for rect in root.iter(f"{_SVG}rect") if "region" in (rect.get("class") or "")
    ]
    assert len(shaded) == 1
    assert shaded[0].find(f"{_SVG}title").text == "low complexity 10-20"


def test_a_track_with_one_window_per_column_draws_no_range_band():
    # The band stands for the spread inside a column. With one window per
    # column there is no spread, and drawing one would invent a range.
    points = [(float(index), float(index)) for index in range(1, 6)]
    root, _ = _parse(chart.track(points, "score"))
    assert not list(root.iter(f"{_SVG}polygon"))


def test_a_grid_needs_a_row_of_values_for_every_row():
    with pytest.raises(ValueError, match="2 rows but 1 rows of values"):
        chart.grid(["A", "C"], ["A", "C"], [[1.0, 1.0]])


def test_a_grid_cell_with_no_value_is_marked_not_assessed():
    root, _ = _parse(chart.grid(["A"], ["C"], [[None]]))
    cell = next(root.iter(f"{_SVG}rect"))
    assert "empty" in cell.get("class")
    assert cell.find(f"{_SVG}title").text == "AC: not assessed"


def test_a_grid_shades_above_and_below_the_centre_differently():
    root, _ = _parse(chart.grid(["A", "C"], ["A"], [[2.0], [0.5]], centre=1.0))
    classes = [rect.get("class") for rect in root.iter(f"{_SVG}rect")]
    assert classes == ["cell over", "cell under"]


def test_a_small_grid_prints_its_numbers_and_a_large_one_does_not():
    small, _ = _parse(chart.grid(["A"], ["C"], [[1.25]]))
    assert any(
        "cellvalue" in (text.get("class") or "") for text in small.iter(f"{_SVG}text")
    )

    residues = list("ACDEFGHIKLMNPQRSTVWY")
    values = [[1.0 for _ in residues] for _ in residues]
    large, _ = _parse(chart.grid(residues, residues, values))
    assert not any(
        "cellvalue" in (text.get("class") or "") for text in large.iter(f"{_SVG}text")
    )
    assert len(list(large.iter(f"{_SVG}title"))) == len(residues) ** 2
