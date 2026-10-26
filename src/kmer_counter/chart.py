"""Charts, drawn as inline SVG.

The report has to open from a local file, from an email attachment and from a
regulatory submission years from now, so the charts cannot be images fetched
from anywhere or drawn by a script that has to run. They are SVG elements
written into the document, styled by its stylesheet, carrying a ``<title>`` per
mark so that hovering a bar names it and a screen reader can read it out.

Two rules keep the charts honest. A track longer than the chart is wide is
reduced by taking the lowest, mean and highest value in each column and
drawing the range as a band behind the mean, so a single low-complexity window
in a megabase cannot be averaged out of sight. And every axis starts at the
measure's own floor - zero for a count, zero bits for an entropy - because a
truncated axis exaggerates a difference the reader did not ask to have
exaggerated.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Sequence

#: Chart width in user units, matching the report's text column.
WIDTH: int = 720

#: Space left for the axis labels on the left of a track or bar chart.
_GUTTER: int = 96

#: Space left below a chart for its axis.
_FOOT: int = 22


def _format(value: float) -> str:
    """Write a number for a tooltip, without a trailing zero parade."""
    if value == int(value) and abs(value) < 1e15:
        return f"{int(value):,}"
    if abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value) >= 1:
        return f"{value:,.2f}"
    return f"{value:,.3f}"


def _coordinate(value: float) -> str:
    """Write a coordinate with enough precision to be smooth, not verbose."""
    return f"{value:.2f}".rstrip("0").rstrip(".")


@dataclass(frozen=True)
class Column:
    """One column of a reduced track.

    Attributes:
        position: the sequence position at the column's centre.
        low: the lowest value in the column.
        mean: the mean value in the column.
        high: the highest value in the column.
        count: how many windows fell in the column.
    """

    position: float
    low: float
    mean: float
    high: float
    count: int


def reduce_track(
    points: Sequence[tuple[float, float]],
    columns: int,
) -> list[Column]:
    """Reduce a track to at most ``columns`` columns, keeping the extremes.

    Points are assigned to columns by position, so a sequence with an
    unevenly sampled profile is still drawn against position rather than
    against sample number.

    Raises:
        ValueError: if ``columns`` is less than one.
    """
    if columns < 1:
        raise ValueError(f"a track needs at least one column, got {columns}")
    if not points:
        return []
    first = points[0][0]
    last = points[-1][0]
    span = last - first
    buckets: list[list[float]] = [[] for _ in range(columns)]
    for position, value in points:
        index = 0 if span <= 0 else int((position - first) / span * (columns - 1))
        buckets[min(columns - 1, max(0, index))].append(value)
    width = span / columns if span > 0 else 0.0
    reduced = []
    for index, values in enumerate(buckets):
        if not values:
            continue
        centre = first + (index + 0.5) * width if width else first
        reduced.append(
            Column(
                position=centre,
                low=min(values),
                mean=sum(values) / len(values),
                high=max(values),
                count=len(values),
            )
        )
    return reduced


def bars(
    labels: Sequence[str],
    values: Sequence[float],
    notes: Sequence[str] = (),
    height: int = 14,
    unit: str = "",
) -> str:
    """A horizontal bar chart, one row per label.

    Args:
        labels: the row labels, drawn in the gutter.
        values: the bar lengths; must be as many as there are labels.
        notes: optional per-row tooltip text, used instead of the value.
        height: the height of one row in user units.
        unit: what the values are, for the tooltip.

    Raises:
        ValueError: if there are not as many values as labels.
    """
    if len(labels) != len(values):
        raise ValueError(
            f"{len(labels)} labels but {len(values)} values"
        )
    if not labels:
        return ""
    widest = max(values) if max(values) > 0 else 1.0
    plot = WIDTH - _GUTTER - 60
    total = height * len(labels)
    parts = [
        f'<svg class="chart" viewBox="0 0 {WIDTH} {total}" width="{WIDTH}" '
        f'height="{total}" role="img" xmlns="http://www.w3.org/2000/svg">'
    ]
    for index, (label, value) in enumerate(zip(labels, values)):
        top = index * height
        length = max(0.0, plot * value / widest)
        note = notes[index] if index < len(notes) else (
            f"{label}: {_format(value)}{(' ' + unit) if unit else ''}"
        )
        parts.append(
            f'<g><title>{escape(note)}</title>'
            f'<text class="tick" x="{_GUTTER - 6}" y="{top + height - 4}" '
            f'text-anchor="end">{escape(label)}</text>'
            f'<rect class="bar" x="{_GUTTER}" y="{top + 2}" '
            f'width="{_coordinate(length)}" height="{height - 5}"></rect>'
            f'<text class="value" x="{_GUTTER + length + 5}" y="{top + height - 4}">'
            f"{escape(_format(value))}</text></g>"
        )
    parts.append("</svg>")
    return "".join(parts)


def track(
    points: Sequence[tuple[float, float]],
    label: str,
    threshold: float | None = None,
    regions: Sequence[tuple[int, int]] = (),
    height: int = 150,
    floor: float = 0.0,
    ceiling: float | None = None,
    columns: int = WIDTH - _GUTTER - 10,
) -> str:
    """A value plotted along the sequence, with its low-complexity regions.

    Args:
        points: ``(position, value)`` in position order.
        label: what the value is, written on the vertical axis.
        threshold: a value to draw a reference line at.
        regions: intervals to shade behind the line.
        height: the plot height in user units.
        floor: the bottom of the vertical axis.
        ceiling: the top; the highest value, or the threshold, by default.
        columns: the most columns to reduce the track to.
    """
    if not points:
        return ""
    reduced = reduce_track(points, columns)
    values = [column.high for column in reduced] + [column.low for column in reduced]
    top = ceiling if ceiling is not None else max(values + ([threshold] if threshold else []))
    if top <= floor:
        top = floor + 1.0
    first = points[0][0]
    last = max(points[-1][0], first + 1)
    plot_width = WIDTH - _GUTTER - 10
    total = height + _FOOT

    def x_of(position: float) -> float:
        return _GUTTER + plot_width * (position - first) / (last - first)

    def y_of(value: float) -> float:
        clamped = min(max(value, floor), top)
        return height - (height - 4) * (clamped - floor) / (top - floor)

    parts = [
        f'<svg class="chart" viewBox="0 0 {WIDTH} {total}" width="{WIDTH}" '
        f'height="{total}" role="img" xmlns="http://www.w3.org/2000/svg">',
        f"<desc>{escape(label)} from position {int(first):,} to {int(last):,}</desc>",
    ]
    for start, end in regions:
        left = x_of(max(start, first))
        right = x_of(min(end, last))
        if right <= left:
            right = left + 0.6
        parts.append(
            f'<rect class="region" x="{_coordinate(left)}" y="0" '
            f'width="{_coordinate(right - left)}" height="{height}">'
            f"<title>low complexity {start:,}-{end:,}</title></rect>"
        )
    band = " ".join(
        f"{_coordinate(x_of(column.position))},{_coordinate(y_of(column.high))}"
        for column in reduced
    )
    band_back = " ".join(
        f"{_coordinate(x_of(column.position))},{_coordinate(y_of(column.low))}"
        for column in reversed(reduced)
    )
    if any(column.count > 1 for column in reduced):
        parts.append(f'<polygon class="band" points="{band} {band_back}"></polygon>')
    line = " ".join(
        f"{_coordinate(x_of(column.position))},{_coordinate(y_of(column.mean))}"
        for column in reduced
    )
    parts.append(f'<polyline class="line" points="{line}"></polyline>')
    if threshold is not None and floor <= threshold <= top:
        y = y_of(threshold)
        parts.append(
            f'<line class="threshold" x1="{_GUTTER}" y1="{_coordinate(y)}" '
            f'x2="{WIDTH - 10}" y2="{_coordinate(y)}"><title>threshold '
            f"{_format(threshold)}</title></line>"
        )
    parts.extend(
        [
            f'<text class="tick" x="{_GUTTER - 6}" y="12" text-anchor="end">'
            f"{escape(_format(top))}</text>",
            f'<text class="tick" x="{_GUTTER - 6}" y="{height}" text-anchor="end">'
            f"{escape(_format(floor))}</text>",
            f'<text class="axis" x="{_GUTTER}" y="{height + 16}">'
            f"{escape(_format(first))}</text>",
            f'<text class="axis" x="{WIDTH - 10}" y="{height + 16}" '
            f'text-anchor="end">{escape(_format(last))}</text>',
            f'<text class="axis" x="{(WIDTH + _GUTTER) // 2}" y="{height + 16}" '
            f'text-anchor="middle">{escape(label)}</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def grid(
    rows: Sequence[str],
    columns: Sequence[str],
    values: Sequence[Sequence[float | None]],
    centre: float = 1.0,
    cell: int = 34,
    unit: str = "",
) -> str:
    """A matrix drawn as a grid of shaded cells.

    Shading is relative to ``centre``: a cell at the centre is unshaded, and
    cells above and below it are shaded in two different directions, so the
    reader can see at a glance which way a cell departs. A cell with no value
    is left empty rather than shaded as if it were at the centre.
    """
    if len(values) != len(rows):
        raise ValueError(f"{len(rows)} rows but {len(values)} rows of values")
    if not rows or not columns:
        return ""
    spread = max(
        (abs((value or centre) - centre) for row in values for value in row),
        default=0.0,
    ) or 1.0
    # A small grid has room to print the number in the cell; a 20-by-20 one
    # does not, and there the shading and the tooltips carry it.
    labelled = len(columns) <= 8 and cell >= 30
    left = 34
    width = left + cell * len(columns)
    total = 18 + cell * len(rows)
    parts = [
        f'<svg class="chart grid" viewBox="0 0 {width} {total}" width="{width}" '
        f'height="{total}" role="img" xmlns="http://www.w3.org/2000/svg">'
    ]
    for index, column in enumerate(columns):
        parts.append(
            f'<text class="tick" x="{left + cell * index + cell / 2}" y="12" '
            f'text-anchor="middle">{escape(column)}</text>'
        )
    for row_index, row in enumerate(rows):
        y = 18 + cell * row_index
        parts.append(
            f'<text class="tick" x="{left - 6}" y="{y + cell / 2 + 4}" '
            f'text-anchor="end">{escape(row)}</text>'
        )
        for column_index, column in enumerate(columns):
            value = values[row_index][column_index]
            x = left + cell * column_index
            if value is None:
                parts.append(
                    f'<rect class="cell empty" x="{x}" y="{y}" width="{cell - 2}" '
                    f'height="{cell - 2}"><title>{escape(row + column)}: not '
                    f"assessed</title></rect>"
                )
                continue
            weight = min(1.0, abs(value - centre) / spread)
            shade = "over" if value >= centre else "under"
            parts.append(
                f'<g><title>{escape(row + column)}: {_format(value)}'
                f'{(" " + unit) if unit else ""}</title>'
                f'<rect class="cell {shade}" x="{x}" y="{y}" width="{cell - 2}" '
                f'height="{cell - 2}" fill-opacity="{_coordinate(0.08 + 0.82 * weight)}">'
                "</rect>"
            )
            if labelled:
                parts.append(
                    f'<text class="cellvalue" x="{x + (cell - 2) / 2}" '
                    f'y="{y + cell / 2 + 3}" text-anchor="middle">'
                    f"{escape(f'{value:.2f}')}</text>"
                )
            parts.append("</g>")
    parts.append("</svg>")
    return "".join(parts)
