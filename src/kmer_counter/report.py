"""The self-contained HTML dashboard.

The document embeds its own style rules and its own charts, loads nothing from
the network and runs no script, so it opens from a local disk, an email
attachment or an archive years from now and looks the same. Output is
byte-deterministic for a given input and set of parameters: nothing in it
depends on the clock, the filesystem or the order of a set.

It is interactive without being programmed. Sections fold with ``<details>``,
every bar, cell and shaded region names itself on hover through an SVG
``<title>``, and each table can be downloaded as the tab-separated file the
command line would have written, as a data URI that needs no server. That is
the whole of the interactivity on purpose: a dashboard that depends on
JavaScript is a dashboard that stops working when it is opened from a
restricted viewer or pasted into a submission.
"""

from __future__ import annotations

import base64
from html import escape
from typing import Mapping, Sequence

from kmer_counter import chart
from kmer_counter.analysis import Analysis
from kmer_counter.complexity import attainable_entropy
from kmer_counter.composition import Table, gc_content
from kmer_counter.profile import Profile
from kmer_counter.repeats import forced_repeat_share, repeat_share
from kmer_counter.rows import (
    COMPOSITION_COLUMNS,
    ENTROPY_COLUMNS,
    RECORD_COLUMNS,
    REGION_COLUMNS,
    REPEAT_COLUMNS,
    SPECTRUM_COLUMNS,
    WINDOW_COLUMNS,
    composition_rows,
    entropy_rows,
    record_rows,
    region_rows,
    render,
    repeat_rows,
    spectrum_rows,
    window_rows,
)
from kmer_counter.text import bits, number, plural, ratio, residues, share, thousands

#: A table larger than this is not embedded for download. Four megabytes of
#: base64 is already a document no mail server will carry, and the file it
#: would hold is one ``--output`` writes in a second.
MAX_EMBEDDED: int = 3_000_000

#: What separates the items of the subtitle line.
_DOT: str = " \u00b7 "

#: How many words the abundance chart shows.
TOP_WORDS: int = 20

#: How many repeats and regions are listed before the table is cut.
MAX_LISTED: int = 50

_STYLE = """
html { -webkit-text-size-adjust: 100%; }
body { font: 16px/1.55 Georgia, 'Times New Roman', serif; color: #1a1a1a;
       background: #fff; margin: 0 auto; padding: 3rem 1.25rem 4rem;
       max-width: 48rem; }
h1 { font-size: 1.5rem; font-weight: normal; margin: 0 0 .25rem; }
h2, h3 { font-size: .8rem; font-variant: small-caps; letter-spacing: .08em;
     font-weight: normal; margin: 2.5rem 0 .6rem; border-bottom: 1px solid #1a1a1a;
     padding-bottom: .2rem; }
h3 { margin-top: 1.8rem; border-bottom: none; color: #555; }
p { margin: .5rem 0; }
.lede { color: #555; font-size: .9rem; margin-bottom: .5rem; }
.note { color: #555; font-size: .85rem; }
code, .seq, td.num, th.num, .mono { font-family: ui-monospace, 'SF Mono', Menlo, Consolas, monospace; }
table { border-collapse: collapse; width: 100%; margin: .6rem 0 .2rem;
        font-size: .9rem; }
th { text-align: left; font-weight: normal; font-variant: small-caps;
     letter-spacing: .04em; border-bottom: 1px solid #1a1a1a; padding: .25rem .5rem .25rem 0; }
td { padding: .22rem .5rem .22rem 0; border-bottom: 1px solid #e6e6e6;
     vertical-align: baseline; }
td.num, th.num { text-align: right; padding-right: 1rem; font-size: .85rem;
     font-variant-numeric: tabular-nums; }
tr:last-child td { border-bottom: none; }
dl.metrics { display: grid; grid-template-columns: max-content 1fr; gap: .15rem 1.2rem;
             margin: .6rem 0; font-size: .9rem; }
dl.metrics dt { color: #555; }
dl.metrics dd { margin: 0; font-family: ui-monospace, 'SF Mono', Menlo, Consolas, monospace; }
.caveat { border-left: 2px solid #8a5a00; padding: .1rem 0 .1rem .7rem; margin: .8rem 0;
          color: #6b4700; font-size: .85rem; }
.caveat p { margin: .25rem 0; }
.tag { font-variant: small-caps; letter-spacing: .04em; font-size: .8rem; color: #0f6f6a; }
.tag.warn { color: #8a5a00; }
details { margin: .4rem 0; }
summary { cursor: pointer; font-size: .85rem; color: #555; }
summary:hover { color: #1a1a1a; }
.download { font-size: .8rem; }
svg.chart { display: block; max-width: 100%; height: auto; margin: .6rem 0 .2rem; }
svg.chart text { font: 11px ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
     fill: #555; }
svg.chart text.value { fill: #8a8a8a; }
svg.chart text.cellvalue { fill: #1a1a1a; font-size: 10px; }
svg.chart .bar { fill: #1a1a1a; }
svg.chart .line { fill: none; stroke: #1a1a1a; stroke-width: 1.2; }
svg.chart .band { fill: #1a1a1a; fill-opacity: .18; stroke: none; }
svg.chart .threshold { stroke: #8a5a00; stroke-width: 1; stroke-dasharray: 4 3; }
svg.chart .region { fill: #8a5a00; fill-opacity: .14; }
svg.chart .cell { stroke: #fff; }
svg.chart .cell.over { fill: #0f6f6a; }
svg.chart .cell.under { fill: #8a5a00; }
svg.chart .cell.empty { fill: #f2f2f2; fill-opacity: 1; }
footer { margin-top: 3rem; padding-top: .5rem; border-top: 1px solid #e6e6e6;
         color: #555; font-size: .8rem; }
footer p { margin: .35rem 0; }
@media print { body { padding: 0; max-width: none; }
  h2, h3 { page-break-after: avoid; } details { display: block; }
  summary { display: none; } }
"""


def _definitions(rows: Sequence[tuple[str, str]]) -> list[str]:
    parts = ['<dl class="metrics">']
    for name, value in rows:
        parts.append(f"<dt>{escape(name)}</dt><dd>{escape(value)}</dd>")
    parts.append("</dl>")
    return parts


def _download(columns: Sequence[str], rows: Sequence[Sequence[object]], filename: str) -> str:
    """A link that downloads a table, with the table inside the link.

    The file is carried as a base64 data URI, so the link works from a
    document that has been copied, mailed or archived, with nothing to fetch.
    """
    text = render(columns, rows)
    payload = text.encode("utf-8")
    if len(payload) > MAX_EMBEDDED:
        return (
            f'<p class="note download">{escape(filename)} is '
            f"{len(payload) / 1_000_000:.1f} MB, too large to embed; write it with "
            "<code>--output</code>.</p>"
        )
    encoded = base64.b64encode(payload).decode("ascii")
    return (
        f'<p class="note download"><a download="{escape(filename)}" '
        f'href="data:text/tab-separated-values;base64,{encoded}">'
        f"download {escape(filename)}</a> &middot; "
        f"{escape(plural(len(rows), 'row'))}</p>"
    )


def _folded(summary: str, parts: Sequence[str], open_by_default: bool = False) -> list[str]:
    """Wrap a long table in a fold, so the dashboard stays scannable.

    The fold is a ``<details>`` element, which needs no script, prints open,
    and leaves the table in the document for anyone searching it.
    """
    marker = " open" if open_by_default else ""
    return [f"<details{marker}><summary>{escape(summary)}</summary>", *parts, "</details>"]


def _spectrum_section(analysis: Analysis) -> list[str]:
    spectrum = analysis.spectrum
    top = spectrum.most_common(TOP_WORDS)
    parts = [f"<h2>Words of {analysis.k} residues</h2>"]
    if not top:
        parts.append(
            "<p>No window of this length could be counted, so there is no "
            "spectrum to show.</p>"
        )
        return parts
    labels = [word for word, _ in top]
    values = [float(number_) for _, number_ in top]
    notes = [
        f"{word}: {thousands(number_)} windows, {share(spectrum.frequency(word))} "
        "of those counted"
        for word, number_ in top
    ]
    parts.append(
        f'<p class="note">The {escape(plural(len(top), "most abundant word"))} of '
        f"length {analysis.k}, of {escape(thousands(spectrum.distinct))} that occur "
        f"and {escape(thousands(spectrum.possible))} that could.</p>"
    )
    parts.append(chart.bars(labels, values, notes, unit="windows"))

    census = spectrum.census()
    if census:
        parts.append("<h3>How often a word occurs</h3>")
        abundances = sorted(census)
        parts.append(
            chart.bars(
                [thousands(abundance) for abundance in abundances],
                [float(census[abundance]) for abundance in abundances],
                [
                    f"{thousands(census[abundance])} words occur "
                    f"{escape(plural(abundance, 'time'))}"
                    for abundance in abundances
                ],
                unit="words",
            )
        )
        parts.append(
            f'<p class="note">{escape(plural(spectrum.singletons(), "word"))} of '
            f"length {analysis.k} occur once. A spectrum with a long tail to the "
            "right is a repetitive sequence: those are the words a repeat is "
            "built from.</p>"
        )
    parts.append(
        _download(SPECTRUM_COLUMNS, spectrum_rows(analysis), f"spectrum-k{analysis.k}.tsv")
    )
    return parts


def _complexity_section(analysis: Analysis) -> list[str]:
    parts = ["<h2>Complexity</h2>"]
    rows = [
        "<table>",
        '<tr><th class="num">k</th><th class="num">Windows</th>'
        '<th class="num">Distinct</th><th class="num">Observable</th>'
        '<th class="num">Saturation</th><th class="num">Entropy</th>'
        '<th class="num">Ceiling</th><th class="num">Efficiency</th></tr>',
    ]
    for k, entropy in analysis.entropies.items():
        spectrum = analysis.spectra[k]
        rows.append(
            f'<tr><td class="num">{k}</td>'
            f'<td class="num">{thousands(spectrum.windows)}</td>'
            f'<td class="num">{thousands(spectrum.distinct)}</td>'
            f'<td class="num">{thousands(spectrum.observable)}</td>'
            f'<td class="num">{share(spectrum.saturation)}</td>'
            f'<td class="num">{number(entropy.bits)}</td>'
            f'<td class="num">{number(entropy.ceiling)}</td>'
            f'<td class="num">{share(entropy.efficiency)}</td></tr>'
        )
    rows.append("</table>")
    parts.extend(rows)
    parts.append(
        '<p class="note">Saturation is the share of the words a sequence of this '
        "length could have shown that it did show; efficiency is its entropy as a "
        "share of the entropy it could have reached. Both are capped by length "
        "rather than by composition once the word space is larger than the number "
        "of windows, which is why the ceiling is reported beside the entropy "
        "rather than left implicit.</p>"
    )
    parts.append(_download(ENTROPY_COLUMNS, entropy_rows(analysis), "complexity.tsv"))
    return parts


def _track_section(analysis: Analysis) -> list[str]:
    tracks = analysis.tracks
    parts = ["<h2>Complexity along the sequence</h2>"]
    if not tracks:
        parts.append(
            "<p>No window profile was kept, so there is no track to plot.</p>"
        )
        return parts
    measure = (
        "DUST score" if analysis.alphabet.is_nucleotide else "residue entropy, bits"
    )
    parts.append(
        f'<p class="note">Each point is a window of {escape(thousands(analysis.window))} '
        f"residues, advancing {escape(plural(analysis.step, 'residue'))} at a time. "
        "Where a column of the chart holds more than one window the band shows the "
        "range in it and the line the mean, so a single low-complexity window is "
        "not averaged away. The dashed line is the threshold at which a window is "
        "called low-complexity, and the shaded bands are the regions called.</p>"
    )
    for profiled in tracks:
        parts.extend(_one_track(profiled, measure))
    windows = window_rows(tracks)
    parts.append(_download(WINDOW_COLUMNS, windows, "windows.tsv"))
    return parts


def _one_track(profiled: Profile, measure: str) -> list[str]:
    regions = [(region.start, region.end) for region in profiled.intervals]
    parts = [f"<h3>{escape(profiled.name)}</h3>"]
    parts.append(
        chart.track(
            [(float(window.start), window.score) for window in profiled.windows],
            label=f"{measure} \u00b7 position",
            threshold=profiled.threshold,
            regions=regions,
            ceiling=profiled.ceiling,
        )
    )
    if not profiled.alphabet.is_nucleotide:
        return parts
    parts.append(
        chart.track(
            [(float(window.start), window.entropy) for window in profiled.windows],
            label="base entropy, bits \u00b7 position",
            ceiling=attainable_entropy(profiled.window, profiled.alphabet.size),
        )
    )
    return parts


def _region_section(analysis: Analysis) -> list[str]:
    rows = region_rows(analysis)
    parts = ["<h3>Low-complexity regions</h3>"]
    if not rows:
        parts.append(
            f'<p class="note">No window reached the threshold of '
            f"{number(analysis.threshold)}, so no region was called.</p>"
        )
        return parts
    listed = rows[:MAX_LISTED]
    parts.extend(
        [
            "<table>",
            '<tr><th>Record</th><th class="num">Region</th>'
            '<th class="num">Length</th><th class="num">Windows</th>'
            '<th class="num">Peak</th><th class="num">Mean</th></tr>',
        ]
    )
    for row in listed:
        record, start, end, length, windows, peak, mean = row
        parts.append(
            f"<tr><td>{escape(str(record))}</td>"
            f'<td class="num">{thousands(int(start))}-{thousands(int(end))}</td>'
            f'<td class="num">{thousands(int(length))}</td>'
            f'<td class="num">{thousands(int(windows))}</td>'
            f'<td class="num">{number(float(peak))}</td>'
            f'<td class="num">{number(float(mean))}</td></tr>'
        )
    parts.append("</table>")
    if len(rows) > len(listed):
        parts.append(
            f'<p class="note">{escape(plural(len(rows) - len(listed), "further region"))} '
            "not listed; the downloadable table holds every one.</p>"
        )
    parts.append(_download(REGION_COLUMNS, rows, "regions.tsv"))
    return parts


def _composition_section(analysis: Analysis) -> list[str]:
    parts = ["<h2>Composition</h2>"]
    if not analysis.tables:
        parts.append(
            "<p>Counts were taken in canonical form, which merges a word with its "
            "reverse complement, so there are no residue frequencies to expect "
            "from and no composition table.</p>"
        )
        return parts

    residue_table = analysis.tables.get(1)
    if residue_table is not None:
        observed = [word for word in residue_table.words if word.count]
        parts.append(
            chart.bars(
                [word.word for word in observed],
                [float(word.count) for word in observed],
                [
                    f"{word.word}: {thousands(word.count)}, {share(word.frequency)}"
                    for word in observed
                ],
                unit="residues",
            )
        )
        parts.append(
            _download(
                COMPOSITION_COLUMNS, composition_rows(residue_table), "residues.tsv"
            )
        )

    pairs = analysis.tables.get(2)
    if pairs is not None:
        parts.extend(_pair_grid(analysis, pairs))

    triples = analysis.tables.get(3)
    if triples is not None:
        parts.extend(_extremes(triples, analysis))
    return parts


def _pair_grid(analysis: Analysis, pairs: Table) -> list[str]:
    alphabet = analysis.alphabet
    order = sorted(alphabet.core)
    lookup = {word.word: word for word in pairs.words}
    values: list[list[float | None]] = [
        [
            (lookup[first + second].ratio if first + second in lookup else None)
            for second in order
        ]
        for first in order
    ]
    parts = [
        "<h3>Pairs, observed against expected</h3>",
        chart.grid(order, order, values, centre=1.0),
        '<p class="note">Each cell is a pair\'s count divided by the count its '
        "residue frequencies imply: the odds ratio of Karlin and Burge. One is a "
        "pair occurring exactly as often as composition predicts, and a cell with "
        "no value is a pair too rare for the ratio to mean anything. The row is "
        "the first residue of the pair, the column the second.</p>",
    ]
    if alphabet.is_nucleotide:
        cg = lookup.get("CG")
        if cg is not None and cg.ratio is not None:
            parts.append(
                f'<p class="note">CpG sits at {escape(ratio(cg.ratio))} of its '
                f"expectation ({escape(thousands(cg.count))} observed against "
                f"{escape(number(cg.expected, 0))} expected). Vertebrate nuclear "
                "DNA runs near 0.2 through deamination of methylated cytosine; a "
                "plasmid or a synthetic payload has whatever its design gave it, "
                "and CpG content is an immunogenicity consideration for a vector "
                "in its own right.</p>"
            )
    parts.append(
        _download(COMPOSITION_COLUMNS, composition_rows(pairs), "pairs.tsv")
    )
    return parts


def _extremes(triples: Table, analysis: Analysis) -> list[str]:
    over, under = triples.extremes(10)
    parts = ["<h3>Triples furthest from expectation</h3>"]
    if not over and not under:
        parts.append(
            '<p class="note">No triple occurs often enough for its ratio to be '
            "assessed.</p>"
        )
        return parts
    parts.extend(
        [
            "<table>",
            '<tr><th>Word</th><th class="num">Count</th><th class="num">Expected</th>'
            '<th class="num">Ratio</th><th class="num">z</th><th>Direction</th></tr>',
        ]
    )
    for word in list(over) + [word for word in under if word not in over]:
        parts.append(
            f'<tr><td class="mono">{escape(word.word)}</td>'
            f'<td class="num">{thousands(word.count)}</td>'
            f'<td class="num">{number(word.expected, 1)}</td>'
            f'<td class="num">{ratio(word.ratio)}</td>'
            f'<td class="num">{number(word.z_score, 1)}</td>'
            f"<td>{escape(word.direction)}</td></tr>"
        )
    parts.append("</table>")
    parts.append(
        '<p class="note">Expectations are from residue frequencies alone, so a '
        "triple built from over-used pairs is over-represented here for that "
        "reason; the z-score is the Poisson approximation and is indicative only, "
        "since overlapping windows are not independent and the expectation is "
        "fitted from the same sequence.</p>"
    )
    parts.append(
        _download(
            COMPOSITION_COLUMNS, composition_rows(triples), f"triples-{analysis.alphabet.name}.tsv"
        )
    )
    return parts


def _repeat_section(analysis: Analysis) -> list[str]:
    rows = repeat_rows(analysis)
    parts = ["<h2>Repeats</h2>"]
    if not rows:
        parts.append(
            '<p class="note">No homopolymer run of the minimum length and no '
            "perfect tandem repeat of the periods searched.</p>"
        )
        return parts
    ordered = sorted(rows, key=lambda row: (-int(row[6]), str(row[0]), int(row[4])))
    listed = ordered[:MAX_LISTED]
    parts.append(
        '<p class="note">Perfect repeats only, longest first. A run is a '
        f"stretch of {escape(plural(analysis.min_run, 'residue'))} or more of one "
        "residue; a tandem repeat is a stretch in which a unit of up to "
        f"{escape(plural(analysis.max_period, 'residue'))} recurs without "
        "interruption. A repeat with a single substitution in it is reported "
        "here as two.</p>"
    )
    table_parts = [
            "<table>",
            '<tr><th>Record</th><th>Kind</th><th>Unit</th>'
            '<th class="num">Period</th><th class="num">Region</th>'
            '<th class="num">Length</th><th class="num">Copies</th></tr>',
    ]
    for row in listed:
        record, kind, unit, period, start, end, length, copies = row
        table_parts.append(
            f"<tr><td>{escape(str(record))}</td><td>{escape(str(kind))}</td>"
            f'<td class="mono">{escape(str(unit))}</td>'
            f'<td class="num">{period}</td>'
            f'<td class="num">{thousands(int(start))}-{thousands(int(end))}</td>'
            f'<td class="num">{thousands(int(length))}</td>'
            f'<td class="num">{number(float(copies), 1)}</td></tr>'
        )
    table_parts.append("</table>")
    if len(ordered) > len(listed):
        table_parts.append(
            f'<p class="note">{escape(plural(len(ordered) - len(listed), "further repeat"))} '
            "not listed; the downloadable table holds every one.</p>"
        )
    parts.extend(
        _folded(
            f"{plural(len(listed), 'repeat')} listed of {plural(len(rows), 'found')}",
            table_parts,
            open_by_default=len(listed) <= 12,
        )
    )
    parts.append(_download(REPEAT_COLUMNS, rows, "repeats.tsv"))
    return parts


def _word_section(analysis: Analysis) -> list[str]:
    if not analysis.abundant:
        return []
    parts = ["<h3>Where the abundant words are</h3>"]
    if not analysis.positions_reported:
        parts.append(
            '<p class="note">Positions belong to a record, and this input holds '
            "several, so the abundant words are reported with counts only.</p>"
        )
        return parts
    parts.extend(
        [
            "<table>",
            '<tr><th>Word</th><th class="num">Count</th><th class="num">Share</th>'
            '<th class="num">First</th><th class="num">Last</th>'
            "<th>Positions</th></tr>",
        ]
    )
    for occurrence in analysis.abundant:
        where = occurrence.positions
        shown = ", ".join(thousands(position) for position in where[:8])
        if occurrence.truncated or len(where) > 8:
            shown += ", ..."
        parts.append(
            f'<tr><td class="mono">{escape(occurrence.word)}</td>'
            f'<td class="num">{thousands(occurrence.count)}</td>'
            f'<td class="num">{share(occurrence.frequency)}</td>'
            f'<td class="num">{thousands(where[0]) if where else "n/a"}</td>'
            f'<td class="num">{thousands(where[-1]) if where else "n/a"}</td>'
            f'<td class="mono">{escape(shown)}</td></tr>'
        )
    parts.append("</table>")
    return parts


def _record_section(analysis: Analysis) -> list[str]:
    rows = record_rows(analysis)
    parts = ["<h2>Records</h2>"]
    listed = rows[:MAX_LISTED]
    parts.extend(
        [
            "<table>",
            '<tr><th>Record</th><th class="num">Length</th>'
            '<th class="num">Counted</th><th class="num">Ambiguous</th>'
            '<th class="num">Entropy</th><th class="num">Regions</th>'
            '<th class="num">Masked</th><th class="num">Longest repeat</th></tr>',
        ]
    )
    for record in analysis.records[: len(listed)]:
        longest = record.longest_tandem
        repeat = (
            f"{longest.unit} \u00d7 {number(longest.copies, 1)}" if longest else "none"
        )
        parts.append(
            f"<tr><td>{escape(record.identifier)}</td>"
            f'<td class="num">{thousands(record.length)}</td>'
            f'<td class="num">{thousands(record.countable)}</td>'
            f'<td class="num">{thousands(record.ambiguous)}</td>'
            f'<td class="num">{number(record.entropy)}</td>'
            f'<td class="num">{thousands(len(record.regions))}</td>'
            f'<td class="num">{share(record.masked_share)}</td>'
            f'<td class="num mono">{escape(repeat)}</td></tr>'
        )
    parts.append("</table>")
    if len(rows) > len(listed):
        parts.append(
            f'<p class="note">{escape(plural(len(rows) - len(listed), "further record"))} '
            "not listed; the downloadable table holds every one.</p>"
        )
    parts.append(_download(RECORD_COLUMNS, rows, "records.tsv"))
    return parts


def _caveats(analysis: Analysis) -> list[str]:
    notes = analysis.qualifications
    if not notes:
        return []
    parts = ['<div class="caveat">']
    for note in notes:
        parts.append(f"<p>{escape(note)}</p>")
    parts.append("</div>")
    return parts


def render_html(
    analysis: Analysis,
    parameters: Mapping[str, str],
    version: str,
) -> str:
    """Render the complete, self-contained dashboard for one input."""
    spectrum = analysis.spectrum
    gc = gc_content(analysis.spectra[1])
    lede = [
        analysis.alphabet.label,
        residues(analysis.length, analysis.alphabet),
        plural(len(analysis.records), "record"),
        f"k={analysis.k}",
    ]
    if gc is not None:
        lede.append(f"GC={share(gc)}")

    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>Word composition of {escape(analysis.name)}</title>",
        f"<style>{_STYLE}</style>",
        "</head>",
        "<body>",
        f"<h1>Word composition of {escape(analysis.name)}</h1>",
        f'<p class="lede">{escape(_DOT.join(lede))}</p>',
    ]
    parts.extend(_caveats(analysis))

    parts.append("<h2>At a glance</h2>")
    parts.extend(
        _definitions(
            [
                (
                    f"Words of {analysis.k}",
                    f"{thousands(spectrum.distinct)} distinct of "
                    f"{thousands(spectrum.observable)} observable, in "
                    f"{thousands(spectrum.windows)} windows",
                ),
                (
                    "Entropy",
                    f"{bits(analysis.entropies[analysis.k].bits)} of "
                    f"{bits(analysis.entropies[analysis.k].ceiling)} possible",
                ),
                ("Linguistic complexity", number(analysis.linguistic, 4)),
                (
                    "Windows in a repeated word",
                    f"{share(repeat_share(spectrum))} against a floor of "
                    f"{share(forced_repeat_share(spectrum))} that this many "
                    "windows force",
                ),
                (
                    "Low-complexity",
                    f"{plural(analysis.regions, 'region')} covering "
                    f"{share(analysis.masked_share)} of the input",
                ),
            ]
        )
    )

    parts.append("<h2>Parameters</h2>")
    parts.extend(_definitions(list(parameters.items())))

    parts.extend(_spectrum_section(analysis))
    parts.extend(_word_section(analysis))
    parts.extend(_complexity_section(analysis))
    parts.extend(_track_section(analysis))
    parts.extend(_region_section(analysis))
    parts.extend(_composition_section(analysis))
    parts.extend(_repeat_section(analysis))
    parts.extend(_record_section(analysis))

    parts.extend(
        [
            "<footer>",
            "<p>Positions are 1-based and inclusive. Counts are taken over a "
            "window advancing one residue at a time, so neighbouring windows "
            "overlap and the counts are not independent observations.</p>",
            "<p>The DUST score follows Morgulis, Gertz, Sch&auml;ffer and "
            "Agarwala (2006), taken here on a fixed window rather than by their "
            "search for perfect intervals. Protein complexity is the K2 of "
            "Wootton and Federhen (1993), which SEG triggers on at 2.2 bits in a "
            "window of twelve. Odds ratios follow Karlin and Burge (1995), and "
            "linguistic complexity Trifonov (1990).</p>",
            f"<p>kmer-counter {escape(version)}</p>",
            "</footer>",
            "</body>",
            "</html>",
            "",
        ]
    )
    return "\n".join(parts)
