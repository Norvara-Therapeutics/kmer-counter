"""The tabular outputs, built once and rendered twice.

Every table the tool can write exists here as columns and rows of text. The
``--output`` file, the tables embedded in the HTML report for download and the
rows printed to standard output are all the same function's result, so a
figure downloaded from the report is the figure the file would have held.

Values are written unrounded: a float is written in the shortest form that
reads back as the same double, never to two decimal places. Rounding belongs
to the human-facing renderings, which call ``text`` for it. A value that does
not exist is written as an empty field rather than as a zero, since a GC
content of 0 and no GC content at all are different facts.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from kmer_counter.analysis import Analysis
from kmer_counter.composition import Table, gc_content
from kmer_counter.distance import Matrix
from kmer_counter.profile import Profile
from kmer_counter.repeats import forced_repeat_share, repeat_share

#: What separates fields. Tab, because a FASTA description may hold a comma.
SEPARATOR: str = "\t"


def field(value: object) -> str:
    """One value, written for a tab-separated file.

    Floats keep full precision, None becomes an empty field, and a tab or
    newline inside a description becomes a space so that one row stays one
    row.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    text = str(value)
    for character in ("\t", "\r", "\n"):
        text = text.replace(character, " ")
    return text


def render(columns: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    """A whole table as text, with its header line and a trailing newline."""
    lines = [SEPARATOR.join(columns)]
    lines.extend(SEPARATOR.join(field(value) for value in row) for row in rows)
    return "\n".join(lines) + "\n"


SPECTRUM_COLUMNS: tuple[str, ...] = ("word", "count", "frequency")


def spectrum_rows(analysis: Analysis) -> list[list[object]]:
    """Every counted word of the reported length, most abundant first."""
    spectrum = analysis.spectrum
    return [
        [word, count, count / spectrum.windows if spectrum.windows else None]
        for word, count in spectrum.sorted_items()
    ]


COMPOSITION_COLUMNS: tuple[str, ...] = (
    "word",
    "count",
    "frequency",
    "expected",
    "ratio",
    "z_score",
    "model",
)


def composition_rows(table: Table) -> list[list[object]]:
    """Every word of one length with its expectation, alphabetically."""
    return [
        [
            word.word,
            word.count,
            word.frequency,
            word.expected,
            word.ratio,
            word.z_score,
            word.model,
        ]
        for word in table.words
    ]


WINDOW_COLUMNS: tuple[str, ...] = (
    "record",
    "start",
    "end",
    "score",
    "entropy_bits",
    "gc",
    "countable",
    "low_complexity",
)


def window_rows(profiles: Sequence[Profile]) -> list[list[object]]:
    """Every reported window of every profiled record."""
    return [
        [
            profile.name,
            window.start,
            window.end,
            window.score,
            window.entropy,
            window.gc,
            window.countable,
            window.low,
        ]
        for profile in profiles
        for window in profile.windows
    ]


REGION_COLUMNS: tuple[str, ...] = (
    "record",
    "start",
    "end",
    "length",
    "windows",
    "peak_score",
    "mean_score",
)


def region_rows(analysis: Analysis) -> list[list[object]]:
    """Every low-complexity region of every record."""
    return [
        [
            record.identifier,
            region.start,
            region.end,
            region.length,
            region.windows,
            region.peak,
            region.mean,
        ]
        for record in analysis.records
        for region in record.regions
    ]


REPEAT_COLUMNS: tuple[str, ...] = (
    "record",
    "kind",
    "unit",
    "period",
    "start",
    "end",
    "length",
    "copies",
)


def repeat_rows(analysis: Analysis) -> list[list[object]]:
    """Every homopolymer run and tandem repeat, in position order per record.

    Runs and tandem repeats are listed together with a 'kind' column rather
    than as two tables, because they are the same kind of finding at
    different periods and a reader scanning for the longest repeat in a
    record should not have to look in two places. A run of five or more
    residues appears as a run; the same region also appears as a tandem
    repeat of period one when it holds three copies or more.
    """
    rows: list[list[object]] = []
    for record in analysis.records:
        findings = [
            (run.start, "run", run.residue, 1, run.start, run.end, run.length, float(run.length))
            for run in record.runs
        ] + [
            (
                repeat.start,
                "tandem",
                repeat.unit,
                repeat.period,
                repeat.start,
                repeat.end,
                repeat.length,
                repeat.copies,
            )
            for repeat in record.tandems
        ]
        for finding in sorted(findings, key=lambda item: (item[0], item[3])):
            rows.append([record.identifier, *finding[1:]])
    return rows


RECORD_COLUMNS: tuple[str, ...] = (
    "source",
    "record",
    "description",
    "length",
    "countable",
    "ambiguous",
    "gaps",
    "stops",
    "entropy_bits",
    "regions",
    "masked",
    "masked_share",
    "longest_run",
    "longest_run_length",
    "longest_tandem_unit",
    "longest_tandem_length",
)


def record_rows(analysis: Analysis) -> list[list[object]]:
    """One row per record of the input."""
    rows: list[list[object]] = []
    for record in analysis.records:
        run = record.longest_run
        tandem = record.longest_tandem
        rows.append(
            [
                analysis.name,
                record.identifier,
                record.description,
                record.length,
                record.countable,
                record.ambiguous,
                record.gaps,
                record.stops,
                record.entropy,
                len(record.regions),
                record.masked,
                record.masked_share,
                run.residue if run else None,
                run.length if run else None,
                tandem.unit if tandem else None,
                tandem.length if tandem else None,
            ]
        )
    return rows


SUMMARY_COLUMNS: tuple[str, ...] = (
    "source",
    "format",
    "alphabet",
    "detected",
    "records",
    "length",
    "k",
    "canonical",
    "windows",
    "skipped_windows",
    "distinct_words",
    "observable_words",
    "possible_words",
    "saturation",
    "entropy_bits",
    "entropy_ceiling_bits",
    "entropy_efficiency",
    "linguistic_complexity",
    "repeat_share",
    "forced_repeat_share",
    "gc",
    "regions",
    "masked",
    "masked_share",
    "window",
    "threshold",
    "measure",
    "qualifications",
)


def summary_row(analysis: Analysis) -> list[object]:
    """One row of whole-input metrics, for a table of several inputs."""
    spectrum = analysis.spectrum
    entropy = analysis.entropies[analysis.k]
    return [
        analysis.name,
        analysis.input_format,
        analysis.alphabet.name,
        analysis.detection is not None,
        len(analysis.records),
        analysis.length,
        analysis.k,
        analysis.canonical,
        spectrum.windows,
        spectrum.skipped,
        spectrum.distinct,
        spectrum.observable,
        spectrum.possible,
        spectrum.saturation,
        entropy.bits,
        entropy.ceiling,
        entropy.efficiency,
        analysis.linguistic,
        repeat_share(spectrum),
        forced_repeat_share(spectrum),
        gc_content(analysis.spectra[1]),
        analysis.regions,
        analysis.masked,
        analysis.masked_share,
        analysis.window,
        analysis.threshold,
        analysis.measure,
        "; ".join(analysis.qualifications),
    ]


ENTROPY_COLUMNS: tuple[str, ...] = (
    "k",
    "windows",
    "skipped_windows",
    "distinct",
    "observable",
    "possible",
    "saturation",
    "entropy_bits",
    "ceiling_bits",
    "efficiency",
    "singletons",
)


def entropy_rows(analysis: Analysis) -> list[list[object]]:
    """One row per counted word length."""
    rows: list[list[object]] = []
    for k, entropy in analysis.entropies.items():
        spectrum = analysis.spectra[k]
        rows.append(
            [
                k,
                spectrum.windows,
                spectrum.skipped,
                spectrum.distinct,
                spectrum.observable,
                spectrum.possible,
                spectrum.saturation,
                entropy.bits,
                entropy.ceiling,
                entropy.efficiency,
                spectrum.singletons(),
            ]
        )
    return rows


DISTANCE_COLUMNS: tuple[str, ...] = ("first", "second", "k", "measure", "distance")


def distance_rows(matrix: Matrix) -> list[list[object]]:
    """Every distinct pair of compared sequences, once."""
    return [
        [first, second, matrix.k, matrix.measure, value]
        for first, second, value in matrix.pairs()
    ]
