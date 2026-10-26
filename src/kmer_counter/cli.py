"""Command-line interface."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import IO, Any, ContextManager, Sequence

from kmer_counter import __version__
from kmer_counter.alphabet import CHOICES
from kmer_counter.analysis import (
    ALL_TRACKS,
    DEFAULT_K,
    DEFAULT_TRACKS,
    Analysis,
    analyse,
)
from kmer_counter.composition import gc_content
from kmer_counter.distance import MEASURES, Matrix, matrix
from kmer_counter.reader import FORMATS, SequenceError, source_name
from kmer_counter.repeats import (
    MAX_PERIOD,
    MIN_RUN,
    forced_repeat_share,
    repeat_share,
)
from kmer_counter.report import render_html
from kmer_counter.rows import (
    COMPOSITION_COLUMNS,
    DISTANCE_COLUMNS,
    ENTROPY_COLUMNS,
    REGION_COLUMNS,
    REPEAT_COLUMNS,
    SPECTRUM_COLUMNS,
    SUMMARY_COLUMNS,
    WINDOW_COLUMNS,
    composition_rows,
    distance_rows,
    entropy_rows,
    region_rows,
    render,
    repeat_rows,
    spectrum_rows,
    summary_row,
    window_rows,
)
from kmer_counter.spectrum import K_MAX, K_MIN
from kmer_counter.text import (
    bar,
    bits,
    number,
    plural,
    residues,
    share,
    thousands,
    truncate,
)

#: What may be written to standard output instead of a table.
STDOUT_FORMATS: tuple[str, ...] = ("table", "tsv", "json", "none")

#: Words listed in the terminal summary.
_TERMINAL_WORDS: int = 8

#: Distinct words a JSON document carries in full before it is truncated to
#: the abundant ones. Beyond this the spectrum belongs in a file.
_JSON_WORDS: int = 4096


def _non_negative(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"must not be negative: {value}")
    return value


def _positive(text: str) -> int:
    value = _non_negative(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or greater: {value}")
    return value


def _word_length(text: str) -> int:
    value = _positive(text)
    if value > K_MAX:
        raise argparse.ArgumentTypeError(
            f"k must be between {K_MIN} and {K_MAX}: {value}"
        )
    return value


def _score(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if value != value or value in (float("inf"), float("-inf")):
        raise argparse.ArgumentTypeError(f"must be a finite number: {text!r}")
    if value < 0:
        raise argparse.ArgumentTypeError(f"must not be negative: {value}")
    return value


def _add_input_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        action="append",
        metavar="PATH",
        help="file to read, or '-' for standard input; repeat for several files",
    )
    parser.add_argument(
        "--input-format",
        choices=FORMATS,
        default="auto",
        help="input format; 'auto' sniffs '>' or '@' from the first line (default: auto)",
    )
    parser.add_argument(
        "--alphabet",
        choices=CHOICES,
        default="auto",
        help=(
            "residues to count; 'auto' reads the alphabet from the first records, "
            "calling a sequence nucleotide when at least 90%% of its letters are "
            "A, C, G, T, U or N (default: auto)"
        ),
    )
    parser.add_argument(
        "--limit",
        type=_non_negative,
        default=0,
        metavar="N",
        help="stop after the first N records of each input",
    )


def _add_window_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--window",
        type=_positive,
        metavar="N",
        help=(
            "window for the complexity profile (default: 64 for nucleotide, 12 "
            "for protein, the conventional windows of the two measures)"
        ),
    )
    parser.add_argument(
        "--step",
        type=_positive,
        default=1,
        metavar="N",
        help=(
            "advance between reported windows; every window is scored whatever "
            "the step, so this changes the size of the output and not the "
            "regions called (default: 1)"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=_score,
        metavar="SCORE",
        help=(
            "score at which a window is called low-complexity (default: 20 for "
            "the DUST score, 2.2 bits for protein complexity). Both defaults "
            "belong with the default window; a shorter window cannot reach 20"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="kmer-counter",
        description=(
            "Count the words of a sequence and measure how much it repeats "
            "itself: k-mer spectra, Shannon entropy and linguistic complexity, "
            "word composition against expectation, and low-complexity regions "
            "along the sequence."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"kmer-counter {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    count = subparsers.add_parser(
        "count",
        help="count words and measure complexity",
        description=(
            "Count every word length from one up to the length asked for, and "
            "report the spectrum, the entropy, the composition against "
            "expectation and the repeats of each input. Compressed input is "
            "recognised from its magic bytes, so gzip, bzip2 and xz files need "
            "no flag and no particular file name."
        ),
    )
    _add_input_options(count)
    count.add_argument(
        "-k",
        "--k",
        type=_word_length,
        default=DEFAULT_K,
        metavar="K",
        help=(
            f"word length to report, from {K_MIN} to {K_MAX}; every shorter "
            "length is counted too, since linguistic complexity is a product "
            f"over lengths and an expectation needs them (default: {DEFAULT_K})"
        ),
    )
    count.add_argument(
        "--canonical",
        action="store_true",
        help=(
            "count a word and its reverse complement as one word, under the "
            "lexicographically smaller of the two; nucleotide input only, and "
            "it leaves no residue frequencies to expect from, so the "
            "composition tables are omitted"
        ),
    )
    _add_window_options(count)
    count.add_argument(
        "--tracks",
        type=_non_negative,
        default=DEFAULT_TRACKS,
        metavar="N",
        help=(
            "how many records keep a full window profile for the report; "
            "regions and per-record figures cover every record whatever this "
            f"is (default: {DEFAULT_TRACKS})"
        ),
    )
    count.add_argument(
        "--top",
        type=_non_negative,
        default=10,
        metavar="N",
        help="how many abundant words to report (default: 10)",
    )
    count.add_argument(
        "--min-run",
        type=_positive,
        default=MIN_RUN,
        metavar="N",
        help=f"shortest homopolymer run to report (default: {MIN_RUN})",
    )
    count.add_argument(
        "--max-period",
        type=_positive,
        default=MAX_PERIOD,
        metavar="N",
        help=f"longest tandem unit to search for (default: {MAX_PERIOD})",
    )
    count.add_argument(
        "--no-bars",
        action="store_true",
        help="omit the word-abundance bars from the terminal summary",
    )
    _add_output_options(count, "the spectrum of the reported word length")
    count.add_argument(
        "--report",
        metavar="PATH",
        help="write the self-contained HTML dashboard here, or '-' for standard output",
    )

    profile = subparsers.add_parser(
        "profile",
        help="score complexity along each sequence",
        description=(
            "Slide a window along each record, score it, and merge the windows "
            "that score as low-complexity into regions. The score is the DUST "
            "score for nucleotide input and Wootton-Federhen complexity in bits "
            "for protein."
        ),
    )
    _add_input_options(profile)
    _add_window_options(profile)
    _add_output_options(profile, "one row per scored window")
    profile.add_argument(
        "--regions",
        metavar="PATH",
        help="write the low-complexity regions as a TSV here, or '-' for standard output",
    )
    profile.add_argument(
        "--bed",
        metavar="PATH",
        help=(
            "write the low-complexity regions as BED here, or '-' for standard "
            "output; BED starts are 0-based and ends are exclusive, so a region "
            "reported as 101-164 is written 100 164"
        ),
    )

    compare = subparsers.add_parser(
        "compare",
        help="measure the distance between the spectra of several inputs",
        description=(
            "Compare inputs by their word frequencies, without aligning them. "
            "Two sequences built from the same words in a different order are "
            "identical to every measure here, so a small distance is evidence "
            "of similar composition and not of homology."
        ),
    )
    _add_input_options(compare)
    compare.add_argument(
        "-k",
        "--k",
        type=_word_length,
        default=DEFAULT_K,
        metavar="K",
        help=f"word length to compare at (default: {DEFAULT_K})",
    )
    compare.add_argument(
        "--measure",
        choices=MEASURES,
        default="cosine",
        help="distance to take between two spectra (default: cosine)",
    )
    compare.add_argument(
        "--canonical",
        action="store_true",
        help="count a word and its reverse complement as one word",
    )
    _add_output_options(compare, "one row per pair of inputs")
    return parser


def _add_output_options(parser: argparse.ArgumentParser, what: str) -> None:
    parser.add_argument(
        "-f",
        "--format",
        choices=STDOUT_FORMATS,
        default="table",
        help="what to write to standard output (default: table)",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help=f"write {what} as a TSV here, or '-' for standard output",
    )


def _parameters(arguments: argparse.Namespace, analysis: Analysis) -> dict[str, str]:
    """The settings a result was produced under, for the report."""
    rows = {
        "Input": analysis.name,
        "Format": analysis.input_format.upper(),
        "Alphabet": (
            f"{analysis.alphabet.label} (detected)"
            if analysis.detection is not None
            else f"{analysis.alphabet.label} (given)"
        ),
        "Word length": f"k={analysis.k}, counted from 1",
        "Counting": "canonical" if analysis.canonical else "both strands apart",
        "Profile window": f"{thousands(analysis.window)} residues, step {thousands(analysis.step)}",
        "Low-complexity threshold": (
            f"{analysis.measure} "
            f"{'>=' if analysis.alphabet.is_nucleotide else '<='} "
            f"{number(analysis.threshold)}"
        ),
        "Repeats": (
            f"runs of {thousands(analysis.min_run)} or more, tandem units to "
            f"{thousands(analysis.max_period)} residues"
        ),
        "kmer-counter": __version__,
    }
    if arguments.limit:
        rows["Record limit"] = thousands(arguments.limit)
    return rows


def _json_payload(analysis: Analysis) -> dict[str, Any]:
    """Every measurement of one input, as JSON-ready data."""
    spectrum = analysis.spectrum
    complete = spectrum.distinct <= _JSON_WORDS
    words = (
        {word: count for word, count in spectrum.sorted_items()}
        if complete
        else {word: count for word, count in spectrum.most_common(len(analysis.abundant))}
    )
    return {
        "source": analysis.name,
        "format": analysis.input_format,
        "alphabet": {
            "name": analysis.alphabet.name,
            "label": analysis.alphabet.label,
            "size": analysis.alphabet.size,
            "detected": analysis.detection is not None,
            "nucleotide_share": (
                analysis.detection.nucleotide_share if analysis.detection else None
            ),
            "certain": analysis.detection.certain if analysis.detection else True,
        },
        "parameters": {
            "k": analysis.k,
            "canonical": analysis.canonical,
            "window": analysis.window,
            "step": analysis.step,
            "threshold": analysis.threshold,
            "measure": analysis.measure,
        },
        "totals": {
            "records": len(analysis.records),
            "length": analysis.length,
            "ambiguous": analysis.ambiguous,
            "gc": gc_content(analysis.spectra[1]),
            "linguistic_complexity": analysis.linguistic,
            "repeat_share": repeat_share(spectrum),
            "forced_repeat_share": forced_repeat_share(spectrum),
            "regions": analysis.regions,
            "masked": analysis.masked,
            "masked_share": analysis.masked_share,
        },
        "lengths": [
            dict(zip(ENTROPY_COLUMNS, row)) for row in entropy_rows(analysis)
        ],
        "spectrum": {
            "k": analysis.k,
            "windows": spectrum.windows,
            "skipped_windows": spectrum.skipped,
            "distinct": spectrum.distinct,
            "observable": spectrum.observable,
            "possible": spectrum.possible,
            "singletons": spectrum.singletons(),
            "words": words,
            "words_complete": complete,
        },
        "abundant": [
            {
                "word": occurrence.word,
                "count": occurrence.count,
                "frequency": occurrence.frequency,
                "positions": list(occurrence.positions),
                "positions_truncated": occurrence.truncated,
            }
            for occurrence in analysis.abundant
        ],
        "tables": {
            str(length): {
                "model": listing.model,
                "words": [
                    dict(zip(COMPOSITION_COLUMNS, row))
                    for row in composition_rows(listing)
                ],
            }
            for length, listing in sorted(analysis.tables.items())
            if len(listing.words) <= _JSON_WORDS
        },
        "tables_omitted": [
            length
            for length, listing in sorted(analysis.tables.items())
            if len(listing.words) > _JSON_WORDS
        ],
        "signature": analysis.signature,
        "regions": [dict(zip(REGION_COLUMNS, row)) for row in region_rows(analysis)],
        "repeats": [dict(zip(REPEAT_COLUMNS, row)) for row in repeat_rows(analysis)],
        "records": [
            {
                "record": record.identifier,
                "description": record.description,
                "length": record.length,
                "countable": record.countable,
                "ambiguous": record.ambiguous,
                "gaps": record.gaps,
                "stops": record.stops,
                "entropy_bits": record.entropy,
                "regions": len(record.regions),
                "masked": record.masked,
                "masked_share": record.masked_share,
            }
            for record in analysis.records
        ],
        "qualifications": analysis.qualifications,
    }


def _profile_payload(analysis: Analysis) -> dict[str, Any]:
    """The window profile of one input, as JSON-ready data."""
    return {
        "source": analysis.name,
        "alphabet": analysis.alphabet.name,
        "measure": analysis.measure,
        "parameters": {
            "window": analysis.window,
            "step": analysis.step,
            "threshold": analysis.threshold,
        },
        "totals": {
            "records": len(analysis.records),
            "length": analysis.length,
            "regions": analysis.regions,
            "masked": analysis.masked,
            "masked_share": analysis.masked_share,
        },
        "windows": [dict(zip(WINDOW_COLUMNS, row)) for row in window_rows(analysis.tracks)],
        "regions": [dict(zip(REGION_COLUMNS, row)) for row in region_rows(analysis)],
        "qualifications": analysis.alphabet_notes + analysis.profile_notes,
    }


def _summary(analysis: Analysis, bars: bool, stream: IO[str]) -> None:
    """Print the readable summary of one input."""
    spectrum = analysis.spectrum
    entropy = analysis.entropies[analysis.k]
    gc = gc_content(analysis.spectra[1])

    head = [
        analysis.input_format.upper(),
        plural(len(analysis.records), "record"),
        residues(analysis.length, analysis.alphabet),
        analysis.alphabet.label,
    ]
    if gc is not None:
        head.append(f"GC={share(gc)}")
    if analysis.ambiguous:
        head.append(f"uncounted={thousands(analysis.ambiguous)}")
    if analysis.canonical:
        head.append("canonical")
    print(f"> {analysis.name}", file=stream)
    print(f"  {'  '.join(head)}", file=stream)

    print(
        f"  words    k={analysis.k}  windows={thousands(spectrum.windows)}"
        f"  distinct={thousands(spectrum.distinct)} of "
        f"{thousands(spectrum.observable)} observable"
        f"  saturation={share(spectrum.saturation)}",
        file=stream,
    )
    print(
        f"  entropy  {bits(entropy.bits)} of {bits(entropy.ceiling)} possible"
        f"  ({share(entropy.efficiency)})"
        f"  linguistic={number(analysis.linguistic, 4)}"
        f"  repeated={share(repeat_share(spectrum))}"
        f" (floor {share(forced_repeat_share(spectrum))})",
        file=stream,
    )
    comparison = (
        ">=" if analysis.alphabet.is_nucleotide else "<="
    )
    print(
        f"  regions  {plural(analysis.regions, 'region')} covering "
        f"{share(analysis.masked_share)}"
        f"  ({analysis.measure}{comparison}{number(analysis.threshold)},"
        f" window {thousands(analysis.window)})",
        file=stream,
    )

    repeats = repeat_rows(analysis)
    if repeats:
        longest = max(repeats, key=lambda row: (int(row[6]), -int(row[4])))
        print(
            f"  repeats  longest {longest[1]} {longest[2]} x "
            f"{number(float(longest[7]), 1)} at "
            f"{thousands(int(longest[4]))}-{thousands(int(longest[5]))}"
            f"  ({thousands(len(repeats))} in all)",
            file=stream,
        )

    if bars and analysis.abundant:
        widest = max(occurrence.count for occurrence in analysis.abundant)
        print(f"  most abundant words of {analysis.k}", file=stream)
        for occurrence in analysis.abundant[:_TERMINAL_WORDS]:
            print(
                f"    {occurrence.word:>{max(4, analysis.k)}}"
                f"  {thousands(occurrence.count):>9}"
                f"  {bar(occurrence.count, widest)}",
                file=stream,
            )

    for note in analysis.qualifications:
        print(f"  ! {note}", file=stream)


def _profile_summary(analysis: Analysis, stream: IO[str]) -> None:
    """Print the readable summary of one profile."""
    comparison = ">=" if analysis.alphabet.is_nucleotide else "<="
    print(f"> {analysis.name}", file=stream)
    print(
        f"  {analysis.alphabet.label}  {plural(len(analysis.records), 'record')}"
        f"  {residues(analysis.length, analysis.alphabet)}"
        f"  window={thousands(analysis.window)}  step={thousands(analysis.step)}"
        f"  {analysis.measure}{comparison}{number(analysis.threshold)}",
        file=stream,
    )
    print(
        f"  {plural(analysis.regions, 'region')} covering "
        f"{thousands(analysis.masked)} residues ({share(analysis.masked_share)})",
        file=stream,
    )
    for record in analysis.records:
        if not record.regions:
            continue
        print(f"  {truncate(record.identifier, 28):<28}", file=stream)
        for region in record.regions[:10]:
            print(
                f"    {thousands(region.start):>10}-{thousands(region.end):<10}"
                f"  {plural(region.length, 'residue')}"
                f"  peak={number(region.peak)}  mean={number(region.mean)}",
                file=stream,
            )
        if len(record.regions) > 10:
            print(
                f"    {plural(len(record.regions) - 10, 'further region')}; "
                "the TSV holds every one",
                file=stream,
            )
    for note in analysis.alphabet_notes + analysis.profile_notes:
        print(f"  ! {note}", file=stream)


def _matrix_summary(result: Matrix, stream: IO[str]) -> None:
    """Print a distance matrix as a readable table."""
    width = max(len(name) for name in result.names)
    print(
        f"  {result.measure} distance between {len(result.names)} spectra at "
        f"k={result.k}",
        file=stream,
    )
    header = " " * (width + 2) + "  ".join(
        f"{truncate(name, 8):>8}" for name in result.names
    )
    print(header, file=stream)
    for row, name in enumerate(result.names):
        cells = "  ".join(
            f"{number(value, 4):>8}" if value is not None else f"{'n/a':>8}"
            for value in result.values[row]
        )
        print(f"  {name:<{width}}{cells}", file=stream)
    closest = result.closest()
    if closest:
        first, second, value = closest
        print(
            f"  closest: {first} and {second} at {number(value, 4)}",
            file=stream,
        )


def _writer(destination: str, stdout: IO[str]) -> ContextManager[IO[str]]:
    """Open an output path for writing, or hand back standard output for '-'."""
    if destination == "-":
        return contextlib.nullcontext(stdout)
    return open(destination, "w", encoding="utf-8", newline="\n")


def _check_outputs(destinations: Sequence[str | None], inputs: Sequence[str]) -> None:
    """Refuse an output that cannot be written, or that would eat something.

    Checked before anything is read, so a run that cannot finish does not
    start, and checked against every final destination the run will write -
    the per-input report names included, since those are the ones a caller
    never typed and so never checked.

    Three things are refused. An output that names an input under any name: a
    hard link and a symlink are the same file, and comparing the names alone
    would let a run truncate what it is reading. Two outputs that name one
    file, which would otherwise leave whichever is written last and silently
    lose the other. And two outputs sharing standard output, since two
    documents cannot share one stream.
    """
    named = [destination for destination in destinations if destination]
    to_stdout = [destination for destination in named if destination == "-"]
    if len(to_stdout) > 1:
        raise SequenceError(
            "two outputs cannot share standard output; write one of them to a file"
        )
    seen: dict[Path, str] = {}
    for destination in named:
        if destination == "-":
            continue
        path = Path(destination)
        if path.is_dir():
            raise SequenceError(f"output is a directory: {destination}")
        parent = path.parent
        if not parent.exists():
            raise SequenceError(f"no such directory: {parent}")
        # resolve() normalises a path that does not exist yet and follows a
        # link that does, so two spellings of one file compare equal here.
        resolved = path.resolve()
        if resolved in seen:
            first = seen[resolved]
            also = f" and {destination}" if first != destination else " twice"
            raise SequenceError(
                f"two outputs name the same file: {first}{also}; "
                "the second would replace the first"
            )
        seen[resolved] = destination
        for source in inputs:
            if source == "-":
                continue
            source_path = Path(source)
            if not source_path.exists():
                continue
            same = resolved == source_path.resolve() or (
                path.exists() and os.path.samefile(path, source)
            )
            if same:
                raise SequenceError(
                    f"output {destination} is the input {source}; writing it "
                    "would truncate what is being read"
                )


def _report_targets(report: str | None, inputs: Sequence[str]) -> list[str]:
    """Where each input's report will be written, in input order.

    Empty when no report was asked for. With one input the report goes where
    it was asked for; with several, each gets a name of its own, worked out
    here rather than at writing time so the preflight can check them.
    """
    if not report:
        return []
    if len(inputs) == 1:
        return [report]
    return [_per_input(report, source_name(source)) for source in inputs]


def _check_inputs(sources: Sequence[str]) -> None:
    """Refuse an input that cannot be read, before any is read."""
    if sources.count("-") > 1:
        raise SequenceError("standard input can only be read once")
    for source in sources:
        if source == "-":
            continue
        path = Path(source)
        if path.is_dir():
            raise SequenceError(f"not a file: {source}")
        if not path.exists():
            raise SequenceError(f"no such file: {source}")


def _stdout_shared(arguments: argparse.Namespace) -> bool:
    """Whether a file output is going to standard output alongside a summary."""
    return any(
        getattr(arguments, name, None) == "-"
        for name in ("output", "report", "regions", "bed")
    )


def _run_count(arguments: argparse.Namespace, stdout: IO[str]) -> int:
    _check_inputs(arguments.input)
    reports = _report_targets(arguments.report, arguments.input)
    if arguments.report == "-" and len(arguments.input) > 1:
        raise SequenceError(
            "several inputs cannot share one report on standard output; "
            "write the report to a file, which is named per input"
        )
    _check_outputs([arguments.output, *reports], arguments.input)
    if _stdout_shared(arguments) and arguments.format != "none":
        raise SequenceError(
            "an output is going to standard output, so the summary has to be "
            "silenced with --format none"
        )

    analyses = [
        analyse(
            source,
            k=arguments.k,
            alphabet=arguments.alphabet,
            canonical=arguments.canonical,
            input_format=arguments.input_format,
            window=arguments.window,
            step=arguments.step,
            threshold=arguments.threshold,
            tracks=arguments.tracks,
            limit=arguments.limit,
            min_run=arguments.min_run,
            max_period=arguments.max_period,
            abundant=arguments.top,
        )
        for source in arguments.input
    ]

    if arguments.output:
        with _writer(arguments.output, stdout) as handle:
            if len(analyses) == 1:
                handle.write(render(SPECTRUM_COLUMNS, spectrum_rows(analyses[0])))
            else:
                handle.write(
                    render(
                        ("source",) + SPECTRUM_COLUMNS,
                        [
                            [analysis.name, *row]
                            for analysis in analyses
                            for row in spectrum_rows(analysis)
                        ],
                    )
                )

    if reports:
        for analysis, destination in zip(analyses, reports):
            document = render_html(
                analysis, _parameters(arguments, analysis), __version__
            )
            with _writer(destination, stdout) as handle:
                handle.write(document)

    if arguments.format == "table":
        for analysis in analyses:
            _summary(analysis, not arguments.no_bars, stdout)
    elif arguments.format == "tsv":
        stdout.write(
            render(SUMMARY_COLUMNS, [summary_row(analysis) for analysis in analyses])
        )
    elif arguments.format == "json":
        payload = [_json_payload(analysis) for analysis in analyses]
        json.dump(payload if len(payload) > 1 else payload[0], stdout, indent=2)
        stdout.write("\n")
    return 0


def _per_input(destination: str, name: str) -> str:
    """Name one input's report when several are written.

    ``--report out.html`` with three inputs writes ``out.aav2.fasta.html`` and
    its siblings, so no input silently overwrites another's report.
    """
    path = Path(destination)
    safe = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in name
    )
    return str(path.with_name(f"{path.stem}.{safe}{path.suffix}"))


def _run_profile(arguments: argparse.Namespace, stdout: IO[str]) -> int:
    _check_inputs(arguments.input)
    _check_outputs(
        [arguments.output, arguments.regions, arguments.bed], arguments.input
    )
    if _stdout_shared(arguments) and arguments.format != "none":
        raise SequenceError(
            "an output is going to standard output, so the summary has to be "
            "silenced with --format none"
        )

    # The window-by-window track is only built when something will render it;
    # the regions are called from every window either way.
    keep = ALL_TRACKS if (arguments.output or arguments.format == "json") else 0
    analyses = [
        analyse(
            source,
            k=1,
            alphabet=arguments.alphabet,
            input_format=arguments.input_format,
            window=arguments.window,
            step=arguments.step,
            threshold=arguments.threshold,
            tracks=keep,
            limit=arguments.limit,
            abundant=0,
            find_repeats=False,
        )
        for source in arguments.input
    ]

    if arguments.output:
        with _writer(arguments.output, stdout) as handle:
            handle.write(
                render(
                    ("source",) + WINDOW_COLUMNS,
                    [
                        [analysis.name, *row]
                        for analysis in analyses
                        for row in window_rows(analysis.tracks)
                    ],
                )
            )

    if arguments.regions:
        with _writer(arguments.regions, stdout) as handle:
            handle.write(
                render(
                    ("source",) + REGION_COLUMNS,
                    [
                        [analysis.name, *row]
                        for analysis in analyses
                        for row in region_rows(analysis)
                    ],
                )
            )

    if arguments.bed:
        with _writer(arguments.bed, stdout) as handle:
            for analysis in analyses:
                for record in analysis.records:
                    for region in record.regions:
                        handle.write(
                            "\t".join(
                                (
                                    record.identifier,
                                    str(region.start - 1),
                                    str(region.end),
                                    f"low_complexity_{analysis.measure}",
                                    f"{region.peak:g}",
                                )
                            )
                            + "\n"
                        )

    if arguments.format == "table":
        for analysis in analyses:
            _profile_summary(analysis, stdout)
    elif arguments.format == "tsv":
        stdout.write(
            render(
                ("source",) + REGION_COLUMNS,
                [
                    [analysis.name, *row]
                    for analysis in analyses
                    for row in region_rows(analysis)
                ],
            )
        )
    elif arguments.format == "json":
        payload = [_profile_payload(analysis) for analysis in analyses]
        json.dump(payload if len(payload) > 1 else payload[0], stdout, indent=2)
        stdout.write("\n")
    return 0


def _run_compare(arguments: argparse.Namespace, stdout: IO[str]) -> int:
    if len(arguments.input) < 2:
        raise SequenceError("comparing needs two or more inputs")
    _check_inputs(arguments.input)
    _check_outputs([arguments.output], arguments.input)
    if _stdout_shared(arguments) and arguments.format != "none":
        raise SequenceError(
            "an output is going to standard output, so the summary has to be "
            "silenced with --format none"
        )

    analyses = [
        analyse(
            source,
            k=arguments.k,
            alphabet=arguments.alphabet,
            canonical=arguments.canonical,
            input_format=arguments.input_format,
            limit=arguments.limit,
            tracks=0,
            abundant=0,
            find_repeats=False,
        )
        for source in arguments.input
    ]
    alphabets = {analysis.alphabet.name for analysis in analyses}
    if len(alphabets) > 1:
        named = ", ".join(sorted(alphabets))
        raise SequenceError(
            f"the inputs are not in one alphabet ({named}); name one with "
            "--alphabet, or compare them separately"
        )

    names = _unique_names([analysis.name for analysis in analyses])
    spectra = {
        name: analysis.spectrum for name, analysis in zip(names, analyses)
    }
    result = matrix(spectra, arguments.measure)

    if arguments.output:
        with _writer(arguments.output, stdout) as handle:
            handle.write(render(DISTANCE_COLUMNS, distance_rows(result)))

    if arguments.format == "table":
        _matrix_summary(result, stdout)
        for analysis in analyses:
            for note in analysis.alphabet_notes + analysis.spectrum_notes:
                print(f"  ! {analysis.name}: {note}", file=stdout)
    elif arguments.format == "tsv":
        stdout.write(render(DISTANCE_COLUMNS, distance_rows(result)))
    elif arguments.format == "json":
        json.dump(
            {
                "measure": result.measure,
                "k": result.k,
                "canonical": arguments.canonical,
                "names": list(result.names),
                "matrix": [list(row) for row in result.values],
                "pairs": [
                    dict(zip(DISTANCE_COLUMNS, row)) for row in distance_rows(result)
                ],
                "qualifications": [
                    f"{analysis.name}: {note}"
                    for analysis in analyses
                    for note in analysis.alphabet_notes + analysis.spectrum_notes
                ],
            },
            stdout,
            indent=2,
        )
        stdout.write("\n")
    return 0


def _unique_names(names: Sequence[str]) -> list[str]:
    """Make the input names distinct, so a matrix can be keyed by them.

    Two directories can hold a file of the same name, and a matrix with two
    identical labels is unreadable; the second becomes 'name (2)'.
    """
    seen: dict[str, int] = {}
    unique = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        unique.append(name if seen[name] == 1 else f"{name} ({seen[name]})")
    return unique


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line, returning the process exit status."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    runners = {
        "count": _run_count,
        "profile": _run_profile,
        "compare": _run_compare,
    }
    try:
        return runners[arguments.command](arguments, sys.stdout)
    except SequenceError as error:
        print(f"kmer-counter: {error}", file=sys.stderr)
        return 2
    except ValueError as error:
        print(f"kmer-counter: {error}", file=sys.stderr)
        return 2
    except (FileNotFoundError, IsADirectoryError, PermissionError) as error:
        print(f"kmer-counter: {error}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        # Downstream closed the pipe, as 'head' does; that is not an error.
        with contextlib.suppress(OSError):
            sys.stdout.close()
        return 0
