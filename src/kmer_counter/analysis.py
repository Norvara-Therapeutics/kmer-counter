"""One input, measured.

This is where the pieces meet: an input is read record by record, its alphabet
is decided once and applied throughout, every word length from one up to the
one asked for is counted, and each record is profiled along its length. The
command-line interface does no measuring of its own, so the terminal summary,
the TSV, the JSON and the HTML report are all renderings of the object this
module returns and cannot drift apart.

Counting every length from one is not waste. Linguistic complexity is a
product over lengths, the Markov expectation of a word needs the counts of its
prefix and suffix, and the residue frequencies that every expectation rests on
are the length-one spectrum.

Two things are deliberately decided once, early, and carried everywhere:

* the alphabet, which is detected from the first records rather than from the
  whole file, because the whole file does not fit in a pipe. What it was
  detected from, and how sure that makes it, travels with the result.
* which records keep their full window profile. The regions called and the
  per-record summary cover every record, but the window-by-window track is
  kept only for the first few, because a file of a million reads would
  otherwise be held in memory as a chart nobody can read.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import IO, Mapping, Sequence

from kmer_counter.alphabet import (
    GAPS,
    STOP,
    Alphabet,
    Detection,
    detect,
    get,
)
from kmer_counter.complexity import (
    Entropy,
    default_threshold,
    default_window,
    entropy_of,
    linguistic_complexity,
    residue_entropy,
    score_name,
)
from kmer_counter.composition import Table, signature, table
from kmer_counter.profile import Interval, Profile, scan, threshold_note
from kmer_counter.reader import Record, SequenceError, read_records, source_name
from kmer_counter.repeats import (
    MAX_PERIOD,
    MIN_RUN,
    Occurrence,
    forced_repeated_windows,
    Run,
    Tandem,
    abundant_words,
    homopolymer_runs,
    tandem_repeats,
)
from kmer_counter.spectrum import Spectrum, check_k, count, merge

#: Word length reported when none is asked for. Four residues of DNA is 256
#: words, which a few kilobases can populate without saturating.
DEFAULT_K: int = 4

#: Composition tables are built for these word lengths, the ones small enough
#: for every word to be listed whether it occurs or not.
TABLE_LENGTHS: tuple[int, ...] = (1, 2, 3)

#: How many records keep their full window profile by default.
DEFAULT_TRACKS: int = 4

#: Pass as ``tracks`` to keep every record's profile, however many there are.
ALL_TRACKS: int = -1

#: How many records may be buffered while the alphabet is being decided. A
#: file of empty records cannot be allowed to buffer without bound.
_DETECTION_RECORDS: int = 1000

#: Residues to see before the alphabet is called with confidence.
_DETECTION_RESIDUES: int = 20


@dataclass(frozen=True)
class RecordSummary:
    """One record, measured.

    Attributes:
        index: the record's position in the input, counting from one.
        identifier: the record's identifier.
        description: the rest of its header line.
        length: how many residues it holds.
        countable: how many of those the alphabet counts.
        ambiguous: how many are letters the alphabet does not count, such as
            an ambiguity code or a residue outside the twenty.
        gaps: how many are alignment gaps.
        stops: how many are translation stops.
        entropy: Shannon entropy of its residue composition, in bits.
        regions: the low-complexity regions called in it.
        runs: its homopolymer runs.
        tandems: its tandem repeats.
        track: its window-by-window profile, kept only for the first records.
    """

    index: int
    identifier: str
    description: str
    length: int
    countable: int
    ambiguous: int
    gaps: int
    stops: int
    entropy: float
    regions: Sequence[Interval] = ()
    runs: Sequence[Run] = ()
    tandems: Sequence[Tandem] = ()
    track: Profile | None = None

    @property
    def masked(self) -> int:
        """How many residues the low-complexity regions cover."""
        return sum(region.length for region in self.regions)

    @property
    def masked_share(self) -> float:
        """Those residues as a share of the record."""
        return self.masked / self.length if self.length else 0.0

    @property
    def longest_run(self) -> Run | None:
        """The longest homopolymer run, earliest first on a tie."""
        if not self.runs:
            return None
        return max(self.runs, key=lambda run: (run.length, -run.start))

    @property
    def longest_tandem(self) -> Tandem | None:
        """The longest tandem repeat, earliest first on a tie."""
        if not self.tandems:
            return None
        return max(self.tandems, key=lambda repeat: (repeat.length, -repeat.start))


@dataclass(frozen=True)
class Analysis:
    """Everything measured from one input.

    Attributes:
        source: the input as it was named on the command line.
        name: how the input is written in output.
        input_format: 'fasta' or 'fastq', as read.
        alphabet: the alphabet everything was counted under.
        detection: how the alphabet was decided, None when it was given.
        k: the longest word counted, and the length the spectrum reports.
        canonical: whether a word and its reverse complement were counted as
            one word.
        records: one summary per record, in input order.
        spectra: the spectrum of every length from one to ``k``, merged over
            the records.
        tables: the composition table of each short word length.
        abundant: the most abundant words of length ``k``.
        positions_reported: whether abundant words carry their positions,
            which is possible only for a single-record input.
        window: the profile window used.
        step: the profile step used.
        threshold: the low-complexity threshold used.
        min_run: the shortest homopolymer run that was reported.
        max_period: the longest tandem unit that was searched for.
    """

    source: str
    name: str
    input_format: str
    alphabet: Alphabet
    detection: Detection | None
    k: int
    canonical: bool
    records: Sequence[RecordSummary]
    spectra: Mapping[int, Spectrum]
    tables: Mapping[int, Table]
    abundant: Sequence[Occurrence]
    positions_reported: bool
    window: int
    step: int
    threshold: float
    min_run: int = MIN_RUN
    max_period: int = MAX_PERIOD
    residues: Mapping[str, int] = field(default_factory=dict)

    @property
    def spectrum(self) -> Spectrum:
        """The spectrum of the reported word length."""
        return self.spectra[self.k]

    @property
    def length(self) -> int:
        """How many residues the input holds."""
        return sum(record.length for record in self.records)

    @property
    def entropies(self) -> dict[int, Entropy]:
        """The entropy of every counted word length."""
        return {k: entropy_of(spectrum) for k, spectrum in sorted(self.spectra.items())}

    @property
    def linguistic(self) -> float | None:
        """Linguistic complexity over every counted word length."""
        return linguistic_complexity(
            [self.spectra[length] for length in sorted(self.spectra)]
        )

    @property
    def signature(self) -> dict[str, float]:
        """The dinucleotide odds ratios, empty when they cannot be taken."""
        if self.canonical or 2 not in self.spectra or 1 not in self.spectra:
            return {}
        return signature(self.spectra)

    @property
    def measure(self) -> str:
        """What the window scores are in: 'dust' or 'complexity'."""
        return score_name(self.alphabet)

    @property
    def regions(self) -> int:
        """How many low-complexity regions were called across the input."""
        return sum(len(record.regions) for record in self.records)

    @property
    def masked(self) -> int:
        """How many residues those regions cover."""
        return sum(record.masked for record in self.records)

    @property
    def masked_share(self) -> float:
        """Those residues as a share of the input."""
        return self.masked / self.length if self.length else 0.0

    @property
    def tracks(self) -> list[Profile]:
        """The window profiles that were kept."""
        return [record.track for record in self.records if record.track is not None]

    @property
    def ambiguous(self) -> int:
        """How many residues the alphabet does not count."""
        return sum(record.ambiguous for record in self.records)

    @property
    def alphabet_notes(self) -> list[str]:
        """What to say about how the alphabet was decided, if anything."""
        if self.detection is None or not self.detection.qualification:
            return []
        return [
            f"alphabet read as {self.alphabet.label}: {self.detection.qualification}"
        ]

    @property
    def spectrum_notes(self) -> list[str]:
        """What a reader must know to read the word counts correctly."""
        notes: list[str] = []
        spectrum = self.spectrum
        if not spectrum.windows:
            return [
                f"no window of length {self.k} could be counted: every record is "
                f"shorter than {self.k} residues, or every window holds a residue "
                f"the {self.alphabet.label} alphabet does not count"
            ]
        if spectrum.skipped:
            notes.append(
                f"{spectrum.skipped:,} of {spectrum.skipped + spectrum.windows:,} "
                f"windows of length {self.k} were not counted, because they hold a "
                f"residue the {self.alphabet.label} alphabet does not count"
            )
        floor = forced_repeated_windows(spectrum.windows, spectrum.possible)
        if floor:
            notes.append(
                f"at k={self.k} there are {spectrum.windows:,} windows for "
                f"{spectrum.possible:,} possible words, so at least {floor:,} of "
                f"them ({floor / spectrum.windows:.1%}) must belong to a repeated "
                "word whatever the sequence; read the share of windows in a "
                "repeated word against that floor, not against zero"
            )
        entropy = entropy_of(spectrum)
        if entropy.length_limited:
            notes.append(
                f"at k={self.k} the sequence has {spectrum.windows:,} windows against "
                f"{spectrum.possible:,} possible words, so its entropy is capped by "
                f"its length at {entropy.ceiling:.2f} bits and is not comparable with "
                "that of a longer sequence"
            )
        if len(self.records) > 1 and not self.positions_reported:
            notes.append(
                f"the input holds {len(self.records):,} records, so abundant words "
                "are reported with counts but without positions, which belong to a "
                "single record"
            )
        return notes

    @property
    def profile_notes(self) -> list[str]:
        """What a reader must know to read the low-complexity calls correctly."""
        notes: list[str] = []
        note = threshold_note(self.alphabet, self.window, self.threshold)
        if note:
            notes.append(f"low-complexity calls: {note}")
        kept = len(self.tracks)
        if kept and len(self.records) > kept:
            notes.append(
                f"window profiles are kept for the first {kept:,} of "
                f"{len(self.records):,} records; regions and per-record figures "
                "cover every record"
            )
        return notes

    @property
    def qualifications(self) -> list[str]:
        """Everything a reader must know to read these numbers correctly.

        Every renderer prints this list, the JSON included: a machine reading
        the spectrum cannot see a warning printed beside it, and a caveat
        shown in one rendering and not another is worse than none at all.
        """
        return self.alphabet_notes + self.spectrum_notes + self.profile_notes


def _census(records: Sequence[Record]) -> dict[str, int]:
    tally: Counter[str] = Counter()
    for record in records:
        tally.update(record.sequence)
    return dict(tally)


def _classify(sequence: str, alphabet: Alphabet) -> tuple[int, int, int, int]:
    """Count core, ambiguous, gap and stop characters in a sequence."""
    core = set(alphabet.core)
    counted = gaps = stops = 0
    for residue in sequence:
        if residue in core:
            counted += 1
        elif residue in GAPS:
            gaps += 1
        elif residue == STOP:
            stops += 1
    ambiguous = len(sequence) - counted - gaps - stops
    return counted, ambiguous, gaps, stops


def check_residues(record: Record, alphabet: Alphabet, requested: str = "auto") -> None:
    """Refuse a record holding residues the alphabet does not accept.

    The alphabet is decided from the first records, so this is what keeps a
    later record honest: a file whose first record is DNA and whose tenth is
    RNA would otherwise have its uracil counted as an ambiguity code and
    quietly skipped, which is a wrong answer rather than an error.

    Raises:
        SequenceError: naming the offending residues and the record.
    """
    unknown = sorted(residue for residue in set(record.sequence) if residue not in alphabet.valid)
    if not unknown:
        return
    shown = ", ".join(repr(residue) for residue in unknown[:5])
    if len(unknown) > 5:
        shown += ", ..."
    hint = (
        "name the alphabet with --alphabet if it was read wrongly"
        if requested == "auto"
        else "check the input, or the alphabet you named"
    )
    raise SequenceError(
        f"record {record.identifier}: the {alphabet.label} alphabet does not "
        f"accept: {shown}; {hint}"
    )


def resolve_alphabet(
    records: Sequence[Record],
    requested: str = "auto",
) -> tuple[Alphabet, Detection | None]:
    """Decide the alphabet from the records read so far, or honour a request.

    Raises:
        SequenceError: if the records hold residues the requested alphabet
            does not accept.
    """
    census = _census(records)
    if requested != "auto":
        alphabet = get(requested)
        detection = None
    else:
        detection = detect(census)
        alphabet = detection.alphabet
    for record in records:
        check_residues(record, alphabet, requested)
    return alphabet, detection


def analyse(
    source: str | IO[str],
    k: int = DEFAULT_K,
    alphabet: str = "auto",
    canonical: bool = False,
    input_format: str = "auto",
    window: int | None = None,
    step: int = 1,
    threshold: float | None = None,
    tracks: int = DEFAULT_TRACKS,
    limit: int = 0,
    min_run: int = MIN_RUN,
    max_period: int = MAX_PERIOD,
    abundant: int = 10,
    find_repeats: bool = True,
) -> Analysis:
    """Measure one input from end to end.

    Args:
        source: path, open text handle, or '-' for standard input.
        k: the longest word length to count, which is also the length the
            spectrum and the abundant words are reported at. Every length
            from one up is counted.
        alphabet: 'auto', 'dna', 'rna' or 'protein'.
        canonical: count a word and its reverse complement as one word.
        input_format: 'auto', 'fasta' or 'fastq'.
        window: the profile window; the alphabet's conventional one by default.
        step: how far the profile window advances between reported windows.
        threshold: the low-complexity threshold; the alphabet's conventional
            one by default.
        tracks: how many records keep their full window profile, or
            ``ALL_TRACKS`` for every record.
        limit: stop after this many records; 0 reads everything.
        min_run: the shortest homopolymer run to report.
        max_period: the longest tandem unit to search for.
        abundant: how many abundant words to report.
        find_repeats: look for homopolymer runs and tandem repeats. Comparing
            two inputs by their spectra needs neither, and on a long sequence
            the search is the slowest thing here.

    Raises:
        SequenceError: if the input cannot be read, or holds residues the
            alphabet does not accept.
        ValueError: if a parameter is out of range.
    """
    check_k(k)
    if tracks < ALL_TRACKS:
        raise ValueError(
            f"tracks must be a count, or {ALL_TRACKS} for every record, got {tracks}"
        )
    if abundant < 0:
        raise ValueError(f"abundant must not be negative, got {abundant}")

    stream = read_records(source, input_format, limit=limit)
    buffered: list[Record] = []
    seen = 0
    for record in stream:
        buffered.append(record)
        seen += len(record.sequence)
        if seen >= _DETECTION_RESIDUES or len(buffered) >= _DETECTION_RECORDS:
            break
    if not buffered:
        raise SequenceError("input contains no records")

    resolved, detection = resolve_alphabet(buffered, alphabet)
    if canonical and not resolved.is_nucleotide:
        raise ValueError(
            f"canonical counting needs a complement, which {resolved.label} has not"
        )
    size = default_window(resolved) if window is None else window
    limit_score = default_threshold(resolved) if threshold is None else threshold

    lengths = list(range(1, k + 1))
    per_length: dict[int, list[Spectrum]] = {length: [] for length in lengths}
    summaries: list[RecordSummary] = []
    residues: Counter[str] = Counter()
    only_sequence: str | None = None

    def take(record: Record, index: int) -> None:
        nonlocal only_sequence
        check_residues(record, resolved, alphabet)
        sequence = record.sequence
        residues.update(sequence)
        for length in lengths:
            per_length[length].append(count(sequence, length, resolved, canonical))
        keep_track = tracks == ALL_TRACKS or index <= tracks
        profiled = scan(
            sequence,
            resolved,
            name=record.identifier,
            window=size,
            step=step,
            threshold=limit_score,
        )
        counted, ambiguous, gaps, stops = _classify(sequence, resolved)
        summaries.append(
            RecordSummary(
                index=index,
                identifier=record.identifier,
                description=record.description,
                length=len(sequence),
                countable=counted,
                ambiguous=ambiguous,
                gaps=gaps,
                stops=stops,
                entropy=residue_entropy(sequence, resolved),
                regions=profiled.intervals,
                runs=homopolymer_runs(sequence, resolved, min_run) if find_repeats else (),
                tandems=(
                    tandem_repeats(sequence, resolved, max_period) if find_repeats else ()
                ),
                track=profiled if keep_track else None,
            )
        )
        if index == 1:
            only_sequence = sequence
        elif index == 2:
            only_sequence = None

    index = 0
    for record in buffered:
        index += 1
        take(record, index)
    for record in stream:
        index += 1
        take(record, index)

    spectra = {length: merge(per_length[length]) for length in lengths}
    tables = {
        length: table(spectra, length, "zero-order")
        for length in TABLE_LENGTHS
        if length <= k and not canonical
    }
    words = (
        abundant_words(spectra[k], only_sequence, abundant)
        if only_sequence is not None and abundant
        else [
            Occurrence(
                word=word,
                count=number,
                frequency=spectra[k].frequency(word),
                positions=(),
                truncated=False,
            )
            for word, number in spectra[k].most_common(abundant)
        ]
    )

    return Analysis(
        source=str(source),
        name=source_name(source),
        input_format="fastq" if buffered[0].quality is not None else "fasta",
        alphabet=resolved,
        detection=detection,
        k=k,
        canonical=canonical,
        records=summaries,
        spectra=spectra,
        tables=tables,
        abundant=words,
        positions_reported=only_sequence is not None,
        window=size,
        step=step,
        threshold=limit_score,
        min_run=min_run,
        max_period=max_period,
        residues=dict(residues),
    )
