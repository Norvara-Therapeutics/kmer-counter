"""Naming the repeats a spectrum only implies.

A spectrum says a sequence repeats itself; it does not say where or as what.
This module answers both, exactly rather than statistically: a run of one
residue, a tandem repeat of a short unit, and the words that occur far more
often than any other are each found by looking, not by modelling.

Only perfect repeats are reported. A tandem repeat with a substitution in it
is two repeats here, and a region that is 95% periodic is not reported at all.
That is a real limitation, and the honest alternative - an approximate matcher
with a scoring scheme and a p-value - is a different tool. What this reports,
it reports exactly: every position can be checked against the sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from kmer_counter.alphabet import Alphabet
from kmer_counter.spectrum import Spectrum, positions

#: Runs of one residue at least this long are reported by default. Five is
#: where homopolymers start to matter to synthesis and to sequencing.
MIN_RUN: int = 5

#: Tandem units up to this length are searched for by default.
MAX_PERIOD: int = 6

#: A tandem repeat is reported once the unit occurs this many times over.
MIN_COPIES: int = 3


@dataclass(frozen=True)
class Run:
    """A run of a single repeated residue.

    Attributes:
        residue: the residue repeated.
        start: 1-based position of the run's first residue.
        end: 1-based position of its last, inclusive.
    """

    residue: str
    start: int
    end: int

    @property
    def length(self) -> int:
        """How many residues the run spans."""
        return self.end - self.start + 1


@dataclass(frozen=True)
class Tandem:
    """A perfect tandem repeat of a short unit.

    Attributes:
        unit: the repeated unit, as it appears at the start of the region.
        start: 1-based position of the region's first residue.
        end: 1-based position of its last, inclusive.
    """

    unit: str
    start: int
    end: int

    @property
    def period(self) -> int:
        """The unit's length."""
        return len(self.unit)

    @property
    def length(self) -> int:
        """How many residues the region spans."""
        return self.end - self.start + 1

    @property
    def copies(self) -> float:
        """How many times the unit fits in the region, part copies included.

        A region ending part-way through a unit is the usual case - CAGCAGCA
        is 2.67 copies of CAG - and rounding it down would lose the two
        residues that make it a repeat rather than a coincidence.
        """
        return self.length / self.period


@dataclass(frozen=True)
class Occurrence:
    """An abundant word and where it occurs.

    Attributes:
        word: the word.
        count: how many windows hold it.
        frequency: that count as a share of the counted windows.
        positions: 1-based starts, in order.
        truncated: whether the positions were cut short of the count.
    """

    word: str
    count: int
    frequency: float
    positions: Sequence[int]
    truncated: bool

    @property
    def spread(self) -> int | None:
        """How many residues separate the first occurrence from the last."""
        if len(self.positions) < 2:
            return None
        return self.positions[-1] - self.positions[0]


def homopolymer_runs(
    sequence: str,
    alphabet: Alphabet | None = None,
    minimum: int = MIN_RUN,
) -> list[Run]:
    """Every run of one repeated residue at least ``minimum`` long.

    Args:
        sequence: residues, uppercased.
        alphabet: when given, only runs of a counted residue are reported, so
            a run of N or of gaps is not mistaken for a homopolymer.
        minimum: the shortest run to report.

    Raises:
        ValueError: if ``minimum`` is less than two.
    """
    if minimum < 2:
        raise ValueError(f"a run is two residues or more, got {minimum}")
    core = set(alphabet.core) if alphabet is not None else None
    runs: list[Run] = []
    start = 0
    for index in range(1, len(sequence) + 1):
        if index < len(sequence) and sequence[index] == sequence[start]:
            continue
        length = index - start
        residue = sequence[start]
        if length >= minimum and (core is None or residue in core):
            runs.append(Run(residue=residue, start=start + 1, end=index))
        start = index
    return runs


def _is_primitive(unit: str) -> bool:
    """Whether a unit is not itself a repetition of a shorter unit.

    Without this, AAAAAA is reported as a repeat of A, of AA and of AAA, and
    CAGCAGCAGCAG as a repeat of CAG and of CAGCAG.
    """
    length = len(unit)
    for period in range(1, length):
        if length % period == 0 and unit == unit[:period] * (length // period):
            return False
    return True


def tandem_repeats(
    sequence: str,
    alphabet: Alphabet | None = None,
    max_period: int = MAX_PERIOD,
    min_copies: float = MIN_COPIES,
) -> list[Tandem]:
    """Every perfect tandem repeat of a unit up to ``max_period`` residues.

    A region is periodic at ``p`` where every residue equals the one ``p``
    before it. Maximal such regions are found for each period in turn, only
    primitive units are kept, and a region contained in one of shorter period
    is dropped, so a poly-A tract is a repeat of A and not also of AA.

    Args:
        sequence: residues, uppercased.
        alphabet: when given, a region containing a residue the alphabet does
            not count is not reported.
        max_period: the longest unit to search for.
        min_copies: how many copies of the unit a region must hold.

    Raises:
        ValueError: if ``max_period`` is less than one or ``min_copies`` is
            less than two.
    """
    if max_period < 1:
        raise ValueError(f"max_period must be 1 or greater, got {max_period}")
    if min_copies < 2:
        raise ValueError(f"a tandem repeat is two copies or more, got {min_copies}")

    core = set(alphabet.core) if alphabet is not None else None
    found: list[Tandem] = []
    for period in range(1, max_period + 1):
        index = 0
        limit = len(sequence) - period
        while index < limit:
            if sequence[index] != sequence[index + period]:
                index += 1
                continue
            run = index
            while run < limit and sequence[run] == sequence[run + period]:
                run += 1
            start, end = index, run + period
            unit = sequence[start: start + period]
            region = sequence[start:end]
            if (
                _is_primitive(unit)
                and (end - start) / period >= min_copies
                and (core is None or all(residue in core for residue in region))
            ):
                found.append(Tandem(unit=unit, start=start + 1, end=end))
            index = run + 1
    return _drop_contained(found)


def _drop_contained(repeats: list[Tandem]) -> list[Tandem]:
    """Drop regions contained in another of shorter period, then order them."""
    ordered = sorted(repeats, key=lambda repeat: (repeat.period, repeat.start))
    kept: list[Tandem] = []
    for repeat in ordered:
        contained = any(
            keeper.start <= repeat.start and repeat.end <= keeper.end
            for keeper in kept
        )
        if not contained:
            kept.append(repeat)
    return sorted(kept, key=lambda repeat: (repeat.start, repeat.period))


def abundant_words(
    spectrum: Spectrum,
    sequence: str,
    number: int = 10,
    limit: int = 100,
) -> list[Occurrence]:
    """The most abundant words, with where each one occurs.

    Args:
        spectrum: the counted words.
        sequence: the sequence they were counted in.
        number: how many words to report.
        limit: the most positions to list for one word; beyond that the list
            is cut and said to be.
    """
    occurrences = []
    for word, count in spectrum.most_common(number):
        where = positions(sequence, word, spectrum.alphabet, spectrum.canonical)
        occurrences.append(
            Occurrence(
                word=word,
                count=count,
                frequency=spectrum.frequency(word),
                positions=where[:limit],
                truncated=len(where) > limit,
            )
        )
    return occurrences


def forced_repeated_windows(windows: int, possible: int) -> int:
    """How many windows must belong to a repeated word whatever the sequence.

    With more windows than the alphabet has words of that length, some word
    has to occur twice: the most singletons ``possible`` words can supply is
    ``possible - 1``, with everything left over piled on one word, so at least
    ``windows - possible + 1`` windows are in a repeated word however
    unrepetitive the sequence is. With no more windows than words, nothing is
    forced - every window can hold a word of its own - so the floor is zero
    at equality and not one.

    This is the floor the observed share should be read against. It is not
    'almost every word must recur': 4,674 windows over 4,096 six-mers force
    only 579 windows, an eighth of them, so an observed three quarters is
    still mostly a fact about the sequence.
    """
    if windows <= 0 or possible <= 0 or windows <= possible:
        return 0
    return windows - possible + 1


def forced_repeat_share(spectrum: Spectrum) -> float:
    """That floor as a share of the counted windows."""
    if not spectrum.windows:
        return 0.0
    return forced_repeated_windows(spectrum.windows, spectrum.possible) / spectrum.windows


def repeat_share(spectrum: Spectrum) -> float:
    """The share of windows holding a word that occurs more than once.

    A blunt but robust statement of how much of a sequence is involved in
    some repetition at this word length, needing no threshold and no model.
    """
    if not spectrum.windows:
        return 0.0
    return spectrum.repeated_windows() / spectrum.windows
