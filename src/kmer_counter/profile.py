"""Complexity along the sequence, rather than over it.

A whole-sequence entropy averages a tandem repeat away: a 4 kb payload with
200 bp of CAG repeat in it has very nearly the entropy of the same payload
without. The repeat is found by scoring a window and sliding it, which is what
this module does, and by then merging the windows that score badly into
intervals, which is what makes the output a list of places to look rather than
a graph to squint at.

Two numbers are kept per window. The score is the alphabet's own
low-complexity measure - the DUST score for nucleotide, Wootton-Federhen
complexity for protein - and is what the interval calling uses. The entropy is
the Shannon entropy of the window's residue composition in bits, which is
directly comparable along the sequence and across sequences of the same
alphabet, and is what the report plots.

The window is scored by rolling counts rather than by recounting, so the whole
profile costs one pass over the sequence whatever the window size, and the
step only changes how many windows are reported.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

from kmer_counter.alphabet import Alphabet
from kmer_counter.complexity import (
    DUST_WINDOW,
    attainable_entropy,
    default_threshold,
    default_window,
    is_low_complexity,
    max_triplet_score,
    score_name,
    shannon_entropy,
)
from kmer_counter.spectrum import window_count


def score_ceiling(alphabet: Alphabet, window: int) -> float:
    """The most extreme score a window of this length could reach.

    Both measures are bounded by the window: the DUST score of a homopolymer
    is ``(window - 2) / 2``, so a 64-residue window cannot exceed 31 and a
    12-residue window cannot exceed 5, while the entropy of a window can be no
    more than log2 of its length.
    """
    if alphabet.is_nucleotide:
        return max_triplet_score(window)
    return attainable_entropy(window, alphabet.size)


def threshold_reachable(alphabet: Alphabet, window: int, threshold: float) -> bool:
    """Whether a window of this length could be called at this threshold.

    The conventional thresholds come with conventional windows - 20 with a
    window of 64 for DUST, 2.2 bits with a window of 12 for protein - and
    moving the window without moving the threshold quietly turns the call into
    a constant. A 12-residue window scores at most 5 on DUST, so a threshold
    of 20 can never be met by one.
    """
    ceiling = score_ceiling(alphabet, window)
    if alphabet.is_nucleotide:
        return threshold <= ceiling
    return threshold < ceiling


def threshold_note(alphabet: Alphabet, window: int, threshold: float) -> str:
    """Why calls at these settings cannot be read plainly, or an empty string.

    Taken from the settings rather than from a scored profile, so the caveat
    reaches every rendering of a run whether or not that rendering kept a
    window profile to show.
    """
    if threshold_reachable(alphabet, window, threshold):
        return ""
    ceiling = score_ceiling(alphabet, window)
    if alphabet.is_nucleotide:
        return (
            f"no window can be called: a {window}-residue window scores at most "
            f"{ceiling:g} on DUST, below the threshold of {threshold:g}; the "
            f"threshold of 20 belongs with a window of {DUST_WINDOW}"
        )
    return (
        f"every window with a countable residue is called: a {window}-residue "
        f"window of {alphabet.label} has at most {ceiling:.2f} bits of residue "
        f"entropy, at or below the threshold of {threshold:g}"
    )


@dataclass(frozen=True)
class Window:
    """One scored window.

    Attributes:
        start: 1-based position of the window's first residue.
        end: 1-based position of its last residue, inclusive.
        score: the alphabet's low-complexity measure over the window.
        entropy: Shannon entropy of the window's residue composition, in bits.
        gc: the window's GC share, None for a protein alphabet.
        countable: how many of the window's residues the alphabet counts.
        low: whether the score calls the window low-complexity.
    """

    start: int
    end: int
    score: float
    entropy: float
    gc: float | None
    countable: int
    low: bool

    @property
    def length(self) -> int:
        """How many residues the window spans."""
        return self.end - self.start + 1

    @property
    def midpoint(self) -> float:
        """The window's middle position, which is where it is plotted."""
        return (self.start + self.end) / 2


@dataclass(frozen=True)
class Interval:
    """A run of low-complexity windows, merged into one region.

    Attributes:
        start: 1-based position of the region's first residue.
        end: 1-based position of its last, inclusive.
        windows: how many windows were called low over the region.
        peak: the most extreme score among them.
        mean: their mean score.
    """

    start: int
    end: int
    windows: int
    peak: float
    mean: float

    @property
    def length(self) -> int:
        """How many residues the region spans."""
        return self.end - self.start + 1


@dataclass(frozen=True)
class Profile:
    """A scored sequence.

    Attributes:
        name: the record the profile was taken over.
        length: the record's length in residues.
        alphabet: the alphabet scored under.
        measure: 'dust' or 'complexity', the measure the scores are in.
        window: the window length in residues.
        step: how far the window advanced between scores.
        threshold: the score at which a window was called low-complexity.
        windows: every scored window, in order.
        intervals: the merged low-complexity regions.
    """

    name: str
    length: int
    alphabet: Alphabet
    measure: str
    window: int
    step: int
    threshold: float
    windows: Sequence[Window]
    intervals: Sequence[Interval]

    @property
    def scored(self) -> int:
        """How many windows were scored."""
        return len(self.windows)

    @property
    def low_windows(self) -> int:
        """How many windows were called low-complexity."""
        return sum(1 for window in self.windows if window.low)

    @property
    def masked(self) -> int:
        """How many residues the low-complexity regions cover."""
        return sum(interval.length for interval in self.intervals)

    @property
    def masked_share(self) -> float:
        """Those residues as a share of the sequence."""
        return self.masked / self.length if self.length else 0.0

    @property
    def scores(self) -> list[float]:
        """Every window score, in order."""
        return [window.score for window in self.windows]

    @property
    def entropies(self) -> list[float]:
        """Every window entropy, in order."""
        return [window.entropy for window in self.windows]

    @property
    def ceiling(self) -> float:
        """The most extreme score a window of this length could reach.

        Both measures are bounded by the window: the DUST score of a
        homopolymer is ``(window - 2) / 2``, so a 64-residue window cannot
        exceed 31 and a 12-residue window cannot exceed 5, while the entropy
        of a window can be no more than log2 of its length.
        """
        return score_ceiling(self.alphabet, self.window)

    @property
    def threshold_reachable(self) -> bool:
        """Whether a window could be called low-complexity at this threshold."""
        return threshold_reachable(self.alphabet, self.window, self.threshold)

    @property
    def qualification(self) -> str:
        """Why the low-complexity calls cannot be read plainly, if they cannot.

        An empty string when they can. This is carried into every rendering,
        including the JSON, because a machine reading the regions cannot see
        a warning printed beside them.
        """
        return threshold_note(self.alphabet, self.window, self.threshold)

    def extreme(self) -> Window | None:
        """The window whose score is most extreme, ties broken by position.

        Most extreme means highest for DUST and lowest for a complexity, so
        the caller does not have to know which measure is in use.
        """
        if not self.windows:
            return None
        if self.alphabet.is_nucleotide:
            return max(self.windows, key=lambda window: (window.score, -window.start))
        return min(self.windows, key=lambda window: (window.score, window.start))


def iter_windows(
    sequence: str,
    alphabet: Alphabet,
    window: int,
    threshold: float,
) -> Iterator[Window]:
    """Score every window in turn, keeping counts rather than recounting.

    Every window is scored, whatever step the caller then samples at, so the
    regions called do not depend on how much of the profile is reported.
    """
    core = set(alphabet.core)
    nucleotide = alphabet.is_nucleotide
    total = window_count(len(sequence), window)
    if total <= 0:
        return

    residues: Counter[str] = Counter()
    triplets: Counter[str] = Counter()
    pairs = 0

    def add_residue(index: int) -> None:
        residue = sequence[index]
        if residue in core:
            residues[residue] += 1

    def drop_residue(index: int) -> None:
        residue = sequence[index]
        if residue in core:
            residues[residue] -= 1
            if not residues[residue]:
                del residues[residue]

    def triplet_at(index: int) -> str | None:
        """The triplet starting at ``index``, or None if it is not countable."""
        if index + 3 > len(sequence):
            return None
        text = sequence[index: index + 3]
        return text if all(residue in core for residue in text) else None

    def add_triplet(index: int) -> None:
        nonlocal pairs
        text = triplet_at(index)
        if text is None:
            return
        pairs += triplets[text]
        triplets[text] += 1

    def drop_triplet(index: int) -> None:
        nonlocal pairs
        text = triplet_at(index)
        if text is None:
            return
        triplets[text] -= 1
        pairs -= triplets[text]
        if not triplets[text]:
            del triplets[text]

    for index in range(window):
        add_residue(index)
    if nucleotide:
        for index in range(max(0, window - 2)):
            add_triplet(index)

    divisor = window - 3
    for start in range(total):
        if start:
            drop_residue(start - 1)
            add_residue(start + window - 1)
            if nucleotide:
                drop_triplet(start - 1)
                add_triplet(start + window - 3)
        countable = sum(residues.values())
        entropy = shannon_entropy(residues.values())
        if nucleotide:
            score = pairs / divisor if divisor > 0 else 0.0
            gc: float | None = (
                (residues.get("G", 0) + residues.get("C", 0)) / countable
                if countable
                else None
            )
        else:
            score = entropy
            gc = None
        yield Window(
            start=start + 1,
            end=start + window,
            score=score,
            entropy=entropy,
            gc=gc,
            countable=countable,
            # A window holding nothing the alphabet counts is not called. Its
            # entropy is zero because there is nothing in it, not because what
            # is in it repeats, and masking a region on that basis would report
            # a run of N or of X as low-complexity sequence. The DUST score of
            # such a window is zero and so never called anyway; this is what
            # makes the two alphabets agree about it.
            low=countable > 0 and is_low_complexity(score, alphabet, threshold),
        )


class _Merger:
    """Accumulates low-complexity windows into regions as they arrive.

    Windows are merged as they come rather than collected first, so a
    sequence that is low-complexity from end to end costs one region in
    memory rather than one object per window.
    """

    def __init__(self, nucleotide: bool) -> None:
        self._nucleotide = nucleotide
        self._open = False
        self._start = 0
        self._end = 0
        self._count = 0
        self._total = 0.0
        self._peak = 0.0
        self.intervals: list[Interval] = []

    def add(self, window: Window) -> None:
        """Take one low-complexity window."""
        if self._open and window.start <= self._end + 1:
            self._end = max(self._end, window.end)
        else:
            self.close()
            self._open = True
            self._start = window.start
            self._end = window.end
            self._count = 0
            self._total = 0.0
            self._peak = window.score
        self._count += 1
        self._total += window.score
        self._peak = (
            max(self._peak, window.score)
            if self._nucleotide
            else min(self._peak, window.score)
        )

    def close(self) -> None:
        """Finish the region being built, if there is one."""
        if not self._open:
            return
        self.intervals.append(
            Interval(
                start=self._start,
                end=self._end,
                windows=self._count,
                peak=self._peak,
                mean=self._total / self._count if self._count else 0.0,
            )
        )
        self._open = False


def merge_low(windows: Iterable[Window], nucleotide: bool = True) -> list[Interval]:
    """Merge overlapping and adjacent low-complexity windows into regions.

    Two windows that overlap, or that abut with no gap, belong to one region;
    a gap of even one residue between them makes two, because a window that
    scored well in between is evidence the sequence recovered there. Windows
    that are not low are ignored, so a whole profile can be passed in.

    Args:
        windows: the windows, in position order.
        nucleotide: whether a high score is the extreme one, as it is for
            DUST. For a protein complexity the peak is the lowest score.
    """
    merger = _Merger(nucleotide)
    for window in windows:
        if window.low:
            merger.add(window)
    merger.close()
    return merger.intervals


def scan(
    sequence: str,
    alphabet: Alphabet,
    name: str = "sequence",
    window: int | None = None,
    step: int = 1,
    threshold: float | None = None,
) -> Profile:
    """Score a sequence window by window and merge its low-complexity regions.

    Args:
        sequence: residues, uppercased.
        alphabet: the alphabet to score under.
        name: the record name, carried into the output.
        window: window length; the alphabet's conventional one by default,
            64 residues for nucleotide and 12 for protein.
        step: how far to advance between reported windows. Every window is
            still scored, so the step changes the size of the output and not
            the regions called.
        threshold: the score at which a window is called low-complexity; the
            alphabet's conventional one by default, 20 for DUST and 2.2 bits
            for Wootton-Federhen complexity.

    Raises:
        ValueError: if the window or step is less than one.
    """
    size = default_window(alphabet) if window is None else window
    limit = default_threshold(alphabet) if threshold is None else threshold
    if size < 1:
        raise ValueError(f"window must be 1 or greater, got {size}")
    if step < 1:
        raise ValueError(f"step must be 1 or greater, got {step}")

    merger = _Merger(alphabet.is_nucleotide)
    windows: list[Window] = []
    for index, scored in enumerate(iter_windows(sequence, alphabet, size, limit)):
        if scored.low:
            merger.add(scored)
        if index % step == 0:
            windows.append(scored)
    merger.close()
    return Profile(
        name=name,
        length=len(sequence),
        alphabet=alphabet,
        measure=score_name(alphabet),
        window=size,
        step=step,
        threshold=limit,
        windows=windows,
        intervals=merger.intervals,
    )
