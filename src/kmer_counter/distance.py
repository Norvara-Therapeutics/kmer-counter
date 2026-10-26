"""Distances between spectra.

Comparing sequences by their word frequencies is alignment-free: it does not
ask what corresponds to what, only whether the same words are used in the same
proportions. That makes it robust to rearrangement and to indels, and blind to
anything positional - two sequences built from the same words in a different
order are identical here. A small distance is therefore evidence of similar
composition, not of homology.

Four distances are offered because they weigh the same disagreement
differently:

* cosine, which compares the direction of the frequency vectors and ignores
  how concentrated they are;
* Jensen-Shannon, in bits, which is bounded by one bit and is the information
  one spectrum gives about the other. Its square root is a metric, so that is
  what is reported when a metric is needed;
* Bray-Curtis, the share of frequency that is not shared, which is the
  ecologist's measure of two communities;
* the Karlin signature difference, the mean absolute difference between two
  sequences' dinucleotide odds ratios, which is the measure in which genomic
  signatures are conventionally compared and is defined only at k = 2.

Spectra must agree on word length, alphabet and canonical form; comparing
counts taken under different rules is the one mistake these measures cannot
survive, so it is refused rather than warned about.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log2, sqrt
from typing import Mapping, Sequence

from kmer_counter.composition import signature
from kmer_counter.spectrum import Spectrum

#: The distances that can be asked for by name.
MEASURES: tuple[str, ...] = ("cosine", "jensen-shannon", "bray-curtis")


def _check_pair(left: Spectrum, right: Spectrum) -> None:
    if left.k != right.k:
        raise ValueError(f"cannot compare k={left.k} with k={right.k}")
    if left.alphabet is not right.alphabet:
        raise ValueError(
            f"cannot compare {left.alphabet.label} with {right.alphabet.label}"
        )
    if left.canonical != right.canonical:
        raise ValueError("cannot compare canonical with non-canonical counts")


def _frequencies(spectrum: Spectrum) -> dict[str, float]:
    if not spectrum.windows:
        return {}
    return {
        word: count / spectrum.windows for word, count in spectrum.counts.items()
    }


def cosine_distance(left: Spectrum, right: Spectrum) -> float | None:
    """One minus the cosine of the angle between two frequency vectors.

    Zero for spectra in the same proportions, one for spectra sharing no word
    at all. None when either spectrum has no counted window.
    """
    _check_pair(left, right)
    first, second = _frequencies(left), _frequencies(right)
    if not first or not second:
        return None
    shared = set(first) & set(second)
    dot = sum(first[word] * second[word] for word in shared)
    norm = sqrt(sum(value * value for value in first.values())) * sqrt(
        sum(value * value for value in second.values())
    )
    if not norm:
        return None
    return 1.0 - dot / norm


def jensen_shannon(left: Spectrum, right: Spectrum) -> float | None:
    """The Jensen-Shannon divergence of two spectra, in bits.

    Zero for identical frequencies and one bit for spectra sharing no word.
    None when either spectrum has no counted window.
    """
    _check_pair(left, right)
    first, second = _frequencies(left), _frequencies(right)
    if not first or not second:
        return None
    divergence = 0.0
    for word in set(first) | set(second):
        p = first.get(word, 0.0)
        q = second.get(word, 0.0)
        m = (p + q) / 2
        if p:
            divergence += p * log2(p / m) / 2
        if q:
            divergence += q * log2(q / m) / 2
    # Rounding can leave a divergence a hair outside [0, 1]; the bound is
    # exact, so it is the arithmetic that is corrected, not the bound.
    return min(1.0, max(0.0, divergence))


def jensen_shannon_distance(left: Spectrum, right: Spectrum) -> float | None:
    """The square root of the Jensen-Shannon divergence, which is a metric."""
    divergence = jensen_shannon(left, right)
    return None if divergence is None else sqrt(divergence)


def bray_curtis(left: Spectrum, right: Spectrum) -> float | None:
    """The share of frequency the two spectra do not share.

    Zero for identical frequencies, one for spectra sharing no word. None
    when either spectrum has no counted window.
    """
    _check_pair(left, right)
    first, second = _frequencies(left), _frequencies(right)
    if not first or not second:
        return None
    difference = sum(
        abs(first.get(word, 0.0) - second.get(word, 0.0))
        for word in set(first) | set(second)
    )
    total = sum(first.values()) + sum(second.values())
    if not total:
        return None
    return difference / total


def signature_difference(
    left: Mapping[int, Spectrum],
    right: Mapping[int, Spectrum],
) -> float | None:
    """The Karlin signature difference between two sequences.

    The mean absolute difference between the two sets of dinucleotide odds
    ratios, over the dinucleotides both sequences can assess, multiplied by a
    thousand as the literature reports it. Around 50 separates sequences from
    the same genome from sequences of different genomes in Karlin's own
    surveys; the scale is only meaningful for sequences of a few kilobases or
    more.

    Returns:
        The difference, or None when fewer than half the dinucleotides can be
        assessed in both sequences.
    """
    first = signature(left)
    second = signature(right)
    shared = sorted(set(first) & set(second))
    space = left[2].alphabet.words(2)
    if not shared or len(shared) * 2 < space:
        return None
    return 1000 * sum(abs(first[word] - second[word]) for word in shared) / len(shared)


def distance(left: Spectrum, right: Spectrum, measure: str) -> float | None:
    """One named distance between two spectra.

    Raises:
        ValueError: if the measure is unknown, or the spectra disagree.
    """
    if measure == "cosine":
        return cosine_distance(left, right)
    if measure == "jensen-shannon":
        return jensen_shannon_distance(left, right)
    if measure == "bray-curtis":
        return bray_curtis(left, right)
    raise ValueError(f"unknown measure: {measure!r}; choose from {', '.join(MEASURES)}")


@dataclass(frozen=True)
class Matrix:
    """A square matrix of distances between named spectra.

    Attributes:
        names: the sequences compared, in the order given.
        measure: the distance used.
        k: the word length compared at.
        values: row by row, with None where a distance could not be taken.
    """

    names: Sequence[str]
    measure: str
    k: int
    values: Sequence[Sequence[float | None]]

    def get(self, row: str, column: str) -> float | None:
        """One distance, by name.

        Raises:
            KeyError: if either name was not compared.
        """
        for name in (row, column):
            if name not in self.names:
                known = ", ".join(self.names)
                raise KeyError(f"{name!r} was not compared; these were: {known}")
        return self.values[self.names.index(row)][self.names.index(column)]

    def pairs(self) -> list[tuple[str, str, float | None]]:
        """Every distinct pair once, in order, with its distance."""
        return [
            (self.names[row], self.names[column], self.values[row][column])
            for row in range(len(self.names))
            for column in range(row + 1, len(self.names))
        ]

    def closest(self) -> tuple[str, str, float] | None:
        """The pair with the smallest distance, or None when none was taken."""
        taken = [pair for pair in self.pairs() if pair[2] is not None]
        if not taken:
            return None
        row, column, value = min(taken, key=lambda pair: (pair[2], pair[0], pair[1]))
        return row, column, float(value)


def matrix(
    spectra: Mapping[str, Spectrum],
    measure: str = "cosine",
) -> Matrix:
    """Every pairwise distance between named spectra.

    Raises:
        ValueError: if the measure is unknown, fewer than two spectra are
            given, or the spectra disagree on word length, alphabet or
            canonical form.
    """
    if measure not in MEASURES:
        raise ValueError(f"unknown measure: {measure!r}; choose from {', '.join(MEASURES)}")
    names = list(spectra)
    if len(names) < 2:
        raise ValueError("comparing needs two or more sequences")
    values: list[list[float | None]] = []
    for row in names:
        values.append(
            [
                0.0 if row == column else distance(spectra[row], spectra[column], measure)
                for column in names
            ]
        )
    return Matrix(names=names, measure=measure, k=spectra[names[0]].k, values=values)
