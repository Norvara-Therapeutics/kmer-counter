"""Counting the words of a sequence.

A spectrum is the count of every word of length ``k`` in a sliding window
that advances one residue at a time, so a sequence of ``L`` residues offers
``L - k + 1`` windows and each residue but the ends appears in ``k`` of them.
Overlap is the point: it is what makes the spectrum sensitive to repeats, and
it is also why the counts are not independent observations and should not be
treated as such by a test that assumes they are.

Windows containing a residue the alphabet does not count - an ambiguity code,
a gap, a stop, a residue outside the twenty - are not counted and not guessed
at. They are counted separately instead, because a sequence that is a quarter
N has a spectrum taken over three quarters of its windows and the reader needs
to know that. The sequence is split into maximal runs of countable residues
and each run is counted on its own, which is both how the skipped windows are
found and the fastest way to do it in pure Python.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Mapping, Sequence

from kmer_counter.alphabet import Alphabet, reverse_complement

#: The shortest word that can be counted.
K_MIN: int = 1

#: The longest word that can be counted. Beyond eight, a word of DNA is
#: longer than most sequences have windows, so every count is one and the
#: spectrum says more about the length of the sequence than its composition.
K_MAX: int = 8


def check_k(k: int) -> int:
    """Validate a word length.

    Raises:
        ValueError: if ``k`` is outside the supported range.
    """
    if k < K_MIN or k > K_MAX:
        raise ValueError(f"k must be between {K_MIN} and {K_MAX}, got {k}")
    return k


def window_count(length: int, k: int) -> int:
    """How many windows of length ``k`` a sequence of ``length`` residues has."""
    return max(0, length - k + 1)


def possible_words(alphabet: Alphabet, k: int, canonical: bool = False) -> int:
    """How many distinct words of length ``k`` could exist.

    Under canonical counting a word and its reverse complement are one word,
    which halves the space except for the palindromes that are their own
    reverse complement. Those exist only for even ``k``, since the middle
    residue of an odd word would have to be its own complement.
    """
    check_k(k)
    total = alphabet.words(k)
    if not canonical:
        return total
    if not alphabet.is_nucleotide:
        raise ValueError(f"{alphabet.label} has no complement, so has no canonical form")
    palindromes = alphabet.words(k // 2) if k % 2 == 0 else 0
    return (total + palindromes) // 2


def segments(sequence: str, alphabet: Alphabet) -> Iterator[tuple[int, str]]:
    """Yield ``(start, text)`` for each maximal run of countable residues.

    ``start`` is the 0-based offset of the run in the sequence. Runs are the
    unit of counting: no window crosses a residue the alphabet does not count.
    """
    core = set(alphabet.core)
    start = -1
    for index, residue in enumerate(sequence):
        if residue in core:
            if start < 0:
                start = index
            continue
        if start >= 0:
            yield start, sequence[start:index]
            start = -1
    if start >= 0:
        yield start, sequence[start:]


def canonical_word(word: str, alphabet: Alphabet) -> str:
    """The lexicographically smaller of a word and its reverse complement."""
    other = reverse_complement(word, alphabet)
    return word if word <= other else other


@dataclass(frozen=True)
class Spectrum:
    """The counted words of one sequence, or of several counted together.

    Attributes:
        k: the word length.
        alphabet: the alphabet the words are drawn from.
        counts: word to how many windows hold it.
        windows: how many windows were counted.
        skipped: how many windows were not counted, because they held a
            residue the alphabet does not count.
        length: how many residues the spectrum was taken over.
        sequences: how many sequences contributed.
        canonical: whether a word and its reverse complement were counted as
            one word, under the lexicographically smaller of the two.
    """

    k: int
    alphabet: Alphabet
    counts: Mapping[str, int] = field(default_factory=dict)
    windows: int = 0
    skipped: int = 0
    length: int = 0
    sequences: int = 0
    canonical: bool = False

    @property
    def distinct(self) -> int:
        """How many different words occur."""
        return len(self.counts)

    @property
    def possible(self) -> int:
        """How many different words the alphabet could spell at this length."""
        return possible_words(self.alphabet, self.k, self.canonical)

    @property
    def observable(self) -> int:
        """How many different words could occur in this many windows.

        A sequence shorter than its word space cannot show every word however
        little it repeats itself, so this is the ceiling the observed count
        should be read against: the smaller of the word space and the number
        of windows.
        """
        return min(self.possible, self.windows)

    @property
    def saturation(self) -> float | None:
        """Distinct words as a share of what the windows could have shown.

        One means every window held a word not seen elsewhere in the
        sequence, which is what an unrepetitive sequence looks like. None when
        there are no windows.
        """
        if not self.observable:
            return None
        return self.distinct / self.observable

    @property
    def coverage(self) -> float | None:
        """Distinct words as a share of the whole word space.

        None when there are no windows. For a sequence far shorter than its
        word space this is small however complex the sequence is, so it says
        more about length than about complexity; read it beside saturation.
        """
        if not self.windows:
            return None
        return self.distinct / self.possible

    @property
    def skipped_share(self) -> float:
        """The share of all windows that were not counted."""
        total = self.windows + self.skipped
        return self.skipped / total if total else 0.0

    def frequency(self, word: str) -> float:
        """The share of counted windows holding a word."""
        if not self.windows:
            return 0.0
        return self.counts.get(word, 0) / self.windows

    def most_common(self, number: int = 10) -> list[tuple[str, int]]:
        """The most abundant words, ties broken alphabetically for stability."""
        ordered = sorted(self.counts.items(), key=lambda item: (-item[1], item[0]))
        return ordered[:number]

    def census(self) -> dict[int, int]:
        """How many distinct words occur once, twice, and so on.

        This is the spectrum in the sense the assembly literature uses the
        word: the distribution of abundances rather than of words. A long tail
        at high abundance is repetition.
        """
        tally: Counter[int] = Counter(self.counts.values())
        return dict(sorted(tally.items()))

    def singletons(self) -> int:
        """How many words occur exactly once."""
        return sum(1 for count in self.counts.values() if count == 1)

    def repeated_windows(self) -> int:
        """How many windows hold a word that occurs more than once.

        Every window in this count is part of some repetition, which is a
        blunter but more robust statement than naming the repeats.
        """
        return sum(count for count in self.counts.values() if count > 1)

    def sorted_items(self) -> list[tuple[str, int]]:
        """Every word with its count, most abundant first then alphabetical."""
        return sorted(self.counts.items(), key=lambda item: (-item[1], item[0]))


def count(
    sequence: str,
    k: int,
    alphabet: Alphabet,
    canonical: bool = False,
) -> Spectrum:
    """Count every word of length ``k`` in one sequence.

    Args:
        sequence: residues, uppercased.
        k: word length, between ``K_MIN`` and ``K_MAX``.
        alphabet: the alphabet whose residues are counted.
        canonical: count a word and its reverse complement as one word.

    Raises:
        ValueError: if ``k`` is out of range, or canonical counting is asked
            for on an alphabet without a complement.
    """
    check_k(k)
    if canonical and not alphabet.is_nucleotide:
        raise ValueError(f"{alphabet.label} has no complement, so has no canonical form")

    counts: Counter[str] = Counter()
    counted = 0
    for _, text in segments(sequence, alphabet):
        limit = window_count(len(text), k)
        if limit <= 0:
            continue
        counted += limit
        counts.update(text[index: index + k] for index in range(limit))

    if canonical:
        collapsed: Counter[str] = Counter()
        for word, number in counts.items():
            collapsed[canonical_word(word, alphabet)] += number
        counts = collapsed

    total = window_count(len(sequence), k)
    return Spectrum(
        k=k,
        alphabet=alphabet,
        counts=dict(counts),
        windows=counted,
        skipped=total - counted,
        length=len(sequence),
        sequences=1 if sequence else 0,
        canonical=canonical,
    )


def merge(spectra: Sequence[Spectrum]) -> Spectrum:
    """Combine spectra of the same word length and alphabet into one.

    Windows are summed, which is what counting several records as one sequence
    means: no window spans two records, so the combined spectrum has fewer
    windows than the concatenated sequence would have. That is the right
    answer - a word straddling two records is not a word in either.

    Raises:
        ValueError: if the spectra disagree on k, alphabet or canonical form,
            or if none are given.
    """
    if not spectra:
        raise ValueError("no spectra to merge")
    first = spectra[0]
    for other in spectra[1:]:
        if other.k != first.k:
            raise ValueError(f"cannot merge k={first.k} with k={other.k}")
        if other.alphabet is not first.alphabet:
            raise ValueError(
                f"cannot merge {first.alphabet.label} with {other.alphabet.label}"
            )
        if other.canonical != first.canonical:
            raise ValueError("cannot merge canonical and non-canonical spectra")

    counts: Counter[str] = Counter()
    for spectrum in spectra:
        counts.update(spectrum.counts)
    return Spectrum(
        k=first.k,
        alphabet=first.alphabet,
        counts=dict(counts),
        windows=sum(spectrum.windows for spectrum in spectra),
        skipped=sum(spectrum.skipped for spectrum in spectra),
        length=sum(spectrum.length for spectrum in spectra),
        sequences=sum(spectrum.sequences for spectrum in spectra),
        canonical=first.canonical,
    )


def count_all(
    sequences: Iterable[str],
    k: int,
    alphabet: Alphabet,
    canonical: bool = False,
) -> Spectrum:
    """Count several sequences as one spectrum, holding one at a time."""
    check_k(k)
    if canonical and not alphabet.is_nucleotide:
        raise ValueError(f"{alphabet.label} has no complement, so has no canonical form")
    counts: Counter[str] = Counter()
    windows = skipped = length = number = 0
    for sequence in sequences:
        spectrum = count(sequence, k, alphabet, canonical)
        counts.update(spectrum.counts)
        windows += spectrum.windows
        skipped += spectrum.skipped
        length += spectrum.length
        number += spectrum.sequences
    return Spectrum(
        k=k,
        alphabet=alphabet,
        counts=dict(counts),
        windows=windows,
        skipped=skipped,
        length=length,
        sequences=number,
        canonical=canonical,
    )


def positions(sequence: str, word: str, alphabet: Alphabet, canonical: bool = False) -> list[int]:
    """Every 1-based start at which a word occurs.

    Under canonical counting the reverse complement is matched too, since it
    was counted as the same word.
    """
    wanted = {word}
    if canonical and alphabet.is_nucleotide:
        wanted.add(reverse_complement(word, alphabet))
    k = len(word)
    found = []
    for index in range(window_count(len(sequence), k)):
        if sequence[index: index + k] in wanted:
            found.append(index + 1)
    return found


def residue_census(sequences: Iterable[str]) -> dict[str, int]:
    """Count every residue across sequences, for deciding the alphabet."""
    tally: Counter[str] = Counter()
    for sequence in sequences:
        tally.update(sequence)
    return dict(tally)
