"""Measures of how much a sequence repeats itself.

Four measures are kept, because they answer different questions and disagree
in informative ways:

* Shannon entropy of the word counts, in bits, with the ceiling the sequence
  could reach at that word length. It asks how evenly the windows are spread
  over the words that occur.
* Linguistic complexity, the product over word lengths of the words seen
  against the words that could have been seen. It asks how much of the
  vocabulary a sequence uses, and unlike entropy it is sensitive to words that
  are absent rather than merely rare.
* The DUST score, the measure BLAST masks nucleotide sequence with. It is a
  count of repeated triplets within a window, so it responds to the local
  tandem repetition that entropy over a whole sequence averages away.
* Wootton-Federhen complexity, the measure SEG masks protein with: the entropy
  of the residue composition of a window.

All logarithms are base two, so entropies are in bits and a protein residue
carries at most log2(20) = 4.32 of them against a nucleotide's 2.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import lgamma, log, log2
from typing import Iterable, Mapping, Sequence

from kmer_counter.alphabet import Alphabet
from kmer_counter.spectrum import Spectrum

#: Window in which the DUST score is conventionally taken, from the defaults
#: of the BLAST implementation.
DUST_WINDOW: int = 64

#: DUST score above which a window is conventionally called low-complexity.
#: In a 64-residue window the score runs from 0 to 31, so 20 is a sequence
#: whose triplets repeat about two thirds as much as a homopolymer's.
DUST_THRESHOLD: float = 20.0

#: Window in which Wootton-Federhen complexity is conventionally taken, from
#: the defaults of SEG.
SEG_WINDOW: int = 12

#: Wootton-Federhen complexity in bits below which a protein window is
#: conventionally called low-complexity, from the SEG trigger.
SEG_THRESHOLD: float = 2.2


def shannon_entropy(counts: Iterable[int]) -> float:
    """Shannon entropy of a set of counts, in bits.

    Zero for a single word, however abundant, and log2(n) for n words of
    equal abundance. Counts of zero contribute nothing and are skipped.
    """
    numbers = [count for count in counts if count > 0]
    total = sum(numbers)
    if total <= 0:
        return 0.0
    entropy = -sum((count / total) * log2(count / total) for count in numbers)
    # A single word gives -0.0, which prints as '-0.00' and reads as a defect.
    return entropy if entropy else 0.0


def max_entropy(words: int) -> float:
    """The entropy that many words would reach if equally abundant, in bits."""
    return log2(words) if words > 0 else 0.0


def attainable_entropy(windows: int, space: int) -> float:
    """The highest entropy ``windows`` windows could reach over ``space`` words.

    Not ``log2(min(space, windows))``: windows are whole, and when there are
    more windows than words they cannot be divided evenly unless the one
    number happens to divide the other. Five windows over four words are at
    their most even as 2, 1, 1, 1, which carries 1.92 bits and not the 2.00
    that four equal shares would. Reporting the round number as the ceiling
    makes a sequence that is as even as it could possibly be look 96%
    efficient.

    The even integer allocation is the maximum because entropy is
    Schur-concave: any transfer away from evenness lowers it.

    This remains an upper bound rather than a reachable value for a real
    sequence, because overlapping windows constrain which allocations a
    sequence can realise at all - the last k-1 residues of one window are the
    first k-1 of the next. It is the bound that does not depend on the
    sequence, which is what a ceiling should be.
    """
    if windows <= 0 or space <= 0:
        return 0.0
    if windows <= space:
        # Every window can hold a word of its own.
        return max_entropy(windows)
    quotient, remainder = divmod(windows, space)
    taller = remainder
    shorter = space - remainder
    entropy = 0.0
    if taller:
        share = (quotient + 1) / windows
        entropy -= taller * share * log2(share)
    if shorter:
        share = quotient / windows
        entropy -= shorter * share * log2(share)
    return entropy


@dataclass(frozen=True)
class Entropy:
    """The entropy of one spectrum, with what it is being read against.

    Attributes:
        k: the word length.
        bits: Shannon entropy of the word counts.
        ceiling: the highest entropy this many windows could reach over this
            word space, allocated as evenly as whole windows allow. A
            sequence with fewer windows than words cannot reach the entropy
            of the word space however complex it is, and one with more
            windows than words cannot divide them evenly unless the numbers
            divide.
        distinct: how many words occur.
        observable: how many could have occurred.
        windows: how many windows were counted.
        space: how many words the alphabet can spell at this length.
    """

    k: int
    bits: float
    ceiling: float
    distinct: int
    observable: int
    windows: int
    space: int

    @property
    def efficiency(self) -> float | None:
        """Entropy as a share of the ceiling, or None when there is none.

        One is a sequence using every word available to it equally often.
        """
        if self.ceiling <= 0:
            return None
        return self.bits / self.ceiling

    @property
    def length_limited(self) -> bool:
        """Whether the ceiling is set by the sequence's length, not the alphabet.

        When it is, the entropy cannot be compared with that of a longer
        sequence at the same word length: the shorter one is being graded
        against an easier ceiling.
        """
        return self.observable < self.space


def entropy_of(spectrum: Spectrum) -> Entropy:
    """Measure the entropy of a spectrum against its own ceiling."""
    return Entropy(
        k=spectrum.k,
        bits=shannon_entropy(spectrum.counts.values()),
        ceiling=attainable_entropy(spectrum.windows, spectrum.possible),
        distinct=spectrum.distinct,
        observable=spectrum.observable,
        windows=spectrum.windows,
        space=spectrum.possible,
    )


def linguistic_complexity(spectra: Sequence[Spectrum]) -> float | None:
    """The product over word lengths of words seen against words observable.

    This is the measure of Trifonov (1990): a sequence scores one when every
    word of every length that could have appeared did, and falls as the
    vocabulary narrows. Taking the product over lengths is what distinguishes
    it from a single-length count, since a sequence can use every residue
    while using only a few of the dinucleotides.

    Word lengths must be consecutive and start at one; a length whose windows
    are all skipped as ambiguous is left out of the product rather than
    counted as zero.

    Returns:
        The complexity, or None when no word length had a countable window.

    Raises:
        ValueError: if the word lengths are not consecutive from one.
    """
    lengths = [spectrum.k for spectrum in spectra]
    if lengths != list(range(1, len(lengths) + 1)):
        raise ValueError(
            "linguistic complexity needs consecutive word lengths from 1, got "
            f"{lengths}"
        )
    product = 1.0
    used = 0
    for spectrum in spectra:
        if not spectrum.observable:
            continue
        product *= spectrum.distinct / spectrum.observable
        used += 1
    return product if used else None


def triplet_score(text: str) -> float:
    """The DUST score of one window.

    Counts how often each triplet repeats within the window, as the number of
    pairs of windows holding the same triplet, and divides by the number of
    triplet positions less one. A window of 64 residues scores 0 when no
    triplet occurs twice and 31 when it is a homopolymer.

    After Morgulis, Gertz, Schaffer and Agarwala (2006). The score is taken
    here on a fixed sliding window rather than by their search for perfect
    intervals, so a masking decision made from it is close to, but not
    identical with, the one their symmetric implementation makes.
    """
    if len(text) < 4:
        return 0.0
    triplets: Counter[str] = Counter(
        text[index: index + 3] for index in range(len(text) - 2)
    )
    pairs = sum(count * (count - 1) // 2 for count in triplets.values())
    return pairs / (len(text) - 3)


def max_triplet_score(length: int) -> float:
    """The DUST score a homopolymer of this length would reach."""
    if length < 4:
        return 0.0
    return (length - 2) / 2


def residue_entropy(text: str, alphabet: Alphabet | None = None) -> float:
    """Wootton-Federhen complexity: the entropy of a window's composition, in bits.

    This is the K2 of Wootton and Federhen (1993), which SEG triggers on at
    2.2 bits in a window of twelve residues. Every character present is
    counted, so a window of gaps or ambiguity codes is described as it is
    rather than as empty; pass an alphabet to count only its residues.
    """
    if alphabet is None:
        residues = text
    else:
        core = set(alphabet.core)
        residues = "".join(residue for residue in text if residue in core)
    return shannon_entropy(Counter(residues).values())


def compositional_complexity(text: str, alphabet: Alphabet) -> float:
    """Wootton-Federhen K1: the log of a window's composition multiplicity.

    How many distinct sequences share this window's residue composition,
    logged to the size of the alphabet and divided by the window length, so
    the measure runs from 0 for a homopolymer to 1 for a window using every
    residue equally. Computed through log-gamma, so a long window does not
    overflow.
    """
    core = set(alphabet.core)
    residues = [residue for residue in text if residue in core]
    length = len(residues)
    if length == 0 or alphabet.size < 2:
        return 0.0
    counts = Counter(residues)
    log_multiplicity = lgamma(length + 1) - sum(
        lgamma(count + 1) for count in counts.values()
    )
    return log_multiplicity / (length * log(alphabet.size))


def default_window(alphabet: Alphabet) -> int:
    """The conventional window for the alphabet's low-complexity measure."""
    return DUST_WINDOW if alphabet.is_nucleotide else SEG_WINDOW


def default_threshold(alphabet: Alphabet) -> float:
    """The conventional low-complexity threshold for the alphabet."""
    return DUST_THRESHOLD if alphabet.is_nucleotide else SEG_THRESHOLD


def is_low_complexity(score: float, alphabet: Alphabet, threshold: float | None = None) -> bool:
    """Whether a window's score calls it low-complexity.

    The two measures run in opposite directions: a high DUST score is
    repetitive, a low Wootton-Federhen complexity is. This is the one place
    that difference is resolved, so no caller has to remember which.
    """
    limit = default_threshold(alphabet) if threshold is None else threshold
    if alphabet.is_nucleotide:
        return score >= limit
    return score <= limit


def summarise(
    spectra: Mapping[int, Spectrum],
) -> dict[int, Entropy]:
    """Measure the entropy of each spectrum, keyed by word length."""
    return {k: entropy_of(spectrum) for k, spectrum in sorted(spectra.items())}


def expected_distinct(windows: int, space: int) -> float:
    """How many distinct words a random sequence of this length would show.

    The expectation under drawing ``windows`` words independently and
    uniformly from a space of ``space`` words. Overlapping windows are not
    independent, so this is an idealisation and sits above what a real random
    sequence shows at word lengths short enough for the overlap to bind; it
    is reported as the yardstick it is, not as a null hypothesis.
    """
    if space <= 0 or windows <= 0:
        return 0.0
    return space * (1.0 - (1.0 - 1.0 / space) ** windows)


def saturation_ratio(spectrum: Spectrum) -> float | None:
    """Distinct words against what a random sequence of this length would show.

    One means as varied as chance would make it; below one is repetition, and
    above one is not reachable by much, since the expectation is already close
    to the ceiling.
    """
    expected = expected_distinct(spectrum.windows, spectrum.possible)
    if expected <= 0:
        return None
    return spectrum.distinct / expected


def window_score(text: str, alphabet: Alphabet) -> float:
    """The low-complexity score of one window, in the alphabet's own measure.

    The DUST score for a nucleotide alphabet, Wootton-Federhen complexity in
    bits for a protein one. Which measure is in use has to travel with the
    number, since the two run in opposite directions.
    """
    if alphabet.is_nucleotide:
        return triplet_score(text)
    return residue_entropy(text, alphabet)


def score_name(alphabet: Alphabet) -> str:
    """What the alphabet's low-complexity score is called in output."""
    return "dust" if alphabet.is_nucleotide else "complexity"
