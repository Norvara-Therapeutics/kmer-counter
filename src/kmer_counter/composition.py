"""Word composition, and whether a word occurs more often than it should.

A count on its own says little: GC-rich sequence has more GGC than ATA for no
reason beyond its base composition, and a tool that reported the raw counts
would invite that mistake. Every table here therefore carries an expected
count beside the observed one, under one of two models:

* zero-order, in which residues are drawn independently with the frequencies
  of this sequence. The ratio of observed to expected under this model is the
  odds ratio of Karlin and Burge (1995), the one in which CpG depletion in
  vertebrate DNA is conventionally expressed.
* maximal-order Markov, in which a word's expectation comes from its own
  prefix and suffix: ``E(w) = n(prefix) * n(suffix) / n(middle)``. This is the
  model that asks whether a word is more frequent than its two halves already
  imply, so it is the one that isolates a genuinely over-used word from a word
  built out of over-used shorter ones.

The z-score reported beside the ratio is the Poisson approximation,
``(observed - expected) / sqrt(expected)``. It is indicative, not a test:
overlapping windows are not independent observations, and the expectation is
fitted from the same sequence, so both effects make the score too large in
absolute value. A word is worth looking at when the ratio is far from one and
the expected count is not tiny.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Mapping, Sequence

from kmer_counter.alphabet import Alphabet
from kmer_counter.spectrum import Spectrum

#: The expectation models a table can be built under.
MODELS: tuple[str, ...] = ("zero-order", "markov")

#: Expected counts below this are too small for a ratio or a z-score to mean
#: anything, so neither is reported for them.
MIN_EXPECTED: float = 1.0


@dataclass(frozen=True)
class Word:
    """One word, with how often it occurs and how often it should.

    Attributes:
        word: the word itself.
        count: how many windows hold it.
        frequency: that count as a share of the counted windows.
        expected: how many windows would hold it under the model.
        model: the model the expectation came from.
    """

    word: str
    count: int
    frequency: float
    expected: float
    model: str

    @property
    def ratio(self) -> float | None:
        """Observed over expected, or None when the expectation is too small."""
        if self.expected < MIN_EXPECTED:
            return None
        return self.count / self.expected

    @property
    def z_score(self) -> float | None:
        """The Poisson z-score, or None when the expectation is too small."""
        if self.expected < MIN_EXPECTED:
            return None
        return (self.count - self.expected) / sqrt(self.expected)

    @property
    def direction(self) -> str:
        """'over', 'under', or 'as expected' at a tenth either way."""
        ratio = self.ratio
        if ratio is None:
            return "not assessed"
        if ratio >= 1.1:
            return "over"
        if ratio <= 0.9:
            return "under"
        return "as expected"


@dataclass(frozen=True)
class Table:
    """Every word of one length, with expectations under one model.

    Attributes:
        k: the word length.
        model: the expectation model.
        words: one row per word of the alphabet, in alphabetical order, so a
            word that never occurs is present with a count of zero.
        windows: how many windows the counts were taken over.
        alphabet: the alphabet the words are drawn from.
    """

    k: int
    model: str
    words: Sequence[Word]
    windows: int
    alphabet: Alphabet

    @property
    def observed(self) -> list[Word]:
        """Only the words that occur."""
        return [word for word in self.words if word.count]

    def by_ratio(self, reverse: bool = True) -> list[Word]:
        """Words with an assessable ratio, most over-represented first."""
        assessable = [word for word in self.words if word.ratio is not None]
        return sorted(
            assessable,
            key=lambda word: (word.ratio or 0.0, word.word),
            reverse=reverse,
        )

    def extremes(self, number: int = 5) -> tuple[list[Word], list[Word]]:
        """The most over- and the most under-represented words."""
        ordered = self.by_ratio()
        return ordered[:number], list(reversed(ordered[-number:]))

    def get(self, word: str) -> Word | None:
        """One word's row, or None when the word is not of this length."""
        for row in self.words:
            if row.word == word:
                return row
        return None


def _enumerate_words(alphabet: Alphabet, k: int) -> list[str]:
    """Every word of length ``k``, in alphabetical order."""
    words = [""]
    for _ in range(k):
        words = [word + residue for word in words for residue in sorted(alphabet.core)]
    return words


def shares(spectrum: Spectrum) -> dict[str, float]:
    """Each word's share of the counted windows, for every word that occurs."""
    if not spectrum.windows:
        return {}
    return {
        word: count / spectrum.windows for word, count in sorted(spectrum.counts.items())
    }


def expected_zero_order(word: str, residues: Spectrum, windows: int) -> float:
    """Expected count of a word from residue frequencies alone.

    Args:
        word: the word.
        residues: the ``k = 1`` spectrum, which supplies the frequencies.
        windows: how many windows of the word's length were counted.
    """
    if not residues.windows or not windows:
        return 0.0
    product = 1.0
    for residue in word:
        product *= residues.counts.get(residue, 0) / residues.windows
    return product * windows


def expected_markov(word: str, spectra: Mapping[int, Spectrum]) -> float:
    """Expected count of a word from its own prefix and suffix.

    ``E(w) = n(w without its last residue) * n(w without its first) /
    n(w without either)``, the maximal-order Markov expectation. For a word of
    two residues the middle term is the empty word, whose count is the number
    of single-residue windows, which makes this the Karlin odds-ratio
    expectation.

    Raises:
        KeyError: if the spectra needed for the word's length are missing.
    """
    k = len(word)
    if k < 2:
        raise ValueError("the Markov expectation needs a word of two or more residues")
    longer = spectra[k - 1]
    prefix = longer.counts.get(word[:-1], 0)
    suffix = longer.counts.get(word[1:], 0)
    if k == 2:
        middle = spectra[1].windows
    else:
        middle = spectra[k - 2].counts.get(word[1:-1], 0)
    if not middle:
        return 0.0
    return prefix * suffix / middle


def table(
    spectra: Mapping[int, Spectrum],
    k: int,
    model: str = "zero-order",
    observed_only: bool = False,
) -> Table:
    """Build the table of every word of length ``k``.

    Args:
        spectra: spectra by word length; ``k`` itself and the lengths the
            model needs must be present. The zero-order model needs length
            one, the Markov model lengths ``k - 1`` and ``k - 2``.
        k: the word length to tabulate.
        model: 'zero-order' or 'markov'.
        observed_only: list only the words that occur. The full enumeration
            is what makes an absent word visible, which is the point of the
            table at two and three residues; at eight residues of protein it
            is 25 billion rows, so a long word is tabulated from what occurred.

    Raises:
        ValueError: if the model is unknown, the spectra are canonical, or a
            spectrum the model needs is missing.
    """
    if model not in MODELS:
        raise ValueError(f"unknown model: {model!r}; choose from {', '.join(MODELS)}")
    if k not in spectra:
        raise ValueError(f"no spectrum for k={k}")
    spectrum = spectra[k]
    if spectrum.canonical:
        raise ValueError(
            "composition tables cannot be built from canonical counts, which "
            "merge a word with its reverse complement and so have no residue "
            "frequencies to expect from"
        )
    if model == "markov" and k < 2:
        raise ValueError("the markov model needs a word of two or more residues")
    needed = [1] if model == "zero-order" else [k - 1] + ([k - 2] if k > 2 else [1])
    for length in needed:
        if length not in spectra:
            raise ValueError(f"the {model} model at k={k} needs the k={length} spectrum")

    listed = (
        sorted(spectrum.counts)
        if observed_only
        else _enumerate_words(spectrum.alphabet, k)
    )
    rows = []
    for word in listed:
        count = spectrum.counts.get(word, 0)
        expected = (
            expected_zero_order(word, spectra[1], spectrum.windows)
            if model == "zero-order"
            else expected_markov(word, spectra)
        )
        rows.append(
            Word(
                word=word,
                count=count,
                frequency=count / spectrum.windows if spectrum.windows else 0.0,
                expected=expected,
                model=model,
            )
        )
    return Table(
        k=k,
        model=model,
        words=rows,
        windows=spectrum.windows,
        alphabet=spectrum.alphabet,
    )


def gc_content(residues: Spectrum) -> float | None:
    """The share of counted residues that are G or C.

    None for a protein alphabet, or when nothing was counted.
    """
    if not residues.alphabet.is_nucleotide or not residues.windows:
        return None
    gc = residues.counts.get("G", 0) + residues.counts.get("C", 0)
    return gc / residues.windows


def dinucleotide_ratio(pair: str, spectra: Mapping[int, Spectrum]) -> float | None:
    """The Karlin odds ratio of one dinucleotide.

    ``f(XY) / (f(X) f(Y))``, the measure in which CpG depletion is reported:
    vertebrate genomes run near 0.2 for CG, against the 1.0 of a sequence
    whose dinucleotides follow its base composition.
    """
    if len(pair) != 2:
        raise ValueError(f"expected a dinucleotide, got {pair!r}")
    if 2 not in spectra or 1 not in spectra:
        raise ValueError("a dinucleotide ratio needs the k=1 and k=2 spectra")
    expected = expected_zero_order(pair, spectra[1], spectra[2].windows)
    if expected < MIN_EXPECTED:
        return None
    return spectra[2].counts.get(pair, 0) / expected


def signature(spectra: Mapping[int, Spectrum]) -> dict[str, float]:
    """Every dinucleotide odds ratio: the genome signature of Karlin and Burge.

    Assessable ratios only, so a dinucleotide too rare to judge is absent
    rather than reported as zero.
    """
    rows = table(spectra, 2, "zero-order")
    return {
        word.word: word.ratio
        for word in rows.words
        if word.ratio is not None
    }
