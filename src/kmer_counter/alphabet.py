"""Residue alphabets, and deciding which one a sequence is written in.

Every count in this package is taken over an alphabet, and the alphabet
decides three things that change the numbers: which residues are countable,
which stand for several residues and so cannot be counted, and how many
distinct words of a given length could exist at all. A protein 4-mer is drawn
from 160,000 possibilities and a DNA 4-mer from 256, so the same entropy in
bits means different things in the two cases, and the normalised figures this
package reports are only comparable within one alphabet.

Detection is deliberately conservative: a sequence counts as nucleotide only
when almost all of it is A, C, G, T, U or N, because the nucleotide codes are a
subset of the protein codes and the cost of the two mistakes is not symmetric.
Reading protein as DNA would silently discard most of the sequence as
ambiguous; reading DNA as protein inflates the word space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

#: Alphabet names accepted on the command line, with 'auto' for detection.
CHOICES: tuple[str, ...] = ("auto", "dna", "rna", "protein")

#: Characters that stand for a missing or deleted residue in an aligned FASTA.
#: They are never counted, and a window containing one is skipped.
GAPS: frozenset[str] = frozenset("-.~")

#: The translation stop written inside a protein sequence. Not a residue.
STOP: str = "*"

_DNA_AMBIGUITY: dict[str, str] = {
    "R": "AG",
    "Y": "CT",
    "S": "CG",
    "W": "AT",
    "K": "GT",
    "M": "AC",
    "B": "CGT",
    "D": "AGT",
    "H": "ACT",
    "V": "ACG",
    "N": "ACGT",
}

_RNA_AMBIGUITY: dict[str, str] = {
    code: residues.replace("T", "U") for code, residues in _DNA_AMBIGUITY.items()
}

#: Protein ambiguity codes. B, Z and J each stand for a pair that mass
#: spectrometry or Edman degradation cannot separate; X stands for any residue.
_PROTEIN_AMBIGUITY: dict[str, str] = {
    "B": "DN",
    "Z": "EQ",
    "J": "IL",
    "X": "ACDEFGHIKLMNPQRSTVWY",
}

#: Residues that are real but outside the counted core: the two genetically
#: encoded residues beyond the twenty. Counting them would make the word space
#: 22 letters wide for every sequence to serve the few that contain them.
_PROTEIN_RARE: dict[str, str] = {
    "U": "selenocysteine",
    "O": "pyrrolysine",
}


@dataclass(frozen=True)
class Alphabet:
    """One residue alphabet.

    Attributes:
        name: the short name used on the command line: 'dna', 'rna' or
            'protein'.
        label: how the alphabet is written in output.
        kind: 'nucleotide' or 'protein'; nucleotide alphabets have a
            complement and so admit canonical counting.
        core: the residues that are counted, in conventional order.
        ambiguity: code to the residues it stands for, for codes that are
            counted as ambiguous rather than as residues.
        rare: residues recognised as valid but not counted, with what they are.
    """

    name: str
    label: str
    kind: str
    core: str
    ambiguity: Mapping[str, str]
    rare: Mapping[str, str]

    @property
    def size(self) -> int:
        """How many residues are counted."""
        return len(self.core)

    @property
    def is_nucleotide(self) -> bool:
        """Whether the alphabet has a complement."""
        return self.kind == "nucleotide"

    @property
    def valid(self) -> frozenset[str]:
        """Every character this alphabet accepts.

        The residues, the ambiguity codes, the rare residues and the alignment
        gaps, plus the translation stop for a protein alphabet. A nucleotide
        alphabet does not accept a stop: '*' in a nucleotide file is not a
        nucleotide.
        """
        extra = {STOP} if not self.is_nucleotide else set()
        return frozenset(
            set(self.core) | set(self.ambiguity) | set(self.rare) | set(GAPS) | extra
        )

    def words(self, k: int) -> int:
        """How many distinct words of length ``k`` the alphabet can spell."""
        if k < 1:
            raise ValueError(f"k must be 1 or greater, got {k}")
        return self.size**k

    def is_core(self, residue: str) -> bool:
        """Whether a residue is counted."""
        return residue in self.core


DNA = Alphabet(
    name="dna",
    label="DNA",
    kind="nucleotide",
    core="ACGT",
    ambiguity=_DNA_AMBIGUITY,
    rare={},
)

RNA = Alphabet(
    name="rna",
    label="RNA",
    kind="nucleotide",
    core="ACGU",
    ambiguity=_RNA_AMBIGUITY,
    rare={},
)

PROTEIN = Alphabet(
    name="protein",
    label="protein",
    kind="protein",
    core="ACDEFGHIKLMNPQRSTVWY",
    ambiguity=_PROTEIN_AMBIGUITY,
    rare=_PROTEIN_RARE,
)

#: Every alphabet, by name.
ALPHABETS: dict[str, Alphabet] = {
    alphabet.name: alphabet for alphabet in (DNA, RNA, PROTEIN)
}

#: Characters any alphabet might accept. The reader checks against this,
#: because the alphabet is not known until a record has been read; the
#: resolved alphabet then makes the stricter check.
ANY_RESIDUE: frozenset[str] = frozenset(
    set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") | set(GAPS) | {STOP}
)

#: Residues that count as evidence of a nucleotide sequence.
_NUCLEOTIDE_EVIDENCE: frozenset[str] = frozenset("ACGTUN")

#: The share of letters that must be nucleotide codes for a sequence to be
#: read as nucleotide. Protein sequences rich in alanine, cysteine, glycine
#: and threonine can reach a surprising share, so the bar is set high.
NUCLEOTIDE_THRESHOLD: float = 0.9

#: Below this many letters, composition carries little information and the
#: detected alphabet is reported as weakly supported rather than as a fact.
SHORT_SEQUENCE: int = 20

_COMPLEMENT = {
    "dna": str.maketrans("ACGTRYSWKMBDHVNacgtryswkmbdhvn", "TGCAYRSWMKVHDBNtgcayrswmkvhdbn"),
    "rna": str.maketrans("ACGURYSWKMBDHVNacguryswkmbdhvn", "UGCAYRSWMKVHDBNugcayrswmkvhdbn"),
}


@dataclass(frozen=True)
class Detection:
    """The outcome of deciding what a sequence is written in.

    Attributes:
        alphabet: the alphabet chosen.
        nucleotide_share: the share of letters that are nucleotide codes.
        letters: how many letters the decision was taken over, gaps and stop
            excluded.
        certain: False when the evidence is thin enough that the caller should
            say so rather than report the alphabet plainly.
    """

    alphabet: Alphabet
    nucleotide_share: float
    letters: int
    certain: bool

    @property
    def qualification(self) -> str:
        """Why the detection is uncertain, or an empty string when it is not."""
        if self.certain:
            return ""
        if not self.letters:
            return "no residues to judge from; assumed DNA"
        return (
            f"only {self.letters} residues to judge from, "
            f"{self.nucleotide_share:.0%} of them nucleotide codes"
        )


def get(name: str) -> Alphabet:
    """Look an alphabet up by name.

    Raises:
        KeyError: if the name is not one of the known alphabets.
    """
    try:
        return ALPHABETS[name.lower()]
    except KeyError:
        known = ", ".join(sorted(ALPHABETS))
        raise KeyError(f"unknown alphabet: {name!r}; choose from {known}") from None


def detect(census: Mapping[str, int]) -> Detection:
    """Decide which alphabet a residue census is written in.

    Args:
        census: how many times each residue occurs, case-insensitively. Gaps
            and the stop character may be present and are ignored.

    Nucleotide is chosen when at least ``NUCLEOTIDE_THRESHOLD`` of the letters
    are nucleotide codes; RNA rather than DNA when the sequence carries more
    U than T. A sequence with both in quantity is read as DNA, since that is
    what a file mixing the two conventions almost always is.
    """
    letters: dict[str, int] = {}
    for residue, count in census.items():
        upper = residue.upper()
        if upper in GAPS or upper == STOP:
            continue
        letters[upper] = letters.get(upper, 0) + count

    total = sum(letters.values())
    if not total:
        return Detection(DNA, 0.0, 0, certain=False)

    nucleotide = sum(
        count for residue, count in letters.items() if residue in _NUCLEOTIDE_EVIDENCE
    )
    share = nucleotide / total
    if share < NUCLEOTIDE_THRESHOLD:
        return Detection(PROTEIN, share, total, certain=total >= SHORT_SEQUENCE)

    thymine = letters.get("T", 0)
    uracil = letters.get("U", 0)
    alphabet = RNA if uracil > thymine else DNA
    return Detection(alphabet, share, total, certain=total >= SHORT_SEQUENCE)


def unknown_residues(census: Mapping[str, int], alphabet: Alphabet) -> list[str]:
    """Residues in a census that the alphabet does not accept, sorted."""
    accepted = alphabet.valid
    return sorted(
        {
            residue.upper()
            for residue in census
            if residue.upper() not in accepted
        }
    )


def reverse_complement(sequence: str, alphabet: Alphabet) -> str:
    """Reverse-complement a nucleotide sequence, ambiguity codes included.

    Gaps and any character outside the alphabet are passed through unchanged,
    so the result has the same length as the input.

    Raises:
        ValueError: if the alphabet has no complement.
    """
    if not alphabet.is_nucleotide:
        raise ValueError(f"{alphabet.label} has no complement")
    return sequence.translate(_COMPLEMENT[alphabet.name])[::-1]
