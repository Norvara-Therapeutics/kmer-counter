"""k-mer spectra, sequence complexity and word composition.

Positions are 1-based and inclusive, as they are in a sequence viewer: a
region reported as 101-164 includes both ends and is 64 residues long. Counts
are taken over a window that advances one residue at a time, so a sequence of
L residues offers L - k + 1 words of length k and the counts of neighbouring
windows are not independent.

Entropies are in bits. Shares are held as fractions and rendered as
percentages only for a human reader; the TSV and JSON outputs carry the
unrounded values.
"""

from kmer_counter.alphabet import (
    ALPHABETS,
    DNA,
    PROTEIN,
    RNA,
    Alphabet,
    Detection,
    detect,
    reverse_complement,
)
from kmer_counter.analysis import Analysis, RecordSummary, analyse
from kmer_counter.complexity import (
    DUST_THRESHOLD,
    DUST_WINDOW,
    SEG_THRESHOLD,
    SEG_WINDOW,
    Entropy,
    compositional_complexity,
    entropy_of,
    linguistic_complexity,
    residue_entropy,
    shannon_entropy,
    triplet_score,
)
from kmer_counter.composition import Table, Word, dinucleotide_ratio, signature, table
# ``distance`` itself is not re-exported here: the name would shadow the
# submodule, so ``import kmer_counter.distance`` would bind the function.
from kmer_counter.distance import Matrix, matrix
from kmer_counter.profile import Interval, Profile, Window, scan
from kmer_counter.reader import Record, SequenceError, read_one, read_records
from kmer_counter.repeats import (
    Occurrence,
    Run,
    Tandem,
    abundant_words,
    homopolymer_runs,
    tandem_repeats,
)
from kmer_counter.spectrum import K_MAX, K_MIN, Spectrum, count, count_all, merge

__version__ = "1.0.0rc1"

__all__ = [
    "ALPHABETS",
    "Alphabet",
    "Analysis",
    "DNA",
    "DUST_THRESHOLD",
    "DUST_WINDOW",
    "Detection",
    "Entropy",
    "Interval",
    "K_MAX",
    "K_MIN",
    "Matrix",
    "Occurrence",
    "PROTEIN",
    "Profile",
    "RNA",
    "Record",
    "RecordSummary",
    "Run",
    "SEG_THRESHOLD",
    "SEG_WINDOW",
    "SequenceError",
    "Spectrum",
    "Table",
    "Tandem",
    "Window",
    "Word",
    "__version__",
    "abundant_words",
    "analyse",
    "compositional_complexity",
    "count",
    "count_all",
    "detect",
    "dinucleotide_ratio",
    "entropy_of",
    "homopolymer_runs",
    "linguistic_complexity",
    "matrix",
    "merge",
    "read_one",
    "read_records",
    "residue_entropy",
    "reverse_complement",
    "scan",
    "shannon_entropy",
    "signature",
    "table",
    "tandem_repeats",
    "triplet_score",
]
