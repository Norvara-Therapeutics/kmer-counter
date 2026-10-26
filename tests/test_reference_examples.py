"""The bundled records, checked against the public records they came from.

``examples/`` holds four unmodified NCBI records. Each assertion below is a
fact about the record as it is published - its accession, its length, what it
is written in, where its low-complexity tract sits - so a change in the
bundled file or a change in how the package reads one shows up here rather
than as a quietly different number in a report.

The records are:

* ``aav2.fasta``, NC_001401.2, the AAV2 genome, 4,679 bp, with the inverted
  terminal repeats at either end;
* ``phix174.fasta``, NC_001422.1, the phiX174 genome, 5,386 bp, the sequencing
  control and the first genome ever sequenced;
* ``aav2_vp1.fasta``, YP_680426.1, the AAV2 VP1 capsid protein, 735 aa;
* ``huntingtin.fasta``, NP_002102.4, huntingtin isoform 2, 3,144 aa, whose
  N-terminus carries the polyglutamine tract the disease is named for and the
  polyproline tract immediately after it.
"""

import pytest

from kmer_counter.analysis import analyse
from kmer_counter.reader import read_one

#: accession, published length, and the alphabet the record is written in.
_RECORDS = [
    ("aav2.fasta", "NC_001401.2", 4679, "dna"),
    ("phix174.fasta", "NC_001422.1", 5386, "dna"),
    ("aav2_vp1.fasta", "YP_680426.1", 735, "protein"),
    ("huntingtin.fasta", "NP_002102.4", 3144, "protein"),
]


@pytest.fixture(scope="module")
def huntingtin(examples):
    return analyse(str(examples / "huntingtin.fasta"), k=4)


@pytest.mark.parametrize(("filename", "accession", "length", "_alphabet"), _RECORDS)
def test_each_bundled_record_has_its_published_identifier_and_length(
    filename, accession, length, _alphabet, examples
):
    record = read_one(str(examples / filename))
    assert record.identifier == accession
    assert len(record.sequence) == length


@pytest.mark.parametrize(("filename", "_accession", "_length", "alphabet"), _RECORDS)
def test_each_bundled_record_is_read_in_the_alphabet_it_is_written_in(
    filename, _accession, _length, alphabet, examples
):
    # Detection is conservative by design: the nucleotide codes are a subset
    # of the protein codes, so a sequence is only called nucleotide when
    # almost all of it is A, C, G, T, U or N. The AAV2 genome and its own
    # capsid protein are the pair that would be confused if it were not.
    result = analyse(str(examples / filename), k=1)
    assert result.alphabet.name == alphabet
    assert result.detection is not None
    assert result.detection.certain


def test_the_aav2_genome_and_its_capsid_protein_are_told_apart(examples):
    genome = analyse(str(examples / "aav2.fasta"), k=1)
    capsid = analyse(str(examples / "aav2_vp1.fasta"), k=1)
    assert genome.alphabet.name == "dna"
    assert genome.alphabet.size == 4
    assert capsid.alphabet.name == "protein"
    assert capsid.alphabet.size == 20
    # The published genome carries no ambiguity code at all, so every letter
    # of it is nucleotide evidence.
    assert genome.detection.nucleotide_share == pytest.approx(1.0)
    assert genome.records[0].ambiguous == 0


def test_huntingtin_calls_its_n_terminal_low_complexity_tract(huntingtin):
    # The polyglutamine tract of huntingtin begins at residue 18 of the
    # isoform bundled here and is followed immediately by a polyproline
    # tract; a 12-residue window of either is far below 2.2 bits, so the
    # first region called must open inside the N-terminal 90 residues.
    regions = huntingtin.records[0].regions
    assert regions
    first = regions[0]
    assert first.start <= 90
    assert first.end > first.start
    # The tract itself is 18-40, so the region has to cover it.
    assert first.start <= 18 and first.end >= 40
    # A window of a single repeated residue has no residue entropy at all.
    assert first.peak == pytest.approx(0.0)


def test_huntingtin_is_most_abundant_in_the_word_its_disease_is_named_for(
    huntingtin,
):
    # A perfect run of n glutamines holds n - 3 windows of QQQQ, so the polyQ
    # tract alone makes QQQQ the most abundant 4-mer of the protein.
    word, count = huntingtin.spectrum.most_common(1)[0]
    assert word == "QQQQ"
    run = huntingtin.records[0].longest_run
    assert run is not None
    assert run.residue == "Q"
    assert count >= run.length - 3


def test_huntingtin_carries_a_long_perfect_tandem_repeat(huntingtin):
    longest = huntingtin.records[0].longest_tandem
    assert longest is not None
    # The polyglutamine tract is a perfect tandem repeat of period one, and
    # it is the longest perfect repeat in the protein.
    assert longest.unit == "Q"
    assert longest.period == 1
    assert longest.length >= 20
    assert longest.copies == pytest.approx(float(longest.length))


def test_phix174_carries_no_long_perfect_tandem_repeat(examples, huntingtin):
    # A 5.4 kb genome of ordinary composition has no perfect repeat to speak
    # of: the longest here is a few units of a short period, an order of
    # magnitude shorter than huntingtin's polyglutamine tract. The bound is
    # deliberately loose - the claim is 'nothing remarkable', not an exact
    # length - and it is checked against the protein in the same run.
    phix = analyse(str(examples / "phix174.fasta"), k=4)
    longest = phix.records[0].longest_tandem
    assert longest is not None
    assert longest.length <= 12
    assert longest.length < huntingtin.records[0].longest_tandem.length


def test_the_nucleotide_genomes_reach_no_dust_region(examples):
    # Neither genome has a 64-base window repetitive enough to score 20 on
    # DUST, which is what a genome with no satellite and no homopolymer tract
    # of any length looks like.
    for filename in ("aav2.fasta", "phix174.fasta"):
        result = analyse(str(examples / filename), k=4)
        assert result.regions == 0
        assert result.masked == 0
        assert result.masked_share == pytest.approx(0.0)


def test_the_aav2_genome_opens_on_its_inverted_terminal_repeat(examples):
    # The inverted terminal repeat occupies the first 145 bases of the
    # published genome and is the most structured part of it; at least one
    # perfect tandem repeat falls inside it, which can be checked by reading
    # the first 145 bases of examples/aav2.fasta.
    result = analyse(str(examples / "aav2.fasta"), k=4)
    inside_itr = [
        repeat for repeat in result.records[0].tandems if repeat.end <= 145
    ]
    assert inside_itr
