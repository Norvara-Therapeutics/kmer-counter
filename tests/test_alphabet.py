"""What ``kmer_counter.alphabet`` must keep true.

The alphabet decides which residues are counted, which stand for several
residues and so cannot be counted, and how wide the word space is, so every
number this package reports rests on it. These tests protect three things: the
shape of each alphabet (size, what it accepts, what it complements), the
conservative detection rule and the boundary it turns on, and the honesty of
the ``certain`` flag and its qualification text when the evidence is thin.

Expected values are derived from the documented rule, not read off a run: the
nucleotide share is a hand-counted fraction, the reverse complement of an
ambiguity code is looked up from the pair it stands for, and the word count is
the alphabet size raised to k.
"""

from collections import Counter

import pytest

from kmer_counter import alphabet as ab
from kmer_counter.reader import read_one


# --- the alphabets themselves -------------------------------------------


def test_the_command_line_offers_detection_and_the_three_alphabets():
    assert ab.CHOICES == ("auto", "dna", "rna", "protein")


def test_every_alphabet_is_registered_under_its_own_name():
    assert set(ab.ALPHABETS) == {"dna", "rna", "protein"}
    for name, alphabet in ab.ALPHABETS.items():
        assert alphabet.name == name


def test_get_looks_an_alphabet_up_whatever_the_case():
    assert ab.get("dna") is ab.DNA
    assert ab.get("DNA") is ab.DNA
    assert ab.get("Protein") is ab.PROTEIN


def test_get_raises_on_an_unknown_name_and_lists_the_known_ones():
    with pytest.raises(KeyError, match="unknown alphabet: 'peptide'"):
        ab.get("peptide")
    with pytest.raises(KeyError, match="choose from dna, protein, rna"):
        ab.get("peptide")


def test_dna_and_rna_count_four_residues_and_protein_twenty():
    assert ab.DNA.size == 4
    assert ab.RNA.size == 4
    assert ab.PROTEIN.size == 20
    assert ab.DNA.core == "ACGT"
    assert ab.RNA.core == "ACGU"


def test_only_the_nucleotide_alphabets_have_a_complement():
    assert ab.DNA.is_nucleotide
    assert ab.RNA.is_nucleotide
    assert not ab.PROTEIN.is_nucleotide
    assert ab.DNA.kind == "nucleotide"
    assert ab.PROTEIN.kind == "protein"


def test_words_is_the_alphabet_size_raised_to_the_word_length():
    # Four residues spell 4**k words; twenty spell 20**k. A protein 4-mer is
    # drawn from 160,000 possibilities and a DNA 4-mer from 256.
    assert ab.DNA.words(1) == 4
    assert ab.DNA.words(4) == 256
    assert ab.DNA.words(8) == 65536
    assert ab.PROTEIN.words(4) == 160000


def test_words_refuses_a_word_shorter_than_one_residue():
    with pytest.raises(ValueError, match="k must be 1 or greater, got 0"):
        ab.DNA.words(0)
    with pytest.raises(ValueError, match="k must be 1 or greater, got -3"):
        ab.PROTEIN.words(-3)


def test_every_alphabet_accepts_the_three_gap_characters():
    assert ab.GAPS == frozenset("-.~")
    for alphabet in (ab.DNA, ab.RNA, ab.PROTEIN):
        assert ab.GAPS <= alphabet.valid


def test_only_protein_accepts_the_stop_character():
    # The stop is written inside a protein sequence; it has no meaning in a
    # nucleotide one, where the three bases that spell it are ordinary bases.
    assert ab.STOP in ab.PROTEIN.valid
    assert ab.STOP not in ab.DNA.valid
    assert ab.STOP not in ab.RNA.valid


def test_protein_accepts_every_letter_of_the_latin_alphabet():
    # The twenty core residues plus the four ambiguity codes (B, Z, J, X) and
    # the two rare encoded residues (U, O) exhaust the twenty-six letters, so
    # no letter is ever reported as unknown for a protein.
    assert set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") <= ab.PROTEIN.valid


def test_each_nucleotide_alphabet_refuses_the_other_ones_uracil_or_thymine():
    assert "U" not in ab.DNA.valid
    assert "T" not in ab.RNA.valid


def test_the_rare_encoded_residues_are_accepted_but_not_counted():
    # Selenocysteine and pyrrolysine are real residues; counting them would
    # widen the word space to 22 letters for every sequence.
    assert set(ab.PROTEIN.rare) == {"U", "O"}
    for residue in ab.PROTEIN.rare:
        assert residue in ab.PROTEIN.valid
        assert not ab.PROTEIN.is_core(residue)


def test_is_core_separates_counted_residues_from_ambiguity_codes():
    assert ab.DNA.is_core("A")
    assert not ab.DNA.is_core("N")
    assert not ab.DNA.is_core("R")
    assert ab.PROTEIN.is_core("W")
    assert not ab.PROTEIN.is_core("X")


def test_ambiguity_codes_stand_for_the_residues_they_abbreviate():
    # R is a purine (A or G); its RNA form replaces T with U throughout.
    assert ab.DNA.ambiguity["R"] == "AG"
    assert ab.DNA.ambiguity["Y"] == "CT"
    assert ab.RNA.ambiguity["Y"] == "CU"
    assert ab.DNA.ambiguity["N"] == "ACGT"
    assert ab.RNA.ambiguity["N"] == "ACGU"
    assert ab.PROTEIN.ambiguity["B"] == "DN"
    assert ab.PROTEIN.ambiguity["Z"] == "EQ"
    assert ab.PROTEIN.ambiguity["J"] == "IL"
    assert len(ab.PROTEIN.ambiguity["X"]) == 20


def test_any_residue_is_the_union_of_what_the_alphabets_accept():
    # The reader checks against this before the alphabet is known, so it must
    # not refuse a character any alphabet would later accept.
    for alphabet in (ab.DNA, ab.RNA, ab.PROTEIN):
        assert alphabet.valid <= ab.ANY_RESIDUE


# --- detection ----------------------------------------------------------


def test_a_pure_nucleotide_census_is_read_as_dna():
    detection = ab.detect({"A": 30, "C": 25, "G": 25, "T": 20})
    assert detection.alphabet is ab.DNA
    assert detection.nucleotide_share == pytest.approx(1.0)
    assert detection.letters == 100
    assert detection.certain


def test_more_uracil_than_thymine_is_read_as_rna():
    detection = ab.detect({"A": 30, "C": 25, "G": 25, "U": 20})
    assert detection.alphabet is ab.RNA
    assert detection.nucleotide_share == pytest.approx(1.0)


def test_equal_uracil_and_thymine_is_read_as_dna():
    # A file mixing the two conventions is almost always DNA, and the rule is
    # strict inequality: RNA only when U outnumbers T.
    detection = ab.detect({"A": 30, "C": 20, "G": 20, "T": 15, "U": 15})
    assert detection.alphabet is ab.DNA


def test_a_trace_of_thymine_in_an_rna_sequence_still_reads_as_rna():
    detection = ab.detect({"A": 30, "C": 25, "G": 25, "U": 19, "T": 1})
    assert detection.alphabet is ab.RNA


def test_exactly_nine_tenths_nucleotide_codes_is_still_nucleotide():
    # 90 of 100 letters are nucleotide codes, so the share is exactly the
    # threshold and the rule admits it: the test is share < threshold.
    detection = ab.detect({"A": 90, "L": 10})
    assert ab.NUCLEOTIDE_THRESHOLD == pytest.approx(0.9)
    assert detection.nucleotide_share == pytest.approx(0.9)
    assert detection.alphabet is ab.DNA


def test_one_letter_below_the_threshold_is_read_as_protein():
    # 89 of 100 letters are nucleotide codes: 0.89 < 0.9, so protein.
    detection = ab.detect({"A": 89, "L": 11})
    assert detection.nucleotide_share == pytest.approx(0.89)
    assert detection.alphabet is ab.PROTEIN


def test_detection_is_case_insensitive():
    lower = ab.detect({"a": 30, "c": 25, "g": 25, "t": 20})
    mixed = ab.detect({"A": 15, "a": 15, "C": 25, "G": 25, "T": 20})
    assert lower.alphabet is ab.DNA
    assert mixed.alphabet is ab.DNA
    assert mixed.letters == 100


def test_only_the_six_unambiguous_codes_count_as_nucleotide_evidence():
    # N is evidence because an N-rich assembly is still DNA, but R, Y and the
    # other ambiguity codes are not: they are protein residues too.
    assert ab.detect({"N": 100}).alphabet is ab.DNA
    assert ab.detect({"R": 100}).alphabet is ab.PROTEIN
    assert ab.detect({"R": 100}).nucleotide_share == pytest.approx(0.0)


def test_gaps_and_the_stop_character_do_not_enter_the_decision():
    # 20 nucleotide letters beside 1,000 gaps is a 20-letter decision taken
    # over a 100% nucleotide census, not a 2% one.
    detection = ab.detect({"A": 10, "C": 10, "-": 500, ".": 300, "~": 200, "*": 50})
    assert detection.letters == 20
    assert detection.nucleotide_share == pytest.approx(1.0)
    assert detection.alphabet is ab.DNA
    assert detection.certain


def test_a_census_of_nothing_but_gaps_has_no_residues_to_judge_from():
    detection = ab.detect({"-": 400, "*": 7})
    assert detection.letters == 0
    assert not detection.certain
    assert detection.qualification == "no residues to judge from; assumed DNA"


def test_an_empty_census_is_reported_as_an_assumption_of_dna():
    detection = ab.detect({})
    assert detection.alphabet is ab.DNA
    assert detection.nucleotide_share == pytest.approx(0.0)
    assert detection.letters == 0
    assert not detection.certain
    assert detection.qualification == "no residues to judge from; assumed DNA"


def test_a_sequence_shorter_than_twenty_residues_is_weakly_supported():
    # SHORT_SEQUENCE is 20 and the rule is total >= SHORT_SEQUENCE, so 19
    # letters is thin evidence and 20 is enough.
    assert ab.SHORT_SEQUENCE == 20
    thin = ab.detect({"A": 10, "C": 9})
    assert thin.letters == 19
    assert not thin.certain
    enough = ab.detect({"A": 10, "C": 10})
    assert enough.letters == 20
    assert enough.certain


def test_the_qualification_names_the_residue_count_and_the_share():
    # 9 of 18 letters are nucleotide codes, so the share renders as 50%.
    detection = ab.detect({"A": 9, "L": 9})
    assert detection.letters == 18
    assert detection.qualification == (
        "only 18 residues to judge from, 50% of them nucleotide codes"
    )


def test_a_certain_detection_carries_no_qualification():
    detection = ab.detect({"A": 500, "C": 500})
    assert detection.certain
    assert detection.qualification == ""


def test_a_short_protein_is_qualified_as_well_as_a_short_nucleotide():
    # The flag is about how much evidence there was, not about which way the
    # decision fell, so the protein branch qualifies on the same count.
    detection = ab.detect({"L": 10})
    assert detection.alphabet is ab.PROTEIN
    assert not detection.certain
    assert "only 10 residues to judge from" in detection.qualification


def test_the_bundled_nucleotide_examples_are_detected_as_dna(examples):
    # aav2.fasta and phix174.fasta are plain ACGT records with no ambiguity
    # codes, so the share is 1.0 and the detection is certain.
    for name, length in (("aav2.fasta", 4679), ("phix174.fasta", 5386)):
        record = read_one(examples / name)
        assert len(record) == length
        detection = ab.detect(Counter(record.sequence))
        assert detection.alphabet is ab.DNA
        assert detection.nucleotide_share == pytest.approx(1.0)
        assert detection.certain


def test_the_bundled_protein_examples_are_detected_as_protein(examples):
    # Both are far below the 90% bar: ACGTUN covers only the alanine,
    # cysteine, glycine, threonine and asparagine content of a real protein.
    for name, length in (("aav2_vp1.fasta", 735), ("huntingtin.fasta", 3144)):
        record = read_one(examples / name)
        assert len(record) == length
        detection = ab.detect(Counter(record.sequence))
        assert detection.alphabet is ab.PROTEIN
        assert detection.nucleotide_share < ab.NUCLEOTIDE_THRESHOLD
        assert detection.certain


# --- unknown residues ---------------------------------------------------


def test_unknown_residues_lists_what_the_alphabet_refuses_sorted():
    census = {"A": 10, "C": 5, "E": 2, "Q": 1, "N": 3}
    assert ab.unknown_residues(census, ab.DNA) == ["E", "Q"]


def test_unknown_residues_is_empty_when_every_residue_is_accepted():
    census = {"A": 10, "C": 5, "G": 5, "T": 5, "N": 1, "R": 1, "-": 4}
    assert ab.unknown_residues(census, ab.DNA) == []


def test_unknown_residues_never_fires_on_letters_for_protein():
    census = {letter: 1 for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ*-"}
    assert ab.unknown_residues(census, ab.PROTEIN) == []


def test_unknown_residues_reports_thymine_against_rna_and_uracil_against_dna():
    assert ab.unknown_residues({"A": 1, "T": 1}, ab.RNA) == ["T"]
    assert ab.unknown_residues({"A": 1, "U": 1}, ab.DNA) == ["U"]


def test_unknown_residues_folds_case_and_reports_once():
    assert ab.unknown_residues({"e": 3, "E": 4, "A": 1}, ab.DNA) == ["E"]


def test_unknown_residues_reports_non_letters():
    assert ab.unknown_residues({"A": 1, "1": 1, "?": 1}, ab.DNA) == ["1", "?"]


# --- reverse complement -------------------------------------------------


def test_reverse_complement_of_a_palindrome_is_the_sequence_itself():
    # ACGT reads the same on both strands: A-T, C-G, G-C, T-A reversed.
    assert ab.reverse_complement("ACGT", ab.DNA) == "ACGT"
    assert ab.reverse_complement("ACGU", ab.RNA) == "ACGU"


def test_reverse_complement_pairs_each_base_with_its_partner():
    # AAAACCCGGT complements base by base to TTTTGGGCCA, and reversing that
    # gives ACCGGGTTTT.
    assert ab.reverse_complement("AAAACCCGGT", ab.DNA) == "ACCGGGTTTT"
    assert ab.reverse_complement("A", ab.DNA) == "T"
    assert ab.reverse_complement("G", ab.DNA) == "C"


def test_rna_complements_adenine_with_uracil():
    assert ab.reverse_complement("AAA", ab.RNA) == "UUU"
    assert ab.reverse_complement("UUU", ab.RNA) == "AAA"


def test_reverse_complement_maps_each_ambiguity_code_to_its_partner_code():
    # R(AG)<->Y(CT), K(GT)<->M(AC), B(CGT)<->V(ACG), D(AGT)<->H(ACT), while
    # S(CG), W(AT) and N are their own complements. Reversing the complement
    # of RYSWKMBDHVN therefore gives NBDHVKMWSRY.
    assert ab.reverse_complement("RYSWKMBDHVN", ab.DNA) == "NBDHVKMWSRY"


def test_reverse_complement_preserves_the_case_of_each_position():
    # Case travels with the residue, so it is reversed along with it:
    # aCgT complements to tGcA, which reversed is AcGt.
    assert ab.reverse_complement("acgt", ab.DNA) == "acgt"
    assert ab.reverse_complement("aCgT", ab.DNA) == "AcGt"
    assert ab.reverse_complement("aaaa", ab.DNA) == "tttt"


def test_reverse_complement_applied_twice_restores_the_sequence():
    sequence = "ACGTTGCARYSWKMBDHVNacgt"
    once = ab.reverse_complement(sequence, ab.DNA)
    assert ab.reverse_complement(once, ab.DNA) == sequence


def test_reverse_complement_passes_gaps_and_foreign_characters_through():
    # The result must have the same length as the input, so nothing is
    # dropped: a gap stays a gap and a thymine in an RNA sequence stays a T.
    # AC-GT: complement of A, C, G, T is T, G, C, A with the gap untouched,
    # giving TG-CA, which reversed is AC-GT again.
    gapped = ab.reverse_complement("AC-GT", ab.DNA)
    assert gapped == "AC-GT"
    assert len(gapped) == 5
    # T is foreign to RNA, so it is carried through rather than complemented:
    # ACGT -> UGC T -> reversed TCGU.
    assert ab.reverse_complement("ACGT", ab.RNA) == "TCGU"
    assert len(ab.reverse_complement("ACGU?", ab.RNA)) == 5


def test_reverse_complement_refuses_an_alphabet_without_a_complement():
    with pytest.raises(ValueError, match="protein has no complement"):
        ab.reverse_complement("MAAD", ab.PROTEIN)


def test_the_empty_sequence_reverse_complements_to_itself():
    assert ab.reverse_complement("", ab.DNA) == ""
