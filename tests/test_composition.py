"""What a word count is read against, and when it cannot be read at all.

A count means nothing without an expectation, so every rule here is about the
expectation: the zero-order one is the number of windows times the product of
the residue frequencies, the maximal-order Markov one is n(prefix) n(suffix) /
n(middle) with the empty middle word standing for the single-residue window
count, and neither is reported when it falls below one count, where a ratio
would be noise. The file also protects the refusals - canonical counts, an
unknown model, a missing prerequisite spectrum - because a table built from
the wrong counts is worse than no table.
"""

from math import sqrt

import pytest

from kmer_counter.alphabet import DNA, PROTEIN
from kmer_counter.composition import (
    MIN_EXPECTED,
    Word,
    _enumerate_words,
    dinucleotide_ratio,
    expected_markov,
    expected_zero_order,
    gc_content,
    shares,
    signature,
    table,
)
from kmer_counter.spectrum import count

#: Eight copies of AACCGGTT: 64 bases, each base exactly sixteen of them, so
#: every residue frequency is 1/4 and every dinucleotide is expected
#: 63 / 16 = 3.9375 times over the 63 windows. Observed counts are eight for
#: each of the eight dinucleotides inside a copy and seven for the TA at the
#: seven junctions - 8 * 7 + 7 = 63, which is every window.
BALANCED = "AACCGGTT" * 8
EXPECTED_DINUCLEOTIDE = 63 / 16


@pytest.fixture
def balanced() -> dict:
    """Spectra of BALANCED at the word lengths the models need."""
    return {k: count(BALANCED, k, DNA) for k in (1, 2, 3)}


def test_the_word_space_is_enumerated_in_alphabetical_order():
    assert _enumerate_words(DNA, 1) == ["A", "C", "G", "T"]
    assert _enumerate_words(DNA, 2)[:5] == ["AA", "AC", "AG", "AT", "CA"]
    assert _enumerate_words(DNA, 2) == sorted(_enumerate_words(DNA, 2))
    assert _enumerate_words(DNA, 3)[-1] == "TTT"


def test_a_table_lists_the_whole_word_space_however_little_of_it_occurs(balanced):
    rows = table(balanced, 2)
    assert len(rows.words) == DNA.words(2) == 16
    assert [row.word for row in rows.words] == _enumerate_words(DNA, 2)
    # CA never occurs in AACCGGTT, and the point of the full enumeration is
    # that it is visible as a zero rather than missing.
    assert rows.get("CA").count == 0
    assert len(rows.observed) == 8


def test_a_protein_table_is_twenty_rows_at_one_residue():
    spectra = {1: count("ACDEFGHIKLMNPQRSTVWY" * 4, 1, PROTEIN)}
    rows = table(spectra, 1)
    assert len(rows.words) == PROTEIN.size == 20
    assert rows.alphabet is PROTEIN


def test_the_zero_order_expectation_is_windows_times_the_residue_frequencies(balanced):
    # Each base is a quarter of the 64 residues, so any dinucleotide is
    # expected (1/4)(1/4) * 63 = 3.9375 times and any trinucleotide
    # (1/4)^3 * 62 = 0.96875 times.
    assert expected_zero_order("AC", balanced[1], balanced[2].windows) == pytest.approx(
        EXPECTED_DINUCLEOTIDE
    )
    assert expected_zero_order("ACG", balanced[1], balanced[3].windows) == pytest.approx(
        62 / 64
    )


def test_the_zero_order_expectation_is_zero_without_windows_to_spread_over(balanced):
    assert expected_zero_order("AC", balanced[1], 0) == 0.0
    assert expected_zero_order("AC", count("NN", 1, DNA), 63) == 0.0


def test_a_table_carries_the_observed_count_beside_that_expectation(balanced):
    rows = table(balanced, 2)
    aa = rows.get("AA")
    assert aa.count == 8
    assert aa.frequency == pytest.approx(8 / 63)
    assert aa.expected == pytest.approx(EXPECTED_DINUCLEOTIDE)
    assert aa.model == "zero-order"
    # Eight observed against 3.9375 expected is 128/63 = 2.03 times over.
    assert aa.ratio == pytest.approx(8 / EXPECTED_DINUCLEOTIDE)


def test_the_markov_expectation_is_the_prefix_and_suffix_over_the_middle():
    # ACGACG holds AC twice, CG twice and C twice, so ACG is expected
    # 2 * 2 / 2 = 2 times, which is exactly how often it occurs.
    spectra = {k: count("ACGACG", k, DNA) for k in (1, 2, 3)}
    assert spectra[2].counts["AC"] == 2
    assert spectra[2].counts["CG"] == 2
    assert spectra[1].counts["C"] == 2
    assert expected_markov("ACG", spectra) == pytest.approx(2.0)
    assert spectra[3].counts["ACG"] == 2


def test_the_markov_expectation_of_a_dinucleotide_divides_by_the_window_count():
    # For a word of two the middle is the empty word, whose count is the
    # number of single-residue windows: A and C occur twice each over six
    # windows, so AC is expected 2 * 2 / 6 = 2/3 times.
    spectra = {k: count("ACGACG", k, DNA) for k in (1, 2)}
    assert spectra[1].windows == 6
    assert expected_markov("AC", spectra) == pytest.approx(2 / 3)


def test_the_markov_expectation_is_zero_when_the_middle_word_is_absent():
    spectra = {k: count("ACGACG", k, DNA) for k in (1, 2, 3)}
    assert expected_markov("ATA", spectra) == 0.0


def test_the_markov_expectation_needs_a_word_of_two_residues(balanced):
    with pytest.raises(ValueError, match="two or more residues"):
        expected_markov("A", balanced)


def test_the_markov_expectation_raises_when_a_spectrum_it_needs_is_missing(balanced):
    with pytest.raises(KeyError):
        expected_markov("ACG", {1: balanced[1]})


def test_a_markov_table_asks_whether_a_word_beats_its_own_two_halves(balanced):
    rows = table(balanced, 3, "markov")
    assert rows.model == "markov"
    # AAC is expected n(AA) n(AC) / n(A) = 8 * 8 / 16 = 4 times and occurs
    # eight, so it is twice what its halves imply.
    assert rows.get("AAC").expected == pytest.approx(4.0)
    assert rows.get("AAC").count == 8
    assert rows.get("AAC").ratio == pytest.approx(2.0)


def test_a_ratio_and_a_z_score_are_withheld_below_one_expected_count():
    # MIN_EXPECTED is one count: below it the Poisson spread is wider than
    # the quantity being measured, so neither figure is reported.
    thin = Word(word="AA", count=1, frequency=0.1, expected=MIN_EXPECTED - 0.01, model="zero-order")
    assert thin.ratio is None
    assert thin.z_score is None
    assert thin.direction == "not assessed"


def test_the_ratio_and_z_score_are_observed_over_and_against_expected():
    word = Word(word="AA", count=10, frequency=0.1, expected=5.0, model="zero-order")
    assert word.ratio == pytest.approx(2.0)
    # The Poisson z-score is (10 - 5) / sqrt(5).
    assert word.z_score == pytest.approx(5 / sqrt(5))


def test_a_word_is_called_over_or_under_at_a_tenth_either_way():
    def judge(count_: int, expected: float) -> str:
        return Word(
            word="AA", count=count_, frequency=0.0, expected=expected, model="zero-order"
        ).direction

    assert judge(11, 10.0) == "over"
    assert judge(109, 100.0) == "as expected"
    assert judge(10, 10.0) == "as expected"
    assert judge(91, 100.0) == "as expected"
    assert judge(9, 10.0) == "under"
    assert judge(0, 10.0) == "under"


def test_a_table_refuses_canonical_counts(balanced):
    canonical = {k: count(BALANCED, k, DNA, canonical=True) for k in (1, 2)}
    with pytest.raises(ValueError, match="canonical counts"):
        table(canonical, 2)


def test_a_table_refuses_an_unknown_model(balanced):
    with pytest.raises(ValueError, match="unknown model: 'poisson'"):
        table(balanced, 2, "poisson")


def test_a_table_refuses_a_word_length_it_was_not_given(balanced):
    with pytest.raises(ValueError, match="no spectrum for k=4"):
        table(balanced, 4)


def test_a_table_refuses_a_model_whose_prerequisite_spectrum_is_missing(balanced):
    with pytest.raises(ValueError, match=r"zero-order model at k=2 needs the k=1"):
        table({2: balanced[2]}, 2)
    with pytest.raises(ValueError, match=r"markov model at k=3 needs the k=2"):
        table({1: balanced[1], 3: balanced[3]}, 3, "markov")


def test_a_markov_table_of_single_residues_is_refused(balanced):
    # A residue has no prefix and no suffix, so the model has nothing to
    # condition on. (The message names a k=0 spectrum rather than saying so;
    # see the defect noted with this suite.)
    with pytest.raises(ValueError):
        table(balanced, 1, "markov")


def test_observed_only_lists_the_words_that_occur(balanced):
    rows = table(balanced, 2, observed_only=True)
    assert len(rows.words) == balanced[2].distinct == 8
    assert all(row.count > 0 for row in rows.words)
    assert [row.word for row in rows.words] == sorted(balanced[2].counts)
    assert rows.get("CA") is None


def test_a_table_has_no_row_for_a_word_of_another_length(balanced):
    assert table(balanced, 2).get("AAA") is None


def test_shares_are_each_word_s_fraction_of_the_counted_windows(balanced):
    fractions = shares(balanced[2])
    assert fractions["AA"] == pytest.approx(8 / 63)
    assert sum(fractions.values()) == pytest.approx(1.0)
    assert shares(count("NN", 2, DNA)) == {}


def test_the_dinucleotide_odds_ratio_is_the_observed_count_over_that_expectation(balanced):
    # CG occurs eight times against 3.9375 expected: 128/63 = 2.03, a
    # CpG-rich sequence by the measure vertebrate DNA scores 0.2 on.
    assert dinucleotide_ratio("CG", balanced) == pytest.approx(128 / 63)
    assert dinucleotide_ratio("CA", balanced) == 0.0


def test_a_dinucleotide_whose_bases_are_too_rare_is_not_assessed():
    # ACAC has no G at all, so CG is expected 0 times and no ratio is taken.
    spectra = {k: count("ACAC" * 20, k, DNA) for k in (1, 2)}
    assert dinucleotide_ratio("CG", spectra) is None
    assert dinucleotide_ratio("CA", spectra) is not None


def test_the_dinucleotide_ratio_refuses_a_word_that_is_not_a_pair(balanced):
    with pytest.raises(ValueError, match="expected a dinucleotide"):
        dinucleotide_ratio("ACG", balanced)


def test_the_dinucleotide_ratio_needs_both_spectra_it_reads(balanced):
    with pytest.raises(ValueError, match="needs the k=1 and k=2 spectra"):
        dinucleotide_ratio("CG", {1: balanced[1]})


def test_the_signature_is_every_assessable_dinucleotide_ratio(balanced):
    signed = signature(balanced)
    # Every dinucleotide is expected 3.9375 times here, so all sixteen are
    # assessable even though only eight of them occur.
    assert len(signed) == 16
    assert signed["AA"] == pytest.approx(128 / 63)
    assert signed["CA"] == 0.0


def test_a_dinucleotide_too_rare_to_judge_is_absent_from_the_signature():
    # Two bases leave the ten dinucleotides involving G or T unassessable.
    spectra = {k: count("ACAC" * 20, k, DNA) for k in (1, 2)}
    signed = signature(spectra)
    assert sorted(signed) == ["AA", "AC", "CA", "CC"]


def test_gc_content_is_the_share_of_counted_residues_that_are_g_or_c():
    assert gc_content(count("AACCGGTT", 1, DNA)) == pytest.approx(0.5)
    assert gc_content(count("GGCC", 1, DNA)) == pytest.approx(1.0)


def test_gc_content_is_none_for_protein_and_for_nothing_counted():
    assert gc_content(count("ACDEFG", 1, PROTEIN)) is None
    assert gc_content(count("NN", 1, DNA)) is None


def test_the_markov_model_refuses_a_single_residue_word_by_name():
    # The refusal has to name the real objection: a one-residue word has no
    # prefix and no suffix, which is not the same as a missing spectrum.
    spectra = {length: count("ACGTACGTAC", length, DNA) for length in (1, 2)}
    with pytest.raises(ValueError, match="two or more residues"):
        table(spectra, 1, "markov")
