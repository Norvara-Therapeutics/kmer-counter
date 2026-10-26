"""What the four complexity measures promise, in closed form.

Every expected value here is derived rather than observed: the entropy of n
equally abundant words is log2(n), the DUST score of a homopolymer is
(L - 2) / 2 because its (L - 2) identical triplets make (L - 2)(L - 3) / 2
pairs over (L - 3) positions, and the linguistic complexity of a sequence that
uses every word available to it is one. The file protects those identities,
the directions the two low-complexity measures run in, and the ceilings the
normalised figures are read against.
"""

from math import log, log2, pi

import pytest

from kmer_counter.alphabet import DNA, PROTEIN, Alphabet
from kmer_counter.complexity import (
    attainable_entropy,
    DUST_THRESHOLD,
    DUST_WINDOW,
    SEG_THRESHOLD,
    SEG_WINDOW,
    compositional_complexity,
    default_threshold,
    default_window,
    entropy_of,
    expected_distinct,
    is_low_complexity,
    linguistic_complexity,
    max_entropy,
    max_triplet_score,
    residue_entropy,
    saturation_ratio,
    score_name,
    shannon_entropy,
    summarise,
    triplet_score,
    window_score,
)
from kmer_counter.spectrum import count

#: A de Bruijn sequence of order three over ACGT, 66 bases long: each of the
#: 64 triplets occurs exactly once along it, which makes it the most
#: non-repetitive DNA that exists at this word length. The property is
#: asserted below rather than taken on trust.
DE_BRUIJN_3 = "AAATTTGTTCTTATGGTGCTGATCGTCCTCATAGTACTAAGGGCGGAGCCGCAGACGAACCCACAA"


def test_the_entropy_of_a_single_word_is_zero_and_never_negative_zero():
    # One word carries no information however abundant it is; -0.0 would print
    # as '-0.00' and read as a defect, so the sign is part of the promise.
    result = shannon_entropy([7])
    assert result == 0.0
    assert str(result) == "0.0"


def test_the_entropy_of_n_equally_abundant_words_is_log2_n():
    assert shannon_entropy([3, 3, 3, 3]) == pytest.approx(2.0)
    assert shannon_entropy([5, 5]) == pytest.approx(1.0)
    assert shannon_entropy([1] * 8) == pytest.approx(3.0)


def test_the_entropy_of_an_uneven_case_is_the_hand_computed_sum():
    # Counts 1, 1, 2 over four windows are shares 1/4, 1/4, 1/2, so the
    # entropy is 2 * (1/4 * 2) + (1/2 * 1) = 1.5 bits.
    assert shannon_entropy([1, 1, 2]) == pytest.approx(1.5)


def test_counts_of_zero_contribute_nothing_to_the_entropy():
    assert shannon_entropy([4, 0, 4]) == pytest.approx(shannon_entropy([4, 4]))


def test_the_entropy_of_nothing_counted_is_zero():
    assert shannon_entropy([]) == 0.0
    assert shannon_entropy([0, 0]) == 0.0


def test_the_entropy_ceiling_of_no_words_is_zero():
    assert max_entropy(0) == 0.0
    assert max_entropy(16) == pytest.approx(4.0)


def test_entropy_of_reads_a_spectrum_against_its_own_ceiling():
    # AACCGGTT holds each base twice in eight single-residue windows, so the
    # entropy is log2(4) = 2 bits and that is also the ceiling: the sequence
    # uses the whole word space equally.
    measured = entropy_of(count("AACCGGTT", 1, DNA))
    assert measured.k == 1
    assert measured.bits == pytest.approx(2.0)
    assert measured.ceiling == pytest.approx(2.0)
    assert measured.distinct == 4
    assert measured.observable == 4
    assert measured.windows == 8
    assert measured.space == 4
    assert measured.efficiency == pytest.approx(1.0)
    assert measured.length_limited is False


def test_a_sequence_with_fewer_windows_than_words_is_graded_against_an_easier_ceiling():
    # ACGT offers three dinucleotide windows against a space of sixteen, so
    # the ceiling is log2(3) and not log2(16); the three distinct words each
    # occur once, so the sequence reaches it.
    measured = entropy_of(count("ACGT", 2, DNA))
    assert measured.observable == 3
    assert measured.space == 16
    assert measured.ceiling == pytest.approx(log2(3))
    assert measured.bits == pytest.approx(log2(3))
    assert measured.efficiency == pytest.approx(1.0)
    assert measured.length_limited is True


def test_efficiency_is_none_when_the_sequence_had_no_ceiling():
    measured = entropy_of(count("NN", 1, DNA))
    assert measured.windows == 0
    assert measured.ceiling == 0.0
    assert measured.efficiency is None


def test_summarise_measures_every_spectrum_keyed_by_word_length():
    spectra = {2: count("ACGT", 2, DNA), 1: count("ACGT", 1, DNA)}
    measured = summarise(spectra)
    assert list(measured) == [1, 2]
    assert measured[1].bits == pytest.approx(2.0)


def test_linguistic_complexity_is_one_when_every_observable_word_occurs():
    # ACGT uses all four bases and all three of the dinucleotides its three
    # windows allow, so both factors are one and so is their product.
    spectra = [count("ACGT", k, DNA) for k in (1, 2)]
    assert linguistic_complexity(spectra) == pytest.approx(1.0)


def test_linguistic_complexity_of_a_homopolymer_is_the_product_of_both_shortfalls():
    # AAAA shows 1 of the 4 observable bases and 1 of the 3 observable
    # dinucleotides, so the complexity is (1/4) * (1/3) = 1/12.
    spectra = [count("AAAA", k, DNA) for k in (1, 2)]
    assert linguistic_complexity(spectra) == pytest.approx(1 / 12)


def test_linguistic_complexity_refuses_word_lengths_that_are_not_consecutive_from_one():
    spectra = [count("ACGT", k, DNA) for k in (1, 3)]
    with pytest.raises(ValueError, match="consecutive word lengths from 1"):
        linguistic_complexity(spectra)


def test_a_word_length_with_no_countable_window_is_left_out_of_the_product():
    # ACN has no countable trinucleotide window at all, since no window of
    # three avoids the N. That length is skipped, not multiplied in as zero,
    # so the complexity stays at the 1.0 the two shorter lengths earn.
    spectra = [count("ACN", k, DNA) for k in (1, 2, 3)]
    assert spectra[2].windows == 0
    assert linguistic_complexity(spectra) == pytest.approx(1.0)


def test_linguistic_complexity_is_none_when_no_word_length_was_countable():
    assert linguistic_complexity([count("N", 1, DNA)]) is None


def test_the_dust_score_of_a_homopolymer_is_the_window_maximum():
    # A homopolymer of L residues holds L - 2 identical triplets, which make
    # (L - 2)(L - 3) / 2 pairs over L - 3 positions, so the score is
    # (L - 2) / 2: 31 in the conventional window of 64.
    assert triplet_score("A" * DUST_WINDOW) == pytest.approx(31.0)
    assert max_triplet_score(DUST_WINDOW) == pytest.approx(31.0)
    for length in (4, 10, 64):
        assert triplet_score("C" * length) == pytest.approx(max_triplet_score(length))


def test_the_dust_score_of_a_window_shorter_than_four_residues_is_zero():
    # Three residues hold one triplet, which cannot pair with anything.
    assert triplet_score("AAA") == 0.0
    assert triplet_score("A") == 0.0
    assert triplet_score("") == 0.0
    assert max_triplet_score(3) == 0.0


def test_the_dust_score_is_zero_when_every_triplet_in_the_window_is_distinct():
    assert len({DE_BRUIJN_3[i: i + 3] for i in range(len(DE_BRUIJN_3) - 2)}) == 64
    window = DE_BRUIJN_3[:DUST_WINDOW]
    assert len({window[i: i + 3] for i in range(len(window) - 2)}) == DUST_WINDOW - 2
    assert triplet_score(window) == 0.0
    assert triplet_score("ACGT") == 0.0


def test_the_dust_score_of_a_tetranucleotide_repeat_is_its_counted_pairs():
    # ACGT repeated through a 64-base window holds 62 triplets: ACG and CGT
    # sixteen times each, GTA and TAC fifteen, so the pairs number
    # 2 * (16 * 15 / 2) + 2 * (15 * 14 / 2) = 450 over 61 positions.
    assert triplet_score("ACGT" * 16) == pytest.approx(450 / 61)


def test_residue_entropy_counts_every_character_unless_an_alphabet_is_given():
    # AANN is two characters twice each, so one bit, until DNA is passed and
    # the two N are dropped, leaving a homopolymer of zero bits.
    assert residue_entropy("AANN") == pytest.approx(1.0)
    assert residue_entropy("AANN", DNA) == 0.0
    assert residue_entropy("ACGT", DNA) == pytest.approx(2.0)


def test_compositional_complexity_is_zero_for_a_homopolymer():
    # One composition spells one sequence, and log of one is zero.
    assert compositional_complexity("A" * SEG_WINDOW, PROTEIN) == 0.0
    assert compositional_complexity("T" * 20, DNA) == 0.0


def test_compositional_complexity_is_the_logged_multiplicity_of_the_composition():
    # ACGT has 4! / (1!)^4 = 24 rearrangements; logged to the alphabet size
    # and divided by the window length that is log(24) / (4 * log(4)).
    assert compositional_complexity("ACGT", DNA) == pytest.approx(
        log(24) / (4 * log(4))
    )
    # AACC has 4! / (2! 2!) = 6 of them.
    assert compositional_complexity("AACC", DNA) == pytest.approx(
        log(6) / (4 * log(4))
    )


def test_compositional_complexity_approaches_one_for_an_equal_composition():
    # Stirling on L!/((L/N)!)^N gives K1 = 1 + [log(2 pi L) / 2
    # - (N / 2) log(2 pi L / N)] / (L log N); for N = 20 residues fifty times
    # each that is 0.9823, just under the one the measure is normalised to.
    length, size = 1000, PROTEIN.size
    window = "".join(residue * (length // size) for residue in PROTEIN.core)
    stirling = 1 + (
        log(2 * pi * length) / 2 - (size / 2) * log(2 * pi * length / size)
    ) / (length * log(size))
    assert compositional_complexity(window, PROTEIN) == pytest.approx(
        stirling, abs=1e-4
    )
    assert compositional_complexity(window, PROTEIN) < 1.0


def test_compositional_complexity_ignores_residues_the_alphabet_does_not_count():
    assert compositional_complexity("ACGTNN", DNA) == pytest.approx(
        compositional_complexity("ACGT", DNA)
    )


def test_compositional_complexity_is_zero_when_the_alphabet_cannot_spell_a_choice():
    single = Alphabet(
        name="single",
        label="single",
        kind="protein",
        core="A",
        ambiguity={},
        rare={},
    )
    assert compositional_complexity("AAAA", single) == 0.0
    assert compositional_complexity("", DNA) == 0.0


def test_low_complexity_runs_in_opposite_directions_for_the_two_alphabets():
    # A high DUST score is repetitive; a low Wootton-Federhen complexity is.
    assert is_low_complexity(25.0, DNA) is True
    assert is_low_complexity(25.0, PROTEIN) is False
    assert is_low_complexity(1.0, PROTEIN) is True
    assert is_low_complexity(1.0, DNA) is False


def test_the_conventional_thresholds_are_met_at_their_own_value():
    assert is_low_complexity(DUST_THRESHOLD, DNA) is True
    assert is_low_complexity(DUST_THRESHOLD - 0.1, DNA) is False
    assert is_low_complexity(SEG_THRESHOLD, PROTEIN) is True
    assert is_low_complexity(SEG_THRESHOLD + 0.1, PROTEIN) is False


def test_a_given_threshold_replaces_the_conventional_one():
    assert is_low_complexity(10.0, DNA, threshold=5.0) is True
    assert is_low_complexity(10.0, DNA, threshold=15.0) is False
    assert is_low_complexity(3.0, PROTEIN, threshold=4.0) is True


def test_the_conventional_window_and_threshold_follow_the_alphabet():
    assert default_window(DNA) == DUST_WINDOW == 64
    assert default_window(PROTEIN) == SEG_WINDOW == 12
    assert default_threshold(DNA) == DUST_THRESHOLD == 20.0
    assert default_threshold(PROTEIN) == SEG_THRESHOLD == 2.2


def test_expected_distinct_words_is_the_occupancy_expectation():
    # Two draws from four words leave a word unseen with probability
    # (3/4)^2, so 4 * (1 - 9/16) = 1.75 words are expected.
    assert expected_distinct(2, 4) == pytest.approx(1.75)
    assert expected_distinct(0, 4) == 0.0
    assert expected_distinct(10, 0) == 0.0


def test_the_saturation_ratio_compares_distinct_words_with_that_expectation():
    # ACGT draws four single-residue windows from a space of four, where
    # 4 * (1 - (3/4)^4) = 175/64 words are expected, so the ratio is
    # 4 / (175/64) = 256/175.
    assert saturation_ratio(count("ACGT", 1, DNA)) == pytest.approx(256 / 175)
    # A homopolymer shows one word where the same expectation stands.
    assert saturation_ratio(count("AAAA", 1, DNA)) == pytest.approx(1 / (175 / 64))


def test_the_saturation_ratio_is_none_without_a_countable_window():
    assert saturation_ratio(count("NN", 1, DNA)) is None


def test_the_window_score_is_the_alphabet_s_own_measure_under_its_own_name():
    window = "ACGT" * 16
    assert window_score(window, DNA) == pytest.approx(triplet_score(window))
    assert window_score("QQQQQQQQQQQQ", PROTEIN) == 0.0
    assert window_score("ACDEFGHIKLMN", PROTEIN) == pytest.approx(
        residue_entropy("ACDEFGHIKLMN", PROTEIN)
    )
    assert score_name(DNA) == "dust"
    assert score_name(PROTEIN) == "complexity"


def test_the_ceiling_allows_for_windows_being_whole():
    # Five windows over four words are at their most even as 2, 1, 1, 1. A
    # ceiling of log2(4) would call the most even sequence possible 96%
    # efficient; the attainable maximum calls it what it is.
    uneven = -((2 / 5) * log2(2 / 5) + 3 * (1 / 5) * log2(1 / 5))
    assert attainable_entropy(5, 4) == pytest.approx(uneven)
    assert attainable_entropy(5, 4) < log2(4)

    measured = entropy_of(count("ACGTA", 1, DNA))
    assert measured.bits == pytest.approx(uneven)
    assert measured.ceiling == pytest.approx(uneven)
    assert measured.efficiency == pytest.approx(1.0)


def test_the_ceiling_is_the_round_number_when_the_windows_divide():
    # Sixty-four windows over four words divide exactly, so the ceiling is
    # log2(4) and a sequence can reach it.
    assert attainable_entropy(64, 4) == pytest.approx(log2(4))
    assert attainable_entropy(4, 4) == pytest.approx(log2(4))


def test_fewer_windows_than_words_puts_the_ceiling_at_the_window_count():
    assert attainable_entropy(10, 256) == pytest.approx(log2(10))
    assert attainable_entropy(1, 256) == 0.0


def test_an_empty_count_has_no_attainable_entropy():
    assert attainable_entropy(0, 256) == 0.0
    assert attainable_entropy(10, 0) == 0.0


def test_the_attainable_maximum_beats_every_other_allocation():
    # Entropy is Schur-concave, so the even integer allocation is the most
    # entropy 11 windows over 4 words can carry. Checked against every
    # allocation of 11 into 4 parts.
    def allocations(total, parts):
        if parts == 1:
            yield [total]
            return
        for first in range(total + 1):
            for rest in allocations(total - first, parts - 1):
                yield [first, *rest]

    best = max(shannon_entropy(counts) for counts in allocations(11, 4))
    assert attainable_entropy(11, 4) == pytest.approx(best)
