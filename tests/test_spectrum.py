"""What ``kmer_counter.spectrum`` must keep true.

A spectrum is the count of every word of length k over a window that advances
one residue at a time, and the numbers it reports are only trustworthy if two
accounting rules hold exactly: a sequence of L residues offers L - k + 1
windows, and every one of those windows is either counted or counted as
skipped. These tests pin that arithmetic, the segmentation that produces it
(no window may cross a residue the alphabet does not count), the canonical
collapse and the size of the word space it leaves, and the derived figures -
census, singletons, saturation, coverage, observable - that the renderers
read.

Expected values are hand-computed from short strings wherever possible: the
spectrum of 'AAAA' at k=2 is three copies of 'AA' because the string has
4 - 2 + 1 = 3 windows and every one of them reads AA. Values taken from the
bundled examples are pinned only as arithmetic on the record length, which can
be rechecked with `grep -v '>' examples/<file> | tr -d '\\n' | wc -c`.
"""

from itertools import groupby

import pytest

from kmer_counter import spectrum as sp
from kmer_counter.alphabet import DNA, PROTEIN, RNA, reverse_complement
from kmer_counter.reader import read_one


# --- word length bounds -------------------------------------------------


def test_the_supported_word_lengths_are_one_to_eight():
    assert sp.K_MIN == 1
    assert sp.K_MAX == 8


def test_check_k_returns_every_length_in_range():
    for k in range(1, 9):
        assert sp.check_k(k) == k


def test_check_k_refuses_a_length_outside_the_range():
    with pytest.raises(ValueError, match="k must be between 1 and 8, got 0"):
        sp.check_k(0)
    with pytest.raises(ValueError, match="k must be between 1 and 8, got 9"):
        sp.check_k(9)
    with pytest.raises(ValueError, match="k must be between 1 and 8, got -2"):
        sp.check_k(-2)


def test_counting_refuses_a_word_length_outside_the_range():
    with pytest.raises(ValueError, match="k must be between 1 and 8"):
        sp.count("ACGTACGTA", 9, DNA)
    with pytest.raises(ValueError, match="k must be between 1 and 8"):
        sp.count_all(["ACGT"], 0, DNA)


# --- window arithmetic ---------------------------------------------------


def test_a_sequence_offers_one_window_per_residue_less_k_minus_one():
    assert sp.window_count(10, 1) == 10
    assert sp.window_count(10, 4) == 7
    assert sp.window_count(4, 4) == 1


def test_a_sequence_shorter_than_the_word_offers_no_windows():
    # Negative window counts would corrupt every share computed from them.
    assert sp.window_count(3, 4) == 0
    assert sp.window_count(0, 1) == 0


def test_the_counted_windows_sum_to_the_counts():
    spectrum = sp.count("ACGTACGTACGTA", 3, DNA)
    assert sum(spectrum.counts.values()) == spectrum.windows
    assert spectrum.windows == 13 - 3 + 1


def test_every_window_is_either_counted_or_counted_as_skipped():
    # The accounting rule: counted + skipped == L - k + 1, whatever the
    # sequence holds.
    for sequence in ("ACGTACGT", "ACGTNNACGT", "NNNNNN", "AC-GT.ACGT", ""):
        for k in (1, 3, 5):
            spectrum = sp.count(sequence, k, DNA)
            assert spectrum.windows + spectrum.skipped == sp.window_count(len(sequence), k)


def test_a_sequence_shorter_than_the_word_skips_nothing():
    spectrum = sp.count("AC", 4, DNA)
    assert spectrum.windows == 0
    assert spectrum.skipped == 0
    assert spectrum.counts == {}
    assert spectrum.length == 2
    assert spectrum.sequences == 1


def test_an_empty_sequence_contributes_no_sequence_at_all():
    spectrum = sp.count("", 3, DNA)
    assert spectrum.sequences == 0
    assert spectrum.length == 0
    assert spectrum.windows == 0
    assert spectrum.skipped == 0
    assert spectrum.skipped_share == pytest.approx(0.0)
    assert spectrum.saturation is None
    assert spectrum.coverage is None
    assert spectrum.frequency("AAA") == pytest.approx(0.0)


# --- segmentation and skipped windows ------------------------------------


def test_segments_are_the_maximal_runs_of_countable_residues():
    assert list(sp.segments("ACGTNNACGT", DNA)) == [(0, "ACGT"), (6, "ACGT")]


def test_segment_starts_are_zero_based_offsets_into_the_sequence():
    assert list(sp.segments("NNACGTN", DNA)) == [(2, "ACGT")]
    assert list(sp.segments("ACGT", DNA)) == [(0, "ACGT")]


def test_gaps_break_a_segment_like_any_uncountable_residue():
    assert list(sp.segments("AC-GT", DNA)) == [(0, "AC"), (3, "GT")]
    assert list(sp.segments("AC.GT~A", DNA)) == [(0, "AC"), (3, "GT"), (6, "A")]


def test_the_stop_character_breaks_a_protein_segment():
    # A stop is not a residue, so no word may read across it.
    assert list(sp.segments("MAA*KL", PROTEIN)) == [(0, "MAA"), (4, "KL")]


def test_an_ambiguity_code_breaks_a_protein_segment():
    assert list(sp.segments("MAXKL", PROTEIN)) == [(0, "MA"), (3, "KL")]


def test_thymine_breaks_an_rna_segment_and_uracil_a_dna_one():
    assert list(sp.segments("ACGUTACGU", RNA)) == [(0, "ACGU"), (5, "ACGU")]
    assert list(sp.segments("ACGTUACGT", DNA)) == [(0, "ACGT"), (5, "ACGT")]


def test_a_sequence_with_nothing_countable_has_no_segments():
    assert list(sp.segments("NNNN", DNA)) == []
    assert list(sp.segments("", DNA)) == []
    assert list(sp.segments("----", DNA)) == []


def test_a_single_ambiguity_code_skips_exactly_the_windows_that_touch_it():
    # ACGTNACGT at k=4 has 9 - 4 + 1 = 6 windows; the N sits at offset 4, so
    # the four windows starting at offsets 1, 2, 3 and 4 all contain it and
    # only the windows at 0 and 5 are counted.
    spectrum = sp.count("ACGTNACGT", 4, DNA)
    assert spectrum.windows == 2
    assert spectrum.skipped == 4
    assert spectrum.counts == {"ACGT": 2}
    assert spectrum.skipped_share == pytest.approx(4 / 6)


def test_segments_too_short_for_the_word_contribute_nothing_but_skips():
    # ACGTNAC at k=4 has 4 windows; the trailing AC is too short to hold one,
    # so only the window at offset 0 is counted.
    spectrum = sp.count("ACGTNAC", 4, DNA)
    assert spectrum.windows == 1
    assert spectrum.skipped == 3


def test_a_sequence_that_is_all_n_has_a_spectrum_of_nothing():
    # Ten Ns at k=3 offer eight windows and not one of them is countable, so
    # every window is skipped and no share of the word space can be reported.
    spectrum = sp.count("N" * 10, 3, DNA)
    assert spectrum.counts == {}
    assert spectrum.distinct == 0
    assert spectrum.windows == 0
    assert spectrum.skipped == 8
    assert spectrum.length == 10
    assert spectrum.sequences == 1
    assert spectrum.skipped_share == pytest.approx(1.0)
    assert spectrum.observable == 0
    assert spectrum.saturation is None
    assert spectrum.coverage is None


def test_a_sequence_with_nothing_skipped_reports_a_zero_share():
    spectrum = sp.count("ACGTACGT", 3, DNA)
    assert spectrum.skipped == 0
    assert spectrum.skipped_share == pytest.approx(0.0)


# --- hand-checked spectra -------------------------------------------------


def test_a_homopolymer_holds_one_word_in_every_window():
    # 'AAAA' at k=2 has 4 - 2 + 1 = 3 windows and every one of them reads AA.
    spectrum = sp.count("AAAA", 2, DNA)
    assert spectrum.counts == {"AA": 3}
    assert spectrum.windows == 3
    assert spectrum.skipped == 0
    assert spectrum.distinct == 1
    assert spectrum.frequency("AA") == pytest.approx(1.0)
    assert spectrum.frequency("AC") == pytest.approx(0.0)


def test_a_short_string_is_counted_window_by_window():
    # AAAACC at k=2 reads AA, AA, AA, AC, CC over its five windows.
    spectrum = sp.count("AAAACC", 2, DNA)
    assert spectrum.counts == {"AA": 3, "AC": 1, "CC": 1}
    assert spectrum.windows == 5


def test_counting_single_residues_is_counting_the_composition():
    # At k=1 every residue is its own window, so the spectrum is the census.
    spectrum = sp.count("AACGTTTT", 1, DNA)
    assert spectrum.counts == {"A": 2, "C": 1, "G": 1, "T": 4}
    assert spectrum.windows == 8
    assert spectrum.frequency("T") == pytest.approx(0.5)


def test_a_protein_spectrum_is_drawn_from_the_wider_word_space():
    # MAAD at k=2 reads MA, AA, AD; the word space is 20**2 = 400.
    spectrum = sp.count("MAAD", 2, PROTEIN)
    assert spectrum.counts == {"MA": 1, "AA": 1, "AD": 1}
    assert spectrum.possible == 400
    assert spectrum.coverage == pytest.approx(3 / 400)


# --- the canonical collapse -----------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6, 7, 8])
def test_canonical_counting_halves_the_word_space_bar_the_palindromes(k):
    # A word and its reverse complement are one word, so the space halves
    # except for words that are their own reverse complement. Those need a
    # self-complementary centre, which only an even-length word has: pairing
    # the first half freely and determining the second gives 4**(k/2) of
    # them. Hence (4**k + 4**(k//2)) / 2 for even k and 4**k / 2 for odd.
    palindromes = 4 ** (k // 2) if k % 2 == 0 else 0
    assert sp.possible_words(DNA, k, canonical=True) == (4**k + palindromes) // 2
    assert sp.possible_words(DNA, k, canonical=False) == 4**k


def test_the_canonical_word_space_at_small_k_is_as_enumerated():
    # k=1: A/T and C/G leave 2. k=2: 16 words, the 4 palindromes AT, TA, CG,
    # GC pair with themselves and the other 12 pair up, giving 4 + 6 = 10.
    assert sp.possible_words(DNA, 1, canonical=True) == 2
    assert sp.possible_words(DNA, 2, canonical=True) == 10
    assert sp.possible_words(DNA, 3, canonical=True) == 32
    assert sp.possible_words(DNA, 4, canonical=True) == 136


def test_the_word_space_is_the_same_size_for_rna_as_for_dna():
    assert sp.possible_words(RNA, 4, canonical=True) == sp.possible_words(DNA, 4, True)
    assert sp.possible_words(RNA, 3) == 64


def test_the_protein_word_space_grows_by_twenty_each_residue():
    assert sp.possible_words(PROTEIN, 1) == 20
    assert sp.possible_words(PROTEIN, 3) == 8000


def test_the_word_space_is_refused_for_a_word_length_out_of_range():
    with pytest.raises(ValueError, match="k must be between 1 and 8"):
        sp.possible_words(DNA, 9)


def test_canonical_counting_is_refused_without_a_complement():
    with pytest.raises(ValueError, match="protein has no complement, so has no canonical form"):
        sp.possible_words(PROTEIN, 3, canonical=True)
    with pytest.raises(ValueError, match="protein has no complement, so has no canonical form"):
        sp.count("MAAD", 2, PROTEIN, canonical=True)
    with pytest.raises(ValueError, match="protein has no complement, so has no canonical form"):
        sp.count_all(["MAAD"], 2, PROTEIN, canonical=True)


def test_the_canonical_word_is_the_smaller_of_the_two_strands():
    # AC and its reverse complement GT: 'AC' sorts first, so AC is canonical
    # and GT is counted under it. CG is its own reverse complement.
    assert sp.canonical_word("AC", DNA) == "AC"
    assert sp.canonical_word("GT", DNA) == "AC"
    assert sp.canonical_word("CG", DNA) == "CG"
    assert sp.canonical_word("TTTT", DNA) == "AAAA"


def test_canonical_counting_folds_a_word_into_its_reverse_complement():
    # ACGT at k=2 reads AC, CG, GT. AC and GT are the two strands of one
    # word, so they merge under AC; CG is its own reverse complement.
    spectrum = sp.count("ACGT", 2, DNA, canonical=True)
    assert spectrum.counts == {"AC": 2, "CG": 1}
    assert spectrum.canonical
    assert spectrum.windows == 3
    assert spectrum.distinct == 2
    assert spectrum.possible == 10


def test_the_canonical_collapse_keeps_the_window_total():
    # Folding renames words; it never loses or invents a window.
    plain = sp.count("ACGTTGCAAGGCTTAC", 3, DNA)
    folded = sp.count("ACGTTGCAAGGCTTAC", 3, DNA, canonical=True)
    assert folded.windows == plain.windows
    assert sum(folded.counts.values()) == sum(plain.counts.values()) == plain.windows
    assert folded.distinct <= plain.distinct


def test_a_sequence_and_its_reverse_complement_share_a_canonical_spectrum():
    # This is the point of canonical counting: which strand was sequenced
    # must not change the answer.
    sequence = "ACGTTGCAAGGCTTACGGATCC"
    forward = sp.count(sequence, 4, DNA, canonical=True)
    reverse = sp.count(reverse_complement(sequence, DNA), 4, DNA, canonical=True)
    assert forward.counts == reverse.counts
    assert forward.windows == reverse.windows


# --- derived figures ------------------------------------------------------


def test_the_census_counts_how_many_words_occur_how_often():
    # AAAACC gives AA three times and AC and CC once each, so two words occur
    # once and one word occurs three times.
    spectrum = sp.count("AAAACC", 2, DNA)
    assert spectrum.census() == {1: 2, 3: 1}
    assert list(spectrum.census()) == [1, 3]


def test_the_census_of_a_spectrum_with_no_words_is_empty():
    assert sp.count("NNNN", 2, DNA).census() == {}


def test_singletons_are_the_words_that_occur_exactly_once():
    spectrum = sp.count("AAAACC", 2, DNA)
    assert spectrum.singletons() == 2


def test_repeated_windows_are_the_windows_holding_a_repeated_word():
    # Only AA repeats, and it fills three of the five windows.
    spectrum = sp.count("AAAACC", 2, DNA)
    assert spectrum.repeated_windows() == 3


def test_an_unrepetitive_sequence_has_no_repeated_windows():
    spectrum = sp.count("ACGT", 2, DNA)
    assert spectrum.repeated_windows() == 0
    assert spectrum.singletons() == 3


def test_most_common_breaks_ties_alphabetically_for_stability():
    # CC, AC and AA all tie at one occurrence after AA's three; the order
    # must not depend on insertion order, so it is alphabetical.
    spectrum = sp.count("AAAACCAG", 2, DNA)
    assert spectrum.counts == {"AA": 3, "AC": 1, "CC": 1, "CA": 1, "AG": 1}
    assert spectrum.most_common() == [
        ("AA", 3),
        ("AC", 1),
        ("AG", 1),
        ("CA", 1),
        ("CC", 1),
    ]


def test_most_common_returns_at_most_the_number_asked_for():
    spectrum = sp.count("AAAACCAG", 2, DNA)
    assert spectrum.most_common(2) == [("AA", 3), ("AC", 1)]
    assert spectrum.most_common(0) == []
    assert len(spectrum.most_common(100)) == spectrum.distinct


def test_sorted_items_is_most_common_over_every_word():
    spectrum = sp.count("AAAACCAG", 2, DNA)
    assert spectrum.sorted_items() == spectrum.most_common(spectrum.distinct)
    assert len(spectrum.sorted_items()) == spectrum.distinct


def test_observable_is_the_smaller_of_the_word_space_and_the_windows():
    # A 5-residue sequence has two 4-mer windows and so cannot show more than
    # two of the 256 possible words however complex it is.
    short = sp.count("ACGTA", 4, DNA)
    assert short.windows == 2
    assert short.possible == 256
    assert short.observable == 2
    # At k=1 the four windows of ACGT reach the whole four-letter space.
    full = sp.count("ACGT", 1, DNA)
    assert full.possible == 4
    assert full.observable == 4


def test_saturation_is_one_when_no_window_repeats_another():
    spectrum = sp.count("ACGTA", 4, DNA)
    assert spectrum.distinct == 2
    assert spectrum.saturation == pytest.approx(1.0)


def test_saturation_falls_as_the_sequence_repeats_itself():
    # AAAA at k=1 shows one of the four words its four windows could have
    # shown, so saturation is 1/4.
    spectrum = sp.count("AAAA", 1, DNA)
    assert spectrum.saturation == pytest.approx(0.25)
    assert spectrum.coverage == pytest.approx(0.25)


def test_coverage_is_the_share_of_the_whole_word_space():
    # Two distinct 4-mers out of 256 possible is 0.78%, however unrepetitive
    # the sequence is: coverage says more about length than complexity.
    spectrum = sp.count("ACGTA", 4, DNA)
    assert spectrum.coverage == pytest.approx(2 / 256)
    assert spectrum.saturation == pytest.approx(1.0)


def test_distinct_never_exceeds_what_the_windows_could_show():
    for sequence in ("ACGTACGTTGCA", "AAAAAAAA", "ACGTNNNNACGT"):
        for k in (1, 2, 4):
            spectrum = sp.count(sequence, k, DNA)
            assert spectrum.distinct <= spectrum.observable


def test_frequency_is_the_share_of_counted_windows_not_of_all_windows():
    # ACGTNACGT at k=4 counts two windows, both ACGT, so ACGT fills all of
    # the counted windows even though four of the six were skipped.
    spectrum = sp.count("ACGTNACGT", 4, DNA)
    assert spectrum.frequency("ACGT") == pytest.approx(1.0)
    assert spectrum.frequency("CGTA") == pytest.approx(0.0)


# --- merging and counting several sequences -------------------------------


def test_counting_one_sequence_through_count_all_matches_counting_it_alone():
    single = sp.count("ACGTTGCAAGGC", 3, DNA)
    together = sp.count_all(["ACGTTGCAAGGC"], 3, DNA)
    assert together == single


def test_merging_one_spectrum_returns_what_it_was_given():
    single = sp.count("ACGTTGCAAGGC", 3, DNA)
    assert sp.merge([single]) == single


def test_merge_and_count_all_agree_on_the_same_sequences():
    sequences = ["ACGTA", "TTTT", "NNNN", ""]
    merged = sp.merge([sp.count(sequence, 2, DNA) for sequence in sequences])
    together = sp.count_all(sequences, 2, DNA)
    assert merged == together


def test_merging_sums_the_windows_rather_than_spanning_the_records():
    # No word straddles two records, so ACGTA (4 windows at k=2) and TTTT
    # (3 windows) make 7, not the 8 the concatenation would give.
    merged = sp.merge([sp.count("ACGTA", 2, DNA), sp.count("TTTT", 2, DNA)])
    assert merged.windows == 7
    assert merged.length == 9
    assert merged.sequences == 2
    assert merged.counts == {"AC": 1, "CG": 1, "GT": 1, "TA": 1, "TT": 3}
    assert sp.window_count(9, 2) == 8


def test_merging_keeps_the_skipped_windows_of_every_spectrum():
    merged = sp.merge([sp.count("ACGTNACGT", 4, DNA), sp.count("NNNNNNN", 4, DNA)])
    assert merged.skipped == 4 + 4
    assert merged.windows == 2


def test_counting_several_sequences_counts_each_as_one_sequence():
    spectrum = sp.count_all(["ACGT", "TTTT", "GG"], 2, DNA)
    assert spectrum.sequences == 3
    assert spectrum.length == 10


def test_count_all_over_nothing_is_an_empty_spectrum():
    spectrum = sp.count_all([], 3, DNA)
    assert spectrum.counts == {}
    assert spectrum.sequences == 0
    assert spectrum.windows == 0
    assert spectrum.length == 0


def test_merging_nothing_is_refused():
    with pytest.raises(ValueError, match="no spectra to merge"):
        sp.merge([])


def test_merging_different_word_lengths_is_refused():
    with pytest.raises(ValueError, match="cannot merge k=2 with k=3"):
        sp.merge([sp.count("ACGT", 2, DNA), sp.count("ACGT", 3, DNA)])


def test_merging_different_alphabets_is_refused():
    with pytest.raises(ValueError, match="cannot merge DNA with RNA"):
        sp.merge([sp.count("ACGT", 2, DNA), sp.count("ACGU", 2, RNA)])
    with pytest.raises(ValueError, match="cannot merge DNA with protein"):
        sp.merge([sp.count("ACGT", 2, DNA), sp.count("MAAD", 2, PROTEIN)])


def test_merging_canonical_with_plain_counting_is_refused():
    with pytest.raises(ValueError, match="cannot merge canonical and non-canonical spectra"):
        sp.merge([sp.count("ACGT", 2, DNA), sp.count("ACGT", 2, DNA, canonical=True)])


def test_a_mismatch_in_a_later_spectrum_is_still_caught():
    # Every spectrum is checked against the first, not just the second.
    with pytest.raises(ValueError, match="cannot merge k=2 with k=4"):
        sp.merge([sp.count("ACGT", 2, DNA), sp.count("ACGT", 2, DNA), sp.count("ACGT", 4, DNA)])


# --- positions ------------------------------------------------------------


def test_positions_are_one_based_starts():
    # ACGTACGT holds ACG at offsets 0 and 4, reported as 1 and 5.
    assert sp.positions("ACGTACGT", "ACG", DNA) == [1, 5]
    assert sp.positions("AAAA", "AA", DNA) == [1, 2, 3]


def test_a_word_that_does_not_occur_has_no_positions():
    assert sp.positions("ACGTACGT", "GGGG", DNA) == []
    assert sp.positions("AC", "ACGT", DNA) == []


def test_positions_match_the_other_strand_under_canonical_counting():
    # CGT was counted under its canonical form ACG, so both strands must be
    # reported: ACGTACGT holds ACG at 1 and 5 and CGT at 2 and 6.
    assert sp.positions("ACGTACGT", "CGT", DNA, canonical=True) == [1, 2, 5, 6]
    assert sp.positions("ACGTACGT", "CGT", DNA, canonical=False) == [2, 6]


def test_a_palindromic_word_is_not_reported_twice_under_canonical_counting():
    # CG is its own reverse complement, so matching both strands finds the
    # same windows once.
    assert sp.positions("ACGT", "CG", DNA, canonical=True) == [2]


def test_positions_ignore_the_strand_for_an_alphabet_without_one():
    # Protein has no complement, so canonical matching has nothing to add
    # rather than raising.
    assert sp.positions("MAADMA", "MA", PROTEIN, canonical=True) == [1, 5]


# --- the residue census ---------------------------------------------------


def test_the_residue_census_counts_every_character_across_sequences():
    assert sp.residue_census(["ACGT", "AANN"]) == {"A": 3, "C": 1, "G": 1, "T": 1, "N": 2}


def test_the_residue_census_of_nothing_is_empty():
    assert sp.residue_census([]) == {}
    assert sp.residue_census(["", ""]) == {}


# --- the bundled examples -------------------------------------------------


def test_a_plain_acgt_genome_skips_no_window(examples):
    # phix174.fasta is 5,386 bases of ACGT with no ambiguity codes, so at
    # k=8 all 5,386 - 8 + 1 = 5,379 windows are counted.
    record = read_one(examples / "phix174.fasta")
    spectrum = sp.count(record.sequence, 8, DNA)
    assert spectrum.length == 5386
    assert spectrum.windows == 5379
    assert spectrum.skipped == 0
    assert sum(spectrum.counts.values()) == 5379
    assert spectrum.sequences == 1


def test_single_residue_counts_of_a_genome_sum_to_its_length(examples):
    # At k=1 every base is a window, so the counts are the base composition
    # and they must add up to the 4,679 bases of the AAV2 genome.
    record = read_one(examples / "aav2.fasta")
    spectrum = sp.count(record.sequence, 1, DNA)
    assert spectrum.windows == 4679
    assert sum(spectrum.counts.values()) == 4679
    assert set(spectrum.counts) == {"A", "C", "G", "T"}
    assert spectrum.distinct == 4
    assert spectrum.saturation == pytest.approx(1.0)


def test_canonical_counting_of_a_genome_keeps_its_window_total(examples):
    record = read_one(examples / "aav2.fasta")
    plain = sp.count(record.sequence, 5, DNA)
    folded = sp.count(record.sequence, 5, DNA, canonical=True)
    assert folded.windows == plain.windows == 4679 - 5 + 1
    assert sum(folded.counts.values()) == plain.windows
    # Odd k has no palindromes, so the canonical space is exactly half.
    assert folded.possible * 2 == plain.possible


def test_a_protein_spectrum_counts_every_window_of_a_clean_record(examples):
    # aav2_vp1.fasta is 735 residues of the twenty core residues, with no X
    # and no stop, so every one of its 734 dipeptide windows is counted.
    record = read_one(examples / "aav2_vp1.fasta")
    spectrum = sp.count(record.sequence, 2, PROTEIN)
    assert spectrum.length == 735
    assert spectrum.windows == 734
    assert spectrum.skipped == 0
    assert spectrum.possible == 400
    assert spectrum.distinct <= 400


def test_a_homopolymer_tract_dominates_the_spectrum_of_huntingtin(examples):
    # The polyglutamine tract is a run of Q of some length R, and a run of R
    # identical residues holds R - 4 + 1 windows of QQQQ, so QQQQ must be
    # counted at least that often and its positions must be consecutive.
    record = read_one(examples / "huntingtin.fasta")
    longest_q = max(
        sum(1 for _ in group)
        for residue, group in groupby(record.sequence)
        if residue == "Q"
    )
    spectrum = sp.count(record.sequence, 4, PROTEIN)
    assert longest_q >= 20
    assert spectrum.counts["QQQQ"] >= longest_q - 3
    found = sp.positions(record.sequence, "QQQQ", PROTEIN)
    assert len(found) == spectrum.counts["QQQQ"]
    assert found == sorted(found)
