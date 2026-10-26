"""What the sliding window promises, and when its calls cannot be read.

Three rules carry this file. The score of a window is the alphabet's own
measure taken over exactly that window, so a homopolymer window scores the
ceiling (window - 2) / 2 and a window in which no triplet repeats scores zero.
The regions called are a property of the sequence and not of the reporting, so
scanning at step 1 and step 7 must call the same intervals. And a threshold
that the window cannot reach turns the low-complexity call into a constant,
which has to be said in the output rather than left for the reader to notice.
"""

from math import log2
from pathlib import Path

import pytest

from kmer_counter.alphabet import DNA, PROTEIN
from kmer_counter.complexity import DUST_THRESHOLD, DUST_WINDOW, SEG_WINDOW, triplet_score
from kmer_counter.profile import (
    Interval,
    Window,
    iter_windows,
    merge_low,
    scan,
    score_ceiling,
    threshold_note,
    threshold_reachable,
)
from kmer_counter.reader import read_one

#: A de Bruijn sequence of order three over ACGT: 66 bases in which each of
#: the 64 triplets occurs exactly once, so no window of it repeats a triplet.
DE_BRUIJN_3 = "AAATTTGTTCTTATGGTGCTGATCGTCCTCATAGTACTAAGGGCGGAGCCGCAGACGAACCCACAA"

#: That sequence either side of a 200-base poly-A tract, which gives one
#: unambiguous low-complexity region with non-repetitive flanks. The tract
#: covers 1-based positions 67 to 266.
TRACT_START, TRACT_END = 67, 266
WITH_TRACT = DE_BRUIJN_3 + "A" * 200 + DE_BRUIJN_3


def _low(start: int, end: int, score: float) -> Window:
    """A window already called low, for testing the merge on its own."""
    return Window(
        start=start,
        end=end,
        score=score,
        entropy=0.0,
        gc=None,
        countable=end - start + 1,
        low=True,
    )


def test_a_homopolymer_window_scores_the_ceiling_and_is_called():
    # 64 identical bases hold 62 identical triplets: (62 * 61 / 2) pairs over
    # 61 positions is 31, the most a 64-base window can score.
    profile = scan("A" * DUST_WINDOW, DNA)
    assert profile.measure == "dust"
    assert profile.scored == 1
    assert profile.scores == [pytest.approx(31.0)]
    assert profile.ceiling == pytest.approx(31.0)
    assert profile.low_windows == 1
    assert list(profile.intervals) == [
        Interval(start=1, end=64, windows=1, peak=31.0, mean=31.0)
    ]
    assert profile.masked == 64
    assert profile.masked_share == pytest.approx(1.0)
    assert profile.qualification == ""


def test_a_window_in_which_no_triplet_repeats_scores_zero():
    # The de Bruijn sequence of order three holds every triplet once, so each
    # of its three 64-base windows holds 62 distinct triplets and no pairs.
    assert len({DE_BRUIJN_3[i: i + 3] for i in range(len(DE_BRUIJN_3) - 2)}) == 64
    profile = scan(DE_BRUIJN_3, DNA)
    assert profile.scored == len(DE_BRUIJN_3) - DUST_WINDOW + 1 == 3
    assert profile.scores == [0.0, 0.0, 0.0]
    assert profile.low_windows == 0
    assert list(profile.intervals) == []
    assert profile.masked_share == 0.0


def test_a_window_entropy_and_gc_are_taken_over_its_countable_residues():
    profile = scan("ACGT" * 16, DNA)
    window = profile.windows[0]
    assert window.countable == 64
    assert window.entropy == pytest.approx(2.0)
    assert window.gc == pytest.approx(0.5)
    assert window.score == pytest.approx(triplet_score("ACGT" * 16))


def test_a_triplet_spanning_an_uncounted_residue_is_not_scored():
    # 32 A, one N, 31 A: the triplets wholly inside the two A runs number
    # 30 and 29, so 59 identical triplets make 59 * 58 / 2 = 1711 pairs over
    # the window's 61 positions.
    sequence = "A" * 32 + "N" + "A" * 31
    profile = scan(sequence, DNA, window=DUST_WINDOW)
    window = profile.windows[0]
    assert window.countable == 63
    assert window.score == pytest.approx(1711 / 61)
    assert window.score < 31.0


def test_a_window_of_nothing_countable_scores_nothing():
    profile = scan("N" * DUST_WINDOW, DNA)
    window = profile.windows[0]
    assert window.countable == 0
    assert window.score == 0.0
    assert window.entropy == 0.0
    assert window.gc is None
    assert window.low is False
    # The whole-window helper, which has no alphabet to filter by, would have
    # read the same window as a homopolymer; the profile counts only the
    # triplets the alphabet can spell.
    assert triplet_score("N" * DUST_WINDOW) == pytest.approx(31.0)


def test_the_regions_called_do_not_depend_on_the_step():
    # Every window is scored whatever the step; the step only thins the
    # reported profile, so the intervals must be identical.
    fine = scan(WITH_TRACT, DNA, step=1)
    coarse = scan(WITH_TRACT, DNA, step=7)
    assert list(fine.intervals) == list(coarse.intervals)
    assert fine.masked == coarse.masked
    assert fine.scored > coarse.scored
    # One region, covering the whole tract and no more than a window beyond
    # it on either side.
    assert len(fine.intervals) == 1
    region = fine.intervals[0]
    assert region.start <= TRACT_START
    assert region.end >= TRACT_END
    assert region.start >= TRACT_START - DUST_WINDOW + 1
    assert region.end <= TRACT_END + DUST_WINDOW - 1
    assert region.peak == pytest.approx(31.0)


def test_the_regions_of_a_protein_do_not_depend_on_the_step_either(examples: Path):
    protein = read_one(examples / "huntingtin.fasta").sequence
    fine = scan(protein, PROTEIN, step=1)
    coarse = scan(protein, PROTEIN, step=7)
    assert list(fine.intervals) == list(coarse.intervals)
    # The polyQ and polyP tracts at the N-terminus are one called region.
    assert fine.intervals[0].start < 100
    assert fine.intervals[0].peak == 0.0


def test_the_step_thins_the_reported_windows_without_changing_them():
    fine = scan(WITH_TRACT, DNA, step=1)
    coarse = scan(WITH_TRACT, DNA, step=7)
    total = len(WITH_TRACT) - DUST_WINDOW + 1
    assert fine.scored == total
    assert coarse.scored == -(-total // 7)
    assert [window.start for window in coarse.windows] == [
        window.start for window in fine.windows[::7]
    ]
    assert coarse.step == 7


def test_merging_joins_windows_that_overlap():
    merged = merge_low([_low(1, 64, 25.0), _low(10, 73, 30.0)])
    assert merged == [Interval(start=1, end=73, windows=2, peak=30.0, mean=27.5)]


def test_merging_joins_windows_that_abut_with_no_gap():
    merged = merge_low([_low(1, 10, 25.0), _low(11, 20, 21.0)])
    assert merged == [Interval(start=1, end=20, windows=2, peak=25.0, mean=23.0)]


def test_a_gap_of_one_residue_makes_two_regions():
    # A window scoring well in between is evidence the sequence recovered.
    merged = merge_low([_low(1, 10, 25.0), _low(13, 22, 21.0)])
    assert merged == [
        Interval(start=1, end=10, windows=1, peak=25.0, mean=25.0),
        Interval(start=13, end=22, windows=1, peak=21.0, mean=21.0),
    ]


def test_the_peak_is_the_highest_dust_score_and_the_lowest_protein_complexity():
    windows = [_low(1, 10, 1.0), _low(5, 14, 3.0), _low(9, 18, 2.0)]
    assert merge_low(windows, nucleotide=True)[0].peak == pytest.approx(3.0)
    assert merge_low(windows, nucleotide=False)[0].peak == pytest.approx(1.0)
    assert merge_low(windows)[0].mean == pytest.approx(2.0)


def test_merging_ignores_the_windows_that_were_not_called():
    high = Window(start=5, end=14, score=0.0, entropy=2.0, gc=None, countable=10, low=False)
    assert merge_low([_low(1, 10, 25.0), high]) == [
        Interval(start=1, end=10, windows=1, peak=25.0, mean=25.0)
    ]
    assert merge_low([high]) == []


def test_a_window_spans_its_inclusive_positions_and_is_plotted_at_its_middle():
    window = _low(1, 64, 31.0)
    assert window.length == 64
    assert window.midpoint == pytest.approx(32.5)
    assert Interval(start=101, end=164, windows=1, peak=0.0, mean=0.0).length == 64


def test_the_ceiling_is_set_by_the_window_in_the_alphabet_s_own_measure():
    # DUST: (window - 2) / 2. Protein: log2 of the window, until the window
    # is wider than the alphabet, when it is log2 of the alphabet.
    assert scan("A" * 64, DNA, window=64).ceiling == pytest.approx(31.0)
    assert scan("A" * 64, DNA, window=12).ceiling == pytest.approx(5.0)
    assert scan("A" * 64, PROTEIN, window=SEG_WINDOW).ceiling == pytest.approx(log2(12))
    # Thirty residues over twenty types are at their most even as ten pairs
    # and ten singletons, which is below the log2(20) of twenty equal shares:
    # a window cannot divide evenly unless its length divides the alphabet.
    thirty = -(10 * (2 / 30) * log2(2 / 30) + 10 * (1 / 30) * log2(1 / 30))
    assert thirty < log2(20)
    assert scan("A" * 64, PROTEIN, window=30).ceiling == pytest.approx(thirty)


def test_the_dust_threshold_of_twenty_cannot_be_met_by_a_twelve_base_window():
    profile = scan("ACGTACGTACGT", DNA, window=12, threshold=DUST_THRESHOLD)
    assert profile.ceiling == pytest.approx(5.0)
    assert profile.threshold_reachable is False
    assert profile.low_windows == 0
    assert profile.qualification == (
        "no window can be called: a 12-residue window scores at most 5 on DUST, "
        "below the threshold of 20; the threshold of 20 belongs with a window of 64"
    )


def test_a_protein_window_too_short_to_exceed_the_trigger_calls_every_window():
    # Four residues carry at most log2(4) = 2 bits, which is at or below the
    # SEG trigger of 2.2, so every window is low whatever the sequence is.
    profile = scan("ACDEFGHIKLMN", PROTEIN, window=4)
    assert profile.ceiling == pytest.approx(2.0)
    assert profile.threshold_reachable is False
    assert profile.low_windows == profile.scored
    assert profile.qualification == (
        "every window with a countable residue is called: a 4-residue window of "
        "protein has at most 2.00 bits of residue entropy, at or below the "
        "threshold of 2.2"
    )


def test_the_conventional_window_and_threshold_need_no_qualification():
    assert scan("A" * 64, DNA).qualification == ""
    assert scan("A" * 64, DNA).threshold_reachable is True
    assert scan("ACDEFGHIKLMN", PROTEIN).qualification == ""
    assert scan("ACDEFGHIKLMN", PROTEIN).threshold_reachable is True


def test_the_extreme_window_is_the_highest_for_dust_and_the_lowest_for_protein():
    profile = scan(DE_BRUIJN_3 + "A" * 64, DNA)
    extreme = profile.extreme()
    assert extreme.score == pytest.approx(31.0)
    assert extreme.score == max(profile.scores)

    protein = scan("Q" * 12 + "ACDEFGHIKLMN", PROTEIN)
    assert protein.extreme().score == 0.0
    assert protein.extreme().score == min(protein.scores)


def test_tied_extremes_are_broken_by_the_earliest_position():
    assert scan("A" * 70, DNA).extreme().start == 1
    assert scan("A" * 20, PROTEIN).extreme().start == 1


def test_a_window_longer_than_the_sequence_scores_nothing():
    profile = scan("ACGT", DNA, window=DUST_WINDOW)
    assert profile.scored == 0
    assert list(profile.intervals) == []
    assert profile.extreme() is None
    assert profile.masked_share == 0.0
    assert list(iter_windows("ACGT", DNA, DUST_WINDOW, DUST_THRESHOLD)) == []


def test_the_profile_records_what_it_was_scanned_with():
    profile = scan("A" * 64, DNA, name="tract", window=64, step=3, threshold=25.0)
    assert profile.name == "tract"
    assert profile.length == 64
    assert profile.alphabet is DNA
    assert profile.window == 64
    assert profile.step == 3
    assert profile.threshold == 25.0
    assert profile.entropies == [0.0]


def test_a_window_or_step_below_one_is_refused():
    with pytest.raises(ValueError, match="window must be 1 or greater, got 0"):
        scan("ACGT", DNA, window=0)
    with pytest.raises(ValueError, match="step must be 1 or greater, got 0"):
        scan("ACGT", DNA, step=0)


def test_a_window_holding_nothing_countable_is_not_called_low_complexity():
    # Its entropy is zero because there is nothing in it, not because what is
    # in it repeats; masking on that basis would report a run of X as
    # low-complexity protein.
    protein = scan("X" * 24, PROTEIN, window=12)
    assert protein.scored > 0
    assert all(window.countable == 0 for window in protein.windows)
    assert protein.low_windows == 0
    assert protein.intervals == []
    assert protein.masked_share == 0.0


def test_the_two_alphabets_agree_about_a_window_of_pure_ambiguity():
    protein = scan("X" * 24, PROTEIN, window=12)
    nucleotide = scan("N" * 76, DNA, window=64)
    assert nucleotide.low_windows == protein.low_windows == 0
    assert nucleotide.intervals == protein.intervals == []


def test_the_threshold_caveat_is_taken_from_the_settings_alone():
    # The same question, asked without a scored profile: a renderer that kept
    # no window track still has to be able to print the caveat.
    assert threshold_reachable(DNA, 64, 20.0) is True
    assert threshold_reachable(DNA, 12, 20.0) is False
    assert threshold_note(DNA, 64, 20.0) == ""
    assert threshold_note(DNA, 12, 20.0) == scan("A" * 40, DNA, window=12).qualification
    assert score_ceiling(DNA, 64) == pytest.approx(31.0)
