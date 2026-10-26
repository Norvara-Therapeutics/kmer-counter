"""Where a repeat is, not just that there is one.

Everything this module reports can be checked against the sequence by eye,
and that is what these tests do: CAGCAGCAGCAG is four copies of CAG from
position 1 to position 12, a poly-A tract is a repeat of A and not also of AA
or AAA, and a run reported as 3-6 is the four residues at those positions.
The rules protected here are the primitive-unit rule, the containment rule,
the boundaries of a run at the start and end of a sequence, and the exact
minimum at which a run or a repeat begins to be reported.
"""

import pytest

from kmer_counter.alphabet import DNA, PROTEIN
from kmer_counter.reader import read_one
from kmer_counter.repeats import (
    forced_repeated_windows,
    forced_repeat_share,
    MIN_COPIES,
    MIN_RUN,
    Run,
    Tandem,
    _drop_contained,
    abundant_words,
    homopolymer_runs,
    repeat_share,
    tandem_repeats,
)
from kmer_counter.spectrum import count


def test_a_run_is_reported_at_exactly_its_minimum_length():
    # ACAAAAG holds four A at 1-based positions 3 to 6.
    assert homopolymer_runs("ACAAAAG", minimum=4) == [Run(residue="A", start=3, end=6)]
    assert homopolymer_runs("ACAAAAG", minimum=5) == []
    assert homopolymer_runs("ACAAAAG", minimum=4)[0].length == 4


def test_a_run_at_the_start_of_the_sequence_is_found():
    assert homopolymer_runs("AAAAACGT") == [Run(residue="A", start=1, end=5)]


def test_a_run_at_the_end_of_the_sequence_is_found():
    assert homopolymer_runs("ACGTAAAAA") == [Run(residue="A", start=5, end=9)]


def test_the_default_minimum_run_is_five_residues():
    assert MIN_RUN == 5
    assert homopolymer_runs("GG" + "A" * 5 + "GG") == [Run(residue="A", start=3, end=7)]
    assert homopolymer_runs("GG" + "A" * 4 + "GG") == []


def test_an_alphabet_keeps_a_run_of_ambiguity_from_being_called_a_homopolymer():
    assert homopolymer_runs("NNNNNACGT") == [Run(residue="N", start=1, end=5)]
    assert homopolymer_runs("NNNNNACGT", DNA) == []


def test_a_run_of_one_residue_is_not_a_run():
    with pytest.raises(ValueError, match="a run is two residues or more, got 1"):
        homopolymer_runs("AAAA", minimum=1)


def test_a_trinucleotide_repeat_is_reported_as_its_unit_and_its_span():
    # CAGCAGCAGCAG is CAG four times over positions 1-12.
    assert tandem_repeats("CAGCAGCAGCAG") == [Tandem(unit="CAG", start=1, end=12)]
    found = tandem_repeats("CAGCAGCAGCAG")[0]
    assert found.period == 3
    assert found.length == 12
    assert found.copies == pytest.approx(4.0)


def test_a_dinucleotide_repeat_is_reported_as_its_unit_and_its_span():
    assert tandem_repeats("ATATATAT") == [Tandem(unit="AT", start=1, end=8)]
    assert tandem_repeats("ATATATAT")[0].copies == pytest.approx(4.0)


def test_a_poly_a_tract_is_a_repeat_of_a_and_not_also_of_aa_or_aaa():
    # AA and AAA are repetitions of A, so they are not primitive units and
    # the tract is reported once, at period one.
    assert tandem_repeats("A" * 10) == [Tandem(unit="A", start=1, end=10)]


def test_a_repeat_of_a_repeated_unit_is_not_reported_beside_its_primitive():
    # CAGCAG is a repetition of CAG, so the period-6 reading is dropped.
    assert [found.period for found in tandem_repeats("CAGCAGCAGCAGCAGCAG")] == [3]


def test_a_region_contained_in_one_of_shorter_period_is_dropped():
    # Both readings describe the same residues, and the shorter period is the
    # one that explains them, so only it survives.
    kept = _drop_contained([Tandem("A", 1, 12), Tandem("CAG", 4, 9)])
    assert kept == [Tandem("A", 1, 12)]
    # A region that merely overlaps is not contained, and is kept.
    kept = _drop_contained([Tandem("A", 1, 12), Tandem("CAG", 10, 21)])
    assert kept == [Tandem("A", 1, 12), Tandem("CAG", 10, 21)]


def test_a_part_copy_at_the_end_is_counted_as_a_fraction():
    # CAGCAGCAGCA is eleven residues of a three-residue unit: 3.67 copies,
    # and rounding down would lose the CA that makes it a repeat.
    found = tandem_repeats("CAGCAGCAGCA")[0]
    assert found == Tandem(unit="CAG", start=1, end=11)
    assert found.copies == pytest.approx(11 / 3)
    # The docstring's own example, at the two copies it needs to be reported.
    partial = tandem_repeats("CAGCAGCA", min_copies=2)[0]
    assert partial == Tandem(unit="CAG", start=1, end=8)
    assert partial.copies == pytest.approx(8 / 3)


def test_three_copies_are_needed_before_a_repeat_is_reported():
    assert MIN_COPIES == 3
    assert tandem_repeats("CAGCAGCA") == []
    assert tandem_repeats("CAGCAGCAG") == [Tandem(unit="CAG", start=1, end=9)]


def test_the_search_stops_at_the_longest_period_asked_for():
    assert tandem_repeats("CAGCAGCAGCAG", max_period=2) == []
    assert tandem_repeats("CAGCAGCAGCAG", max_period=3) == [
        Tandem(unit="CAG", start=1, end=12)
    ]


def test_a_repeat_holding_an_uncounted_residue_is_not_reported():
    assert tandem_repeats("CANCANCANCAN") == [Tandem(unit="CAN", start=1, end=12)]
    assert tandem_repeats("CANCANCANCAN", DNA) == []


def test_a_protein_tract_is_found_in_the_same_way(examples):
    # Huntingtin opens with a polyQ tract followed by a polyP tract; both are
    # perfect runs of one residue and so are reported at period one.
    protein = read_one(examples / "huntingtin.fasta").sequence
    runs = homopolymer_runs(protein, PROTEIN, minimum=10)
    assert runs[0].residue == "Q"
    assert runs[0].start == 18
    assert protein[runs[0].start - 1: runs[0].end] == "Q" * runs[0].length
    assert any(run.residue == "P" for run in runs)


def test_a_repeat_at_the_end_of_the_sequence_is_found():
    # TTTTCAGCAGCAG: the four T are themselves four copies of T at 1-4, and
    # the trinucleotide repeat runs from 5 to the last residue, 13.
    assert tandem_repeats("TTTT" + "CAG" * 3) == [
        Tandem(unit="T", start=1, end=4),
        Tandem(unit="CAG", start=5, end=13),
    ]


def test_a_region_is_reported_from_where_its_periodicity_begins():
    # In GGGGCAGCAGCAG the G at position 4 already matches the G three
    # residues later, so the period-3 region opens there and its unit is GCA
    # rather than CAG: the unit is the one at the start of the region.
    found = [repeat for repeat in tandem_repeats("GGGG" + "CAG" * 3) if repeat.period == 3]
    assert found == [Tandem(unit="GCA", start=4, end=13)]


def test_a_period_below_one_and_a_single_copy_are_refused():
    with pytest.raises(ValueError, match="max_period must be 1 or greater, got 0"):
        tandem_repeats("AAAA", max_period=0)
    with pytest.raises(ValueError, match="a tandem repeat is two copies or more, got 1"):
        tandem_repeats("AAAA", min_copies=1)


def test_the_abundant_words_carry_every_position_they_occur_at():
    # AAAAA holds AA in each of its four dinucleotide windows.
    spectrum = count("AAAAA", 2, DNA)
    (occurrence,) = abundant_words(spectrum, "AAAAA")
    assert occurrence.word == "AA"
    assert occurrence.count == 4
    assert occurrence.frequency == pytest.approx(1.0)
    assert list(occurrence.positions) == [1, 2, 3, 4]
    assert occurrence.truncated is False
    assert occurrence.spread == 3


def test_a_long_position_list_is_cut_and_said_to_be():
    spectrum = count("AAAAA", 2, DNA)
    (occurrence,) = abundant_words(spectrum, "AAAAA", limit=2)
    assert list(occurrence.positions) == [1, 2]
    assert occurrence.count == 4
    assert occurrence.truncated is True


def test_only_the_number_of_words_asked_for_is_reported():
    spectrum = count("AAACGT", 2, DNA)
    assert len(abundant_words(spectrum, "AAACGT", number=2)) == 2
    assert [found.word for found in abundant_words(spectrum, "AAACGT", number=2)] == [
        "AA",
        "AC",
    ]


def test_a_word_occurring_once_has_no_spread():
    spectrum = count("ACGT", 2, DNA)
    assert abundant_words(spectrum, "ACGT")[0].spread is None


def test_the_repeat_share_is_the_windows_holding_a_word_seen_more_than_once():
    # AAACGT has five dinucleotide windows: AA twice, AC, CG and GT once
    # each, so two of the five windows are involved in a repetition.
    assert repeat_share(count("AAACGT", 2, DNA)) == pytest.approx(0.4)
    assert repeat_share(count("AAAAA", 2, DNA)) == pytest.approx(1.0)
    assert repeat_share(count("ACGT", 2, DNA)) == 0.0
    assert repeat_share(count("NN", 2, DNA)) == 0.0


def test_the_forced_floor_is_what_the_pigeonhole_actually_forces():
    # With W windows over P words, the most singletons P words can supply is
    # P - 1, everything left over piled on one word, so W - P + 1 windows are
    # forced into a repeated word. For the README's k=6 example that is 579
    # of 4,674 windows - an eighth, not 'almost every word'.
    assert forced_repeated_windows(4674, 4096) == 579
    assert forced_repeated_windows(4674, 4096) / 4674 == pytest.approx(0.1238, abs=1e-4)


def test_nothing_is_forced_when_there_is_room_for_every_window():
    assert forced_repeated_windows(100, 4096) == 0
    assert forced_repeated_windows(4096, 4096) == 0
    assert forced_repeated_windows(4097, 4096) == 2


def test_the_forced_floor_is_empty_without_windows():
    assert forced_repeated_windows(0, 4) == 0
    assert forced_repeated_windows(10, 0) == 0
    assert forced_repeat_share(count("", 2, DNA)) == 0.0


def test_the_floor_matches_what_an_exhaustive_search_of_allocations_forces():
    # Brute force for small cases: over every way of putting W windows into P
    # words, the fewest windows that end up in a word used more than once.
    def allocations(total, parts):
        if parts == 1:
            yield (total,)
            return
        for first in range(total + 1):
            for rest in allocations(total - first, parts - 1):
                yield (first, *rest)

    for possible in (2, 3, 4):
        for windows in range(1, 9):
            fewest = min(
                sum(number for number in counts if number > 1)
                for counts in allocations(windows, possible)
            )
            assert forced_repeated_windows(windows, possible) == fewest


def test_the_observed_share_is_never_below_the_floor_it_is_read_against():
    # The floor is a lower bound, so a sequence cannot fall under it. Checked
    # on a repetitive sequence and on one that is as varied as it can be.
    for sequence in ("ACGT" * 40, "ACGTTGCAAGGCCTTAGCATCGATTGCAGCTTACGGATCC" * 3):
        spectrum = count(sequence, 3, DNA)
        assert repeat_share(spectrum) >= forced_repeat_share(spectrum)
