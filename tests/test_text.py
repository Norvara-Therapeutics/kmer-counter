"""What a rendered number is allowed to say.

``text`` is the only place this package rounds, and the rounding rules exist
to stop a rendering from making a claim the value does not support: a share
that is not quite 100% is never written as 100%, a share that is not quite 0%
is never written as 0%, a ratio that is not quite 1 is never written as 1.00,
and a length is written in the unit of its own alphabet. Every expected value
below is derived from the documented rule rather than read off a run.
"""

from kmer_counter import text
from kmer_counter.alphabet import DNA, PROTEIN, RNA
from kmer_counter.text import (
    bar,
    bits,
    number,
    percent,
    plural,
    position_range,
    ratio,
    residues,
    share,
    thousands,
    truncate,
)


def test_percent_never_rounds_a_near_miss_up_to_one_hundred():
    # 99.9986 and 99.999 both round to 100.00 at two places. The bound
    # 100 - 10**-2 is written instead, because a report claiming that every
    # window of a sequence is low-complexity when one of them is not is a
    # claim the reader will act on.
    assert percent(99.9986) == ">99.99%"
    assert percent(99.999) == ">99.99%"


def test_percent_never_rounds_a_near_miss_down_to_zero():
    # 0.004 rounds to 0.00 at two places; a word that occurs is not absent.
    assert percent(0.004) == "<0.01%"
    assert percent(1e-9) == "<0.01%"


def test_percent_writes_an_exact_hundred_and_an_exact_zero_plainly():
    # The guard is for near misses only: a value that really is 100 or 0
    # carries no marker.
    assert percent(100.0) == "100.00%"
    assert percent(0.0) == "0.00%"


def test_percent_marks_a_value_a_hair_above_one_hundred():
    # 100.004 rounds to 100.00 from above, so it is written against the bound
    # above it rather than as a flat 100%.
    assert percent(100.004) == "<100.01%"


def test_percent_honours_the_number_of_places_asked_for():
    # At one place the near-miss bound moves with the precision: 99.96 rounds
    # to 100.0, so the bound is 100 - 10**-1.
    assert percent(12.3456, places=1) == "12.3%"
    assert percent(99.96, places=1) == ">99.9%"


def test_percent_reports_a_missing_value_rather_than_inventing_a_zero():
    assert percent(None) == "n/a"
    assert percent(None, missing="-") == "-"


def test_share_renders_a_fraction_as_a_percentage_under_the_same_rule():
    assert share(0.5379) == "53.79%"
    assert share(None) == "n/a"
    # 99.99986% rounds to 100.00%, and the near-miss rule applies to a share
    # held as a fraction exactly as it does to one held as a percentage.
    assert share(0.9999986) == ">99.99%"


def test_ratio_never_rounds_a_near_miss_to_one():
    # A dinucleotide at 0.996 of its expectation is not at its expectation,
    # and a reader scanning a column of ratios for departures from 1.00 will
    # not look twice at one written as 1.00. Each of these rounds to 1.00 at
    # two places, so each must be rendered as a bound with a marker.
    # (The direction of that marker is wrong in this build: see the defect
    # reported for text.ratio. This test pins only what is certain - that the
    # value is not presented as an exact 1.00, and that it is flagged.)
    for value in (0.996, 0.9999, 1.0001, 1.004):
        rendered = ratio(value)
        assert rendered != "1.00"
        assert rendered[0] in "<>"


def test_ratio_writes_an_exact_one_plainly():
    assert ratio(1.0) == "1.00"


def test_ratio_never_rounds_a_small_value_to_zero():
    # CpG at 0.004 of its expectation is depleted, not absent.
    assert ratio(0.004) == "<0.01"
    assert ratio(0.0) == "0.00"


def test_ratio_writes_an_ordinary_value_to_two_places():
    assert ratio(0.204) == "0.20"
    assert ratio(2.5) == "2.50"
    assert ratio(None) == "n/a"


def test_residues_uses_bases_for_nucleotide_and_residues_for_protein():
    # The two AAV2 records bundled with the package: the genome in bases, the
    # capsid protein in residues.
    assert residues(4679, DNA) == "4,679 bp"
    assert residues(4679, RNA) == "4,679 bp"
    assert residues(735, PROTEIN) == "735 aa"


def test_bar_is_one_mark_wide_for_the_smallest_non_zero_count():
    # 24 * 1/1000 rounds to no marks at all, but a word that occurs has to
    # show as something.
    assert bar(1, 1000) == "#"


def test_bar_is_proportional_to_the_widest_count():
    assert bar(10, 10) == "#" * 24
    assert bar(5, 10) == "#" * 12


def test_bar_is_empty_when_there_is_nothing_to_draw():
    assert bar(0, 10) == ""
    assert bar(5, 0) == ""
    assert bar(-1, 10) == ""


def test_bar_honours_the_width_and_mark_it_is_given():
    assert bar(1, 2, width=10, mark="*") == "*****"


def test_plural_agrees_the_noun_with_the_number():
    assert plural(1, "record") == "1 record"
    assert plural(0, "record") == "0 records"
    assert plural(2000, "record") == "2,000 records"
    assert plural(2, "further region") == "2 further regions"


def test_plural_takes_an_irregular_suffix():
    assert plural(2, "stretch", suffix="es") == "2 stretches"


def test_truncate_leaves_an_identifier_that_fits():
    assert truncate("NC_001401.2") == "NC_001401.2"
    assert truncate("abcde", 5) == "abcde"


def test_truncate_marks_what_it_cut_without_exceeding_the_width():
    # One ellipsis replaces the last character kept, so the result is exactly
    # as wide as the column it was cut for.
    cut = truncate("abcdefgh", 5)
    assert cut == "abcd\u2026"
    assert len(cut) == 5


def test_number_and_bits_report_a_missing_value_rather_than_a_zero():
    assert number(None) == "n/a"
    assert bits(None) == "n/a"
    assert number(0.0) == "0.00"
    assert bits(7.8349) == "7.83 bits"


def test_number_groups_thousands_and_takes_a_precision():
    assert number(4679.5) == "4,679.50"
    assert number(0.99612, 4) == "0.9961"


def test_thousands_groups_an_integer():
    assert thousands(4679) == "4,679"
    assert thousands(0) == "0"


def test_position_range_writes_an_inclusive_interval():
    # 101-164 is 64 residues, both ends included, as the package's positions
    # are everywhere.
    assert position_range(101, 164) == "101-164"
    assert position_range(1000, 2000) == "1,000-2,000"


def test_a_near_miss_ratio_is_written_as_a_bound_the_value_satisfies():
    # Below one the value sits above the next bound down, and above one below
    # the next bound up. A marker pointing the other way states something
    # false about the number it is standing in for.
    assert text.ratio(0.996) == ">0.99"
    assert text.ratio(1.004) == "<1.01"
    assert float(text.ratio(0.996)[1:]) < 0.996
    assert float(text.ratio(1.004)[1:]) > 1.004
    assert text.ratio(1.0) == "1.00"
