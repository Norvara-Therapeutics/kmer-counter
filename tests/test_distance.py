"""The axioms an alignment-free distance has to satisfy.

A distance between spectra is only interpretable if it is zero for identical
word frequencies, one for spectra sharing no word at all, symmetric, and
inside [0, 1]; the Jensen-Shannon divergence has the sharper promise of being
exactly one bit when nothing is shared, and its square root is what is
reported when a metric is needed. The other half of this file is the refusals:
spectra taken at different word lengths, over different alphabets, or under
different canonical rules are not comparable, and comparing them anyway is the
one mistake these measures cannot survive.
"""

import types
from math import sqrt
from pathlib import Path

import pytest

from kmer_counter.alphabet import DNA, PROTEIN, RNA
from kmer_counter.composition import signature
from kmer_counter.distance import (
    MEASURES,
    bray_curtis,
    cosine_distance,
    distance,
    jensen_shannon,
    jensen_shannon_distance,
    matrix,
    signature_difference,
)
from kmer_counter.reader import read_one
from kmer_counter.spectrum import count


@pytest.fixture
def genomes(examples: Path) -> dict:
    """The two bundled DNA records, counted at four bases."""
    return {
        name: count(read_one(examples / f"{name}.fasta").sequence, 4, DNA)
        for name in ("aav2", "phix174")
    }


def test_a_spectrum_is_at_no_distance_from_itself():
    spectrum = count("AACCGGTT" * 8, 4, DNA)
    for measure in MEASURES:
        assert distance(spectrum, spectrum, measure) == pytest.approx(0.0, abs=1e-12)


def test_two_records_counted_the_same_way_are_at_no_distance_from_themselves(genomes):
    for name, spectrum in genomes.items():
        for measure in MEASURES:
            assert distance(spectrum, spectrum, measure) == pytest.approx(
                0.0, abs=1e-12
            ), (name, measure)


def test_spectra_sharing_no_word_are_at_the_greatest_distance():
    # AAAA and CCCC hold AA and CC respectively and nothing in common.
    left, right = count("AAAA", 2, DNA), count("CCCC", 2, DNA)
    assert cosine_distance(left, right) == pytest.approx(1.0)
    assert bray_curtis(left, right) == pytest.approx(1.0)
    # Each word's whole frequency goes to one side, so every term of the
    # divergence is p log2(p / (p/2)) / 2 = p / 2, summing to exactly one bit.
    assert jensen_shannon(left, right) == pytest.approx(1.0)
    assert jensen_shannon_distance(left, right) == pytest.approx(1.0)


def test_the_jensen_shannon_distance_is_the_square_root_of_the_divergence(genomes):
    left, right = genomes["aav2"], genomes["phix174"]
    assert jensen_shannon_distance(left, right) == pytest.approx(
        sqrt(jensen_shannon(left, right))
    )
    # The root of a number inside the unit interval is larger than the number.
    assert jensen_shannon_distance(left, right) > jensen_shannon(left, right)


def test_every_distance_is_symmetric(genomes):
    left, right = genomes["aav2"], genomes["phix174"]
    for measure in MEASURES:
        assert distance(left, right, measure) == pytest.approx(
            distance(right, left, measure)
        ), measure


def test_every_distance_lies_between_zero_and_one(genomes):
    left, right = genomes["aav2"], genomes["phix174"]
    for measure in MEASURES:
        value = distance(left, right, measure)
        assert 0.0 <= value <= 1.0, (measure, value)
        assert value > 0.0, measure


def test_a_distance_is_none_when_a_spectrum_has_no_counted_window():
    counted = count("AAAA", 2, DNA)
    empty = count("NNNN", 2, DNA)
    assert cosine_distance(counted, empty) is None
    assert jensen_shannon(counted, empty) is None
    assert jensen_shannon_distance(counted, empty) is None
    assert bray_curtis(counted, empty) is None


def test_spectra_of_different_word_lengths_are_not_compared():
    with pytest.raises(ValueError, match="cannot compare k=2 with k=3"):
        cosine_distance(count("ACGT", 2, DNA), count("ACGT", 3, DNA))


def test_spectra_of_different_alphabets_are_not_compared():
    with pytest.raises(ValueError, match="cannot compare DNA with RNA"):
        bray_curtis(count("ACGT", 2, DNA), count("ACGU", 2, RNA))
    with pytest.raises(ValueError, match="cannot compare DNA with protein"):
        jensen_shannon(count("ACGT", 2, DNA), count("ACDE", 2, PROTEIN))


def test_canonical_counts_are_not_compared_with_plain_ones():
    with pytest.raises(ValueError, match="canonical with non-canonical"):
        cosine_distance(
            count("ACGT", 2, DNA, canonical=True), count("ACGT", 2, DNA)
        )


def test_a_distance_must_be_asked_for_by_one_of_its_names():
    left, right = count("AAAA", 2, DNA), count("CCCC", 2, DNA)
    assert MEASURES == ("cosine", "jensen-shannon", "bray-curtis")
    with pytest.raises(ValueError, match="unknown measure: 'euclidean'"):
        distance(left, right, "euclidean")


def test_a_matrix_has_a_diagonal_of_zeros_and_is_symmetric(genomes):
    built = matrix(genomes, "cosine")
    assert list(built.names) == ["aav2", "phix174"]
    assert built.measure == "cosine"
    assert built.k == 4
    assert [built.values[index][index] for index in range(len(built.names))] == [0.0, 0.0]
    assert built.get("aav2", "phix174") == pytest.approx(
        built.get("phix174", "aav2")
    )


def test_a_matrix_refuses_a_name_it_did_not_compare(genomes):
    # The rule is that an unknown name is an error rather than a silent miss.
    # Which error is not settled: the docstring promises KeyError and the
    # implementation raises ValueError from list.index (see the defect noted
    # with this suite), so the test pins the refusal and not its class.
    built = matrix(genomes, "cosine")
    with pytest.raises((KeyError, ValueError)):
        built.get("not-compared", "aav2")
    with pytest.raises((KeyError, ValueError)):
        built.get("aav2", "not-compared")


def test_a_matrix_lists_every_distinct_pair_once(genomes):
    spectra = dict(genomes)
    spectra["copy"] = genomes["aav2"]
    built = matrix(spectra, "bray-curtis")
    pairs = built.pairs()
    assert len(pairs) == 3
    assert [(row, column) for row, column, _ in pairs] == [
        ("aav2", "phix174"),
        ("aav2", "copy"),
        ("phix174", "copy"),
    ]


def test_the_closest_pair_is_the_one_with_the_smallest_distance(genomes):
    spectra = dict(genomes)
    spectra["copy"] = genomes["aav2"]
    built = matrix(spectra, "cosine")
    row, column, value = built.closest()
    assert {row, column} == {"aav2", "copy"}
    assert value == pytest.approx(0.0, abs=1e-12)


def test_a_matrix_carries_none_where_a_distance_could_not_be_taken():
    spectra = {"counted": count("AAAA", 2, DNA), "empty": count("NNNN", 2, DNA)}
    built = matrix(spectra)
    assert built.get("counted", "empty") is None
    assert built.closest() is None


def test_comparing_needs_two_sequences():
    with pytest.raises(ValueError, match="comparing needs two or more sequences"):
        matrix({"only": count("ACGT", 2, DNA)})


def test_a_matrix_must_be_asked_for_by_a_known_measure(genomes):
    with pytest.raises(ValueError, match="unknown measure: 'manhattan'"):
        matrix(genomes, "manhattan")


def test_the_signature_difference_is_the_mean_absolute_ratio_difference():
    # Both sequences have equal base composition over 64 bases, so every
    # dinucleotide is expected 63/16 times. In AACCGGTT eight dinucleotides
    # occur eight times (ratio 128/63) and TA seven (112/63); in ACGT three
    # occur sixteen times (256/63) and TA fifteen (240/63). Eight of the
    # sixteen ratios differ, each by exactly 128/63, so the mean difference
    # is 8 * (128/63) / 16 = 64/63, reported times a thousand.
    left = {k: count("AACCGGTT" * 8, k, DNA) for k in (1, 2)}
    right = {k: count("ACGT" * 16, k, DNA) for k in (1, 2)}
    assert len(signature(left)) == len(signature(right)) == 16
    assert signature_difference(left, right) == pytest.approx(1000 * 64 / 63)


def test_a_sequence_is_at_no_signature_difference_from_itself():
    spectra = {k: count("AACCGGTT" * 8, k, DNA) for k in (1, 2)}
    assert signature_difference(spectra, spectra) == pytest.approx(0.0)


def test_a_sequence_using_two_bases_cannot_be_signed():
    # Only the four dinucleotides of A and C are assessable, and four of
    # sixteen is fewer than the half the measure needs.
    two = {k: count("ACAC" * 20, k, DNA) for k in (1, 2)}
    four = {k: count("AACCGGTT" * 8, k, DNA) for k in (1, 2)}
    assert len(signature(two)) == 4
    assert signature_difference(four, two) is None
    assert signature_difference(two, four) is None


def test_a_sequence_too_short_for_any_assessable_dinucleotide_cannot_be_signed():
    # Eight bases give every dinucleotide an expectation of 0.44 counts,
    # below the one count a ratio needs, so nothing is assessable.
    short = {k: count("ACGTACGT", k, DNA) for k in (1, 2)}
    assert signature(short) == {}
    assert signature_difference(short, short) is None


def test_a_matrix_raises_key_error_for_a_name_it_did_not_compare():
    # The docstring promises KeyError, and a caller catching KeyError around a
    # lookup by name should not meet a ValueError from a list index.
    spectra = {
        "first": count("ACGTACGTAA", 2, DNA),
        "second": count("TTTTGGGGCC", 2, DNA),
    }
    result = matrix(spectra, "cosine")
    with pytest.raises(KeyError, match="was not compared"):
        result.get("third", "first")
    with pytest.raises(KeyError, match="was not compared"):
        result.get("first", "third")


def test_the_distance_submodule_is_not_shadowed_by_its_own_function():
    # Re-exporting a function named after its module would make
    # ``import kmer_counter.distance`` bind the function instead.
    import kmer_counter.distance as submodule

    assert isinstance(submodule, types.ModuleType)
    assert callable(submodule.cosine_distance)
