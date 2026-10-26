"""The tabular outputs: one table, built once, rendered the same way twice.

``rows`` is what the ``--output`` file, the embedded report downloads and the
TSV on standard output all come from, so the rules it has to keep are about
fidelity rather than appearance: a float keeps enough digits to read back as
the same double, a missing value is an empty field and not a zero, a tab or a
newline inside a FASTA description does not become a second row, and every
builder yields exactly as many values as it declares columns.
"""

import pytest

from kmer_counter import rows
from kmer_counter.analysis import analyse
from kmer_counter.distance import matrix
from kmer_counter.rows import field, render


@pytest.fixture(scope="module")
def dna(examples):
    """The AAV2 genome, counted to k=4."""
    return analyse(str(examples / "aav2.fasta"), k=4)


@pytest.fixture(scope="module")
def protein(examples):
    """Huntingtin, which has both low-complexity regions and long repeats."""
    return analyse(str(examples / "huntingtin.fasta"), k=4)


def test_field_writes_nothing_at_all_for_a_missing_value():
    # A GC content of 0 and no GC content are different facts, so None is an
    # empty field rather than a zero.
    assert field(None) == ""


def test_field_writes_a_boolean_as_a_word_not_as_a_number():
    # bool is a subclass of int, so this is the one type order that matters.
    assert field(True) == "true"
    assert field(False) == "false"


def test_field_round_trips_a_float_at_full_precision():
    # 0.1 + 0.2 is not 0.3, and a table written to two decimal places would
    # lose the difference the reader may be looking for.
    for value in (0.1 + 0.2, 1 / 3, 31.0, 1e-17, 4.440892098500626e-16):
        assert float(field(value)) == value
    assert field(0.1) == "0.1"


def test_field_flattens_a_tab_or_newline_to_a_space_so_one_row_stays_one_row():
    written = field("major coat\tprotein\nVP1\r[AAV2]")
    assert written == "major coat protein VP1 [AAV2]"
    assert "\t" not in written and "\n" not in written and "\r" not in written


def test_render_writes_a_header_line_and_ends_with_a_newline():
    text = render(("word", "count"), [["ACGT", 7]])
    assert text == "word\tcount\nACGT\t7\n"
    assert text.endswith("\n")


def test_render_writes_one_line_per_row_and_a_header():
    rendered = render(("a", "b"), [[1, 2], [3, 4], [5, 6]])
    assert rendered.splitlines() == ["a\tb", "1\t2", "3\t4", "5\t6"]


def test_render_writes_a_header_alone_for_an_empty_table():
    assert render(("a", "b"), []) == "a\tb\n"


def test_render_leaves_a_missing_value_as_an_empty_field():
    assert render(("count", "gc"), [[3, None]]) == "count\tgc\n3\t\n"


def test_spectrum_rows_match_their_columns(dna):
    built = rows.spectrum_rows(dna)
    assert built
    assert all(len(row) == len(rows.SPECTRUM_COLUMNS) for row in built)
    # One row per word that occurs, most abundant first.
    assert len(built) == dna.spectrum.distinct
    assert [row[1] for row in built] == sorted(
        (row[1] for row in built), reverse=True
    )


def test_spectrum_rows_carry_the_frequency_the_count_implies(dna):
    word, count, frequency = rows.spectrum_rows(dna)[0]
    assert count == dna.spectrum.counts[word]
    assert frequency == pytest.approx(count / dna.spectrum.windows)


def test_window_rows_cover_every_window_of_every_profile_kept(dna):
    built = rows.window_rows(dna.tracks)
    assert all(len(row) == len(rows.WINDOW_COLUMNS) for row in built)
    assert len(built) == sum(len(track.windows) for track in dna.tracks)
    # A 4,679 bp record offers 4,679 - 64 + 1 windows of the default 64.
    assert len(built) == 4679 - dna.window + 1


def test_region_rows_cover_every_region_of_every_record(protein):
    built = rows.region_rows(protein)
    assert built
    assert all(len(row) == len(rows.REGION_COLUMNS) for row in built)
    assert len(built) == sum(len(record.regions) for record in protein.records)
    # start, end and length agree: positions are 1-based and inclusive.
    for _, start, end, length, _, _, _ in built:
        assert length == end - start + 1


def test_repeat_rows_match_their_columns_and_name_their_kind(protein):
    built = rows.repeat_rows(protein)
    assert built
    assert all(len(row) == len(rows.REPEAT_COLUMNS) for row in built)
    assert {row[1] for row in built} == {"run", "tandem"}


def test_repeat_rows_list_a_run_before_the_tandem_repeat_over_the_same_residues(
    protein,
):
    # Huntingtin's polyglutamine tract is both a homopolymer run and a tandem
    # repeat of period one; the docstring says both are reported, and the
    # position order puts the run first.
    at_18 = [row for row in rows.repeat_rows(protein) if row[4] == 18]
    assert [row[1] for row in at_18] == ["run", "tandem"]
    assert all(row[2] == "Q" and row[3] == 1 for row in at_18)


def test_record_rows_give_one_row_per_record(dna):
    built = rows.record_rows(dna)
    assert all(len(row) == len(rows.RECORD_COLUMNS) for row in built)
    assert len(built) == len(dna.records) == 1
    assert built[0][0] == dna.name


def test_entropy_rows_give_one_row_per_counted_word_length(dna):
    built = rows.entropy_rows(dna)
    assert all(len(row) == len(rows.ENTROPY_COLUMNS) for row in built)
    # Every length from one up to k is counted, in ascending order.
    assert [row[0] for row in built] == [1, 2, 3, 4]


def test_summary_row_matches_its_columns(dna):
    row = rows.summary_row(dna)
    assert len(row) == len(rows.SUMMARY_COLUMNS)
    assert row[0] == dna.name
    assert row[2] == "dna"


def test_summary_row_carries_every_qualification_in_one_field(dna):
    row = rows.summary_row(dna)
    written = row[rows.SUMMARY_COLUMNS.index("qualifications")]
    assert written == "; ".join(dna.qualifications)
    # A qualification is prose, and the TSV is one row per input, so every
    # qualification of the run has to fit in this one field.
    assert all(note in written for note in dna.qualifications)


def test_composition_rows_enumerate_every_word_of_the_length(dna):
    built = rows.composition_rows(dna.tables[2])
    assert all(len(row) == len(rows.COMPOSITION_COLUMNS) for row in built)
    # Four bases spell 16 pairs, and the table lists a pair that never occurs
    # rather than omitting it.
    assert len(built) == 16
    assert [row[0] for row in built] == sorted(row[0] for row in built)


def test_distance_rows_give_each_distinct_pair_once(dna):
    spectra = {"first": dna.spectrum, "second": dna.spectra[4], "third": dna.spectrum}
    result = matrix(spectra, "cosine")
    built = rows.distance_rows(result)
    assert all(len(row) == len(rows.DISTANCE_COLUMNS) for row in built)
    # Three sequences make three unordered pairs, not nine ordered ones.
    assert len(built) == 3
    assert [(row[0], row[1]) for row in built] == [
        ("first", "second"),
        ("first", "third"),
        ("second", "third"),
    ]
    assert all(row[2] == 4 and row[3] == "cosine" for row in built)


def test_a_rendered_table_can_be_read_back_as_the_values_that_built_it(protein):
    # The guarantee behind every downloadable table: parse the TSV again and
    # the floats are the floats, not their two-decimal renderings.
    text = render(rows.REGION_COLUMNS, rows.region_rows(protein))
    lines = text.splitlines()
    assert lines[0] == "\t".join(rows.REGION_COLUMNS)
    first = lines[1].split("\t")
    region = protein.records[0].regions[0]
    assert int(first[1]) == region.start
    assert int(first[2]) == region.end
    assert float(first[5]) == region.peak
    assert float(first[6]) == region.mean
