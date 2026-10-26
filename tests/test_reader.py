"""What ``kmer_counter.reader`` must keep true.

Everything downstream is measured on the records this module hands over, so a
silent misparse is worse than a crash: a column of numbers absorbed as
residues would be reported with an entropy and a spectrum. These tests protect
the parsing rules (wrapped FASTA lines, strict four-line FASTQ), the promise
that a record's sequence is uppercased and free of whitespace, the recognition
of compression from magic bytes rather than from the file name, and the
promise that every malformed or wrongly encoded input arrives as a
``SequenceError`` naming the line, never as a traceback.

Compressed fixtures are written in ``tmp_path`` with the stdlib compressors so
nothing is downloaded and nothing is left behind.
"""

import bz2
import gc
import gzip
import io
import lzma
import sys
import warnings

import pytest

from kmer_counter import reader
from kmer_counter.reader import Record, SequenceError

FASTA = ">one first record\nACGTACGTAC\nGTACGT\n>two\nTTTT\n"


def lines(text):
    """The text as a line iterator, which is what a pipe hands the parser."""
    return iter(text.splitlines(keepends=True))


# --- the record ---------------------------------------------------------


def test_the_formats_offered_are_sniffing_fasta_and_fastq():
    assert reader.FORMATS == ("auto", "fasta", "fastq")


def test_a_record_is_as_long_as_its_sequence():
    record = Record("id", "a description", "ACGTN")
    assert len(record) == 5
    assert record.quality is None


def test_the_header_rejoins_the_identifier_and_the_description():
    assert Record("id", "some text", "A").header == "id some text"


def test_a_record_without_a_description_has_a_bare_header():
    # The join must not leave the trailing space behind.
    assert Record("id", "", "A").header == "id"


# --- FASTA --------------------------------------------------------------


def test_wrapped_sequence_lines_are_joined_into_one_sequence():
    first, second = reader.parse_fasta(lines(FASTA))
    assert first.identifier == "one"
    assert first.description == "first record"
    assert first.sequence == "ACGTACGTACGTACGT"
    assert len(first) == 16
    assert second.identifier == "two"
    assert second.sequence == "TTTT"


def test_blank_lines_anywhere_are_ignored():
    text = "\n\n>one\n\nACGT\n\n\nACGT\n\n>two\n\nTT\n\n"
    records = list(reader.parse_fasta(lines(text)))
    assert [(r.identifier, r.sequence) for r in records] == [
        ("one", "ACGTACGT"),
        ("two", "TT"),
    ]


def test_lowercase_input_is_uppercased():
    # Soft-masked genome FASTA is lowercase for repeats; the counts must not
    # treat a masked base as a different residue.
    record = reader.read_one(io.StringIO(">m\nacgtACGTnnnn\n"))
    assert record.sequence == "ACGTACGTNNNN"


def test_whitespace_inside_a_sequence_line_is_removed():
    record = reader.read_one(io.StringIO(">m\nACG T\tACGT\n"))
    assert record.sequence == "ACGTACGT"


def test_a_byte_order_mark_does_not_hide_the_first_header():
    record = reader.read_one(io.StringIO("\ufeff>bom\nACGT\n"))
    assert record.identifier == "bom"
    assert record.sequence == "ACGT"


def test_carriage_returns_are_stripped_from_every_line(tmp_path):
    path = tmp_path / "crlf.fasta"
    path.write_bytes(b">one win\r\nACGT\r\nACGT\r\n>two\r\nTTTT\r\n")
    records = list(reader.read_records(path))
    assert [(r.identifier, r.description, r.sequence) for r in records] == [
        ("one", "win", "ACGTACGT"),
        ("two", "", "TTTT"),
    ]


def test_a_header_with_no_sequence_gives_an_empty_record():
    # A record with no residues is legal FASTA and must be reported, not
    # dropped: dropping it would silently change how many records a file has.
    records = list(reader.parse_fasta(lines(">empty\n>full\nACGT\n")))
    assert len(records) == 2
    assert records[0].identifier == "empty"
    assert records[0].sequence == ""
    assert len(records[0]) == 0


def test_a_file_whose_only_record_is_empty_is_still_one_record():
    record = reader.read_one(io.StringIO(">empty\n"))
    assert record.identifier == "empty"
    assert record.sequence == ""


def test_the_identifier_is_the_header_up_to_the_first_whitespace():
    record = reader.read_one(io.StringIO(">NC_001401.2 Adeno-associated virus - 2\nACGT\n"))
    assert record.identifier == "NC_001401.2"
    assert record.description == "Adeno-associated virus - 2"


def test_a_header_with_no_identifier_is_refused():
    with pytest.raises(SequenceError, match="line 1: header has no identifier"):
        list(reader.parse_fasta(lines(">\nACGT\n")))


def test_a_header_of_only_whitespace_is_refused():
    with pytest.raises(SequenceError, match="line 2: header has no identifier"):
        list(reader.parse_fasta(lines(">first\n>   \nACGT\n")))


def test_sequence_data_before_the_first_header_is_refused():
    with pytest.raises(SequenceError, match="line 1: sequence data before any '>' header"):
        list(reader.parse_fasta(lines("ACGT\n>one\nACGT\n")))


def test_an_input_with_no_records_at_all_is_refused():
    with pytest.raises(SequenceError, match="input contains no FASTA records"):
        list(reader.parse_fasta(lines("")))


def test_an_input_of_only_blank_lines_holds_no_records():
    with pytest.raises(SequenceError, match="input contains no FASTA records"):
        list(reader.parse_fasta(lines("\n\n\n")))


def test_data_that_is_not_sequence_is_refused_with_the_offending_characters():
    # A saved error page or a column of numbers must not be absorbed as
    # residues; digits are in no alphabet, so the sieve catches them.
    with pytest.raises(
        SequenceError,
        match=r"line 2: not sequence data; unexpected character\(s\): '1', '2', '3'",
    ):
        list(reader.parse_fasta(lines(">one\n123\n")))


def test_at_most_five_offending_characters_are_shown():
    with pytest.raises(SequenceError, match=r"'1', '2', '3', '4', '5', \.\.\."):
        list(reader.parse_fasta(lines(">one\n123456789\n")))


def test_gaps_and_the_stop_character_pass_the_residue_check():
    # The reader checks against the union of the alphabets, because the
    # alphabet is not known until a record has been read.
    record = reader.read_one(io.StringIO(">aligned\nAC-GT.AC~GT*\n"))
    assert record.sequence == "AC-GT.AC~GT*"


# --- FASTQ --------------------------------------------------------------


def test_a_four_line_fastq_record_carries_its_quality_string():
    record = reader.read_one(io.StringIO("@r1 first read\nACGTA\n+\nIIIII\n"), "fastq")
    assert record.identifier == "r1"
    assert record.description == "first read"
    assert record.sequence == "ACGTA"
    assert record.quality == "IIIII"
    assert len(record) == 5


def test_several_fastq_records_are_read_in_order():
    text = "@r1\nACGT\n+\nIIII\n@r2\nTTTT\n+r2\n####\n"
    records = list(reader.parse_fastq(lines(text)))
    assert [(r.identifier, r.sequence, r.quality) for r in records] == [
        ("r1", "ACGT", "IIII"),
        ("r2", "TTTT", "####"),
    ]


def test_fastq_sequence_is_uppercased_but_quality_is_left_alone():
    # Quality is Phred text, where case is a score; only residues are folded.
    record = reader.read_one(io.StringIO("@r1\nacgt\n+\naBcD\n"), "fastq")
    assert record.sequence == "ACGT"
    assert record.quality == "aBcD"


def test_blank_lines_between_fastq_records_are_ignored():
    text = "\n@r1\nACGT\n+\nIIII\n\n@r2\nTTTT\n+\nIIII\n"
    records = list(reader.parse_fastq(lines(text)))
    assert [r.identifier for r in records] == ["r1", "r2"]


def test_a_truncated_fastq_record_is_refused_with_its_line_count():
    with pytest.raises(
        SequenceError,
        match="line 1: truncated FASTQ record; expected 4 lines, got 3",
    ):
        list(reader.parse_fastq(lines("@r1\nACGT\n+\n")))


def test_a_truncated_record_is_reported_at_the_line_it_started_on():
    text = "@r1\nACGT\n+\nIIII\n@r2\nTTTT\n"
    with pytest.raises(SequenceError, match="line 5: truncated FASTQ record"):
        list(reader.parse_fastq(lines(text)))


def test_a_record_that_does_not_begin_with_an_at_sign_is_refused():
    with pytest.raises(SequenceError, match="line 1: FASTQ record does not begin with '@'"):
        list(reader.parse_fastq(lines("r1\nACGT\n+\nIIII\n")))


def test_a_missing_separator_line_is_refused():
    with pytest.raises(SequenceError, match=r"line 3: expected a '\+' separator line"):
        list(reader.parse_fastq(lines("@r1\nACGT\nIIII\n+\n")))


def test_line_wrapped_fastq_is_refused_rather_than_guessed_at():
    # Wrapping puts sequence where the '+' belongs. The format gives no
    # reliable way to tell a wrapped quality line from the next record, so a
    # guess would silently misreport every metric.
    wrapped = "@r1\nACGT\nACGT\n+\nIIII\nIIII\n"
    with pytest.raises(SequenceError, match="line-wrapped FASTQ is not supported"):
        list(reader.parse_fastq(lines(wrapped)))


def test_a_quality_string_of_the_wrong_length_is_refused():
    with pytest.raises(
        SequenceError,
        match="line 4: quality string is 3 characters for a 4-residue sequence",
    ):
        list(reader.parse_fastq(lines("@r1\nACGT\n+\nIII\n")))


def test_a_quality_string_longer_than_the_sequence_is_refused():
    with pytest.raises(SequenceError, match="quality string is 6 characters for a 4-residue"):
        list(reader.parse_fastq(lines("@r1\nACGT\n+\nIIIIII\n")))


def test_a_quality_line_that_is_not_printable_ascii_is_refused():
    # Every Phred encoding sits inside printable ASCII, so anything outside it
    # means the file is not what it claims to be.
    with pytest.raises(SequenceError, match="line 4: not a quality string"):
        list(reader.parse_fastq(lines("@r1\nACGT\n+\nII\u00b5I\n")))


def test_a_fastq_header_with_no_identifier_is_refused():
    with pytest.raises(SequenceError, match="line 1: header has no identifier"):
        list(reader.parse_fastq(lines("@\nACGT\n+\nIIII\n")))


def test_fastq_input_with_no_records_is_refused():
    with pytest.raises(SequenceError, match="input contains no FASTQ records"):
        list(reader.parse_fastq(lines("")))


# --- format sniffing ----------------------------------------------------


def test_sniffing_recognises_fasta_and_restores_the_line_it_read():
    # Sniffing must cost nothing and work on a pipe, so the consumed line is
    # handed back rather than re-read from the source.
    source = (line for line in [">one\n", "ACGT\n"])
    detected, restored = reader.detect_format(source)
    assert detected == "fasta"
    assert list(restored) == [">one\n", "ACGT\n"]


def test_sniffing_recognises_fastq_from_the_at_sign():
    source = (line for line in ["@r1\n", "ACGT\n", "+\n", "IIII\n"])
    detected, restored = reader.detect_format(source)
    assert detected == "fastq"
    assert list(restored)[0] == "@r1\n"


def test_sniffing_skips_leading_blank_lines_and_keeps_them():
    # The blank lines are restored too, so the line numbers in any later error
    # still refer to the file as it is on disk.
    source = (line for line in ["\n", "\n", ">one\n", "ACGT\n"])
    detected, restored = reader.detect_format(source)
    assert detected == "fasta"
    assert list(restored) == ["\n", "\n", ">one\n", "ACGT\n"]


def test_sniffing_sees_past_a_byte_order_mark():
    source = (line for line in ["\ufeff>one\n", "ACGT\n"])
    detected, _ = reader.detect_format(source)
    assert detected == "fasta"


def test_sniffing_refuses_input_that_is_neither_format():
    with pytest.raises(SequenceError, match="input is neither FASTA nor FASTQ"):
        reader.detect_format(lines("ACGT\n>one\n"))


def test_sniffing_names_the_character_the_input_begins_with():
    with pytest.raises(SequenceError, match=r"the first line begins with '\{', not '>' or '@'"):
        reader.detect_format(lines('{"error": "not found"}\n'))


def test_sniffing_refuses_empty_input():
    with pytest.raises(SequenceError, match="input is empty"):
        reader.detect_format(lines(""))


def test_an_unknown_input_format_is_refused_by_name():
    with pytest.raises(SequenceError, match="unknown input format: 'fast5'"):
        list(reader.read_records(io.StringIO(FASTA), "fast5"))


# --- opening, compression and encoding ----------------------------------


@pytest.mark.parametrize(
    "suffix,compress",
    [(".gz", gzip.compress), (".bz2", bz2.compress), (".xz", lzma.compress)],
)
def test_compression_is_recognised_from_the_magic_bytes(tmp_path, suffix, compress):
    path = tmp_path / ("records.fasta" + suffix)
    path.write_bytes(compress(FASTA.encode()))
    records = list(reader.read_records(path))
    assert [(r.identifier, r.sequence) for r in records] == [
        ("one", "ACGTACGTACGTACGT"),
        ("two", "TTTT"),
    ]


def test_compression_is_recognised_whatever_the_file_is_called(tmp_path):
    # The name is not evidence: a gzip saved as .fasta is still a gzip, and a
    # plain file named .gz is still plain text.
    compressed = tmp_path / "plain.fasta"
    compressed.write_bytes(gzip.compress(FASTA.encode()))
    assert [r.identifier for r in reader.read_records(compressed)] == ["one", "two"]
    plain = tmp_path / "misnamed.fasta.gz"
    plain.write_bytes(FASTA.encode())
    assert [r.identifier for r in reader.read_records(plain)] == ["one", "two"]


def test_a_truncated_compressed_file_is_an_input_error(tmp_path):
    # A half-finished download is an ordinary event: it must arrive as a
    # SequenceError about the stream, not as a bare EOFError.
    blob = gzip.compress(FASTA.encode() * 20)
    path = tmp_path / "half.fasta.gz"
    path.write_bytes(blob[: len(blob) // 2])
    with pytest.raises(SequenceError, match="compressed input is truncated"):
        list(reader.read_records(path))


def test_a_file_saved_as_utf_16_is_an_input_error_not_a_traceback(tmp_path):
    path = tmp_path / "utf16.fasta"
    path.write_bytes(FASTA.encode("utf-16"))
    with pytest.raises(SequenceError, match="input is not UTF-8 text"):
        list(reader.read_records(path))


def test_the_utf_16_error_says_how_to_fix_the_file(tmp_path):
    path = tmp_path / "utf16.fasta"
    path.write_bytes(FASTA.encode("utf-16"))
    with pytest.raises(SequenceError, match="has to be re-encoded"):
        list(reader.read_records(path))


def test_a_utf_8_byte_order_mark_on_disk_is_consumed_by_the_decoder(tmp_path):
    path = tmp_path / "bom.fasta"
    path.write_bytes(b"\xef\xbb\xbf" + FASTA.encode())
    assert [r.identifier for r in reader.read_records(path)] == ["one", "two"]


def test_opening_a_directory_or_a_missing_file_says_which(tmp_path):
    with pytest.raises(IsADirectoryError, match="not a file"):
        reader.open_text(str(tmp_path))
    with pytest.raises(FileNotFoundError, match="no such file"):
        reader.open_text(str(tmp_path / "absent.fasta"))


def test_an_open_text_handle_is_passed_through_untouched():
    handle = io.StringIO(FASTA)
    assert reader.open_text(handle) is handle


def test_a_dash_reads_standard_input(monkeypatch):
    stdin = type("Stdin", (), {})()
    stdin.buffer = io.BytesIO(FASTA.encode())
    monkeypatch.setattr(sys, "stdin", stdin)
    assert [r.identifier for r in reader.read_records("-")] == ["one", "two"]


def test_standard_input_is_decompressed_like_a_file(monkeypatch):
    stdin = type("Stdin", (), {})()
    stdin.buffer = io.BytesIO(gzip.compress(FASTA.encode()))
    monkeypatch.setattr(sys, "stdin", stdin)
    assert [r.identifier for r in reader.read_records("-")] == ["one", "two"]


# --- read_records, read_one and naming ----------------------------------


def test_the_limit_stops_after_that_many_records():
    assert len(list(reader.read_records(io.StringIO(FASTA), limit=1))) == 1
    assert len(list(reader.read_records(io.StringIO(FASTA), limit=0))) == 2
    assert len(list(reader.read_records(io.StringIO(FASTA), limit=9))) == 2


def test_read_one_returns_the_single_record_of_a_single_record_file(examples):
    record = reader.read_one(examples / "phix174.fasta")
    assert record.identifier == "NC_001422.1"
    assert len(record) == 5386


def test_read_one_refuses_a_file_that_holds_several_records():
    with pytest.raises(SequenceError, match="holds several"):
        reader.read_one(io.StringIO(FASTA))


def test_read_one_refuses_input_with_no_records():
    with pytest.raises(SequenceError, match="input is empty"):
        reader.read_one(io.StringIO(""))


def test_the_bundled_examples_read_at_their_documented_lengths(examples):
    expected = {
        "aav2.fasta": ("NC_001401.2", 4679),
        "phix174.fasta": ("NC_001422.1", 5386),
        "aav2_vp1.fasta": ("YP_680426.1", 735),
        "huntingtin.fasta": ("NP_002102.4", 3144),
    }
    for name, (identifier, length) in expected.items():
        record = reader.read_one(examples / name)
        assert record.identifier == identifier
        assert len(record) == length
        assert record.sequence == record.sequence.upper()
        assert record.quality is None


def test_source_name_calls_standard_input_by_its_name():
    assert reader.source_name("-") == "standard input"


def test_source_name_drops_the_directory_of_a_path(tmp_path):
    path = tmp_path / "records.fasta"
    path.write_text(FASTA, encoding="utf-8")
    assert reader.source_name(path) == "records.fasta"
    assert reader.source_name(str(path)) == "records.fasta"


def test_source_name_of_an_unnamed_handle_is_input():
    assert reader.source_name(io.StringIO(FASTA)) == "input"


def test_source_name_of_an_open_file_is_the_name_it_was_opened_with(tmp_path):
    path = tmp_path / "records.fasta"
    path.write_text(FASTA, encoding="utf-8")
    with path.open() as handle:
        assert reader.source_name(handle) == str(path)


def test_a_handle_this_module_opened_is_closed_when_iteration_ends(tmp_path):
    # A caller reading a directory of files should not leak a descriptor per
    # file; the generator closes what it opened, exhausted or not.
    path = tmp_path / "two.fasta"
    path.write_text(">a\nACGT\n>b\nTGCA\n", encoding="utf-8")

    records = reader.read_records(path)
    next(records)
    records.close()

    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        gc.collect()

    exhausted = list(reader.read_records(path))
    assert len(exhausted) == 2


def test_a_handle_passed_in_is_left_open_for_its_owner():
    # The caller opened it, so the caller closes it: a reader that shut a
    # handle it was lent would break the second read of the same stream.
    handle = io.StringIO(">a\nACGT\n")
    list(reader.read_records(handle))
    assert not handle.closed


def test_reading_a_limited_number_of_records_still_closes_the_file(tmp_path):
    path = tmp_path / "many.fasta"
    path.write_text("".join(f">r{index}\nACGT\n" for index in range(5)), encoding="utf-8")
    assert len(list(reader.read_records(path, limit=2))) == 2
    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        gc.collect()


def test_a_corrupt_gzip_stream_is_an_input_error_not_a_traceback(tmp_path):
    # gzip reports a damaged trailer as BadGzipFile, which is an OSError and
    # not a zlib error, so it used to reach the user as a traceback.
    good = (">a\n" + "ACGT" * 10 + "\n").encode()
    damaged = bytearray(gzip.compress(good))
    damaged[-3] ^= 0xFF
    path = tmp_path / "broken.fasta.gz"
    path.write_bytes(bytes(damaged))
    with pytest.raises(SequenceError, match="corrupt|cannot be read|truncated"):
        list(reader.read_records(path))


def test_a_corrupt_bzip2_stream_is_an_input_error_not_a_traceback(tmp_path):
    # bzip2 reports a damaged stream as a plain OSError.
    good = (">a\n" + "ACGT" * 10 + "\n").encode()
    damaged = bytearray(bz2.compress(good))
    damaged[-3] ^= 0xFF
    path = tmp_path / "broken.fasta.bz2"
    path.write_bytes(bytes(damaged))
    with pytest.raises(SequenceError, match="corrupt|cannot be read|truncated"):
        list(reader.read_records(path))


def test_a_truncated_xz_stream_is_an_input_error_too(tmp_path):
    good = (">a\n" + "ACGT" * 10 + "\n").encode()
    path = tmp_path / "cut.fasta.xz"
    path.write_bytes(lzma.compress(good)[:-12])
    with pytest.raises(SequenceError, match="corrupt|cannot be read|truncated"):
        list(reader.read_records(path))
