"""The command line: what it refuses, what it writes, and what it warns about.

Three rules are being protected here.

Nothing is read before every output has been checked, so a run that cannot
finish does not start and never truncates an input it is reading: an output
naming an input under any name - a hard link, a symlink - is refused, as are
two outputs sharing standard output and an output in a directory that does
not exist.

A bad argument exits 2 with a message on standard error rather than raising,
because a shell script checks the status.

And every qualification printed in the terminal summary is also in the JSON of
the same run. A machine reading the spectrum cannot see a warning printed
beside it, so a caveat that reaches one renderer and not another is worse than
no caveat at all; that parity is asserted for an ambiguity-heavy input, a
short input whose alphabet detection is uncertain, and a multi-record input.

The CLI is driven in-process through ``cli.main`` so that ``capsys`` sees the
same stream the code writes to.
"""

import argparse
import json
import os

import pytest

from kmer_counter import __version__, cli

#: A non-repetitive DNA flank, written out so the fixtures below are fixed
#: text rather than anything generated. Neither end is an A, so a poly-A
#: tract inserted between two copies of it starts and ends where it is put.
_FLANK = (
    "CGTTGCAATCGGATCCTAGCATGACTGCAATGGCCTTAGCATCGATTGCA"
    "GCTTACGGATCCAATGCATCGTTAGCCATGGATCAGCTAGCTTAACGGTC"
)


def _run(args, capsys):
    """Run the CLI, returning (status, stdout, stderr)."""
    status = cli.main(args)
    captured = capsys.readouterr()
    return status, captured.out, captured.err


def _fails(args, capsys, message):
    """Assert the CLI exits 2 with ``message`` on standard error."""
    status, out, err = _run(args, capsys)
    assert status == 2
    assert message in err
    assert err.startswith("kmer-counter: ")
    return out, err


def _notes(summary):
    """The qualifications a terminal summary printed, in order."""
    return [
        line.strip()[2:] for line in summary.splitlines() if line.strip().startswith("! ")
    ]


@pytest.fixture
def dna_file(tmp_path):
    """A short DNA record, long enough for a window and quick to count."""
    path = tmp_path / "small.fasta"
    path.write_text(">small a short record\n" + _FLANK + "\n", encoding="utf-8")
    return path


@pytest.fixture
def poly_a_file(tmp_path):
    """One 64-base poly-A tract at positions 101-164, flanked either side.

    The flanks are 100 bases each and neither abuts an A, so the tract is
    exactly 101-164 and a window of 64 placed on it is a homopolymer.
    """
    sequence = _FLANK + "A" * 64 + _FLANK[50:] + _FLANK[:50]
    path = tmp_path / "polya.fasta"
    path.write_text(">synth poly-A at 101-164\n" + sequence + "\n", encoding="utf-8")
    return path


@pytest.fixture
def ambiguous_file(tmp_path):
    """A record a third of whose windows hold an N the alphabet cannot count."""
    path = tmp_path / "ambiguous.fasta"
    path.write_text(
        ">ambiguous\n" + "ACGTNNNNACGTACGTNNNNNNNNACGTACGTACGTNNNN" * 5 + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def tiny_file(tmp_path):
    """Six residues: too few for the alphabet detection to be called certain."""
    path = tmp_path / "tiny.fasta"
    path.write_text(">tiny\nACGTAC\n", encoding="utf-8")
    return path


@pytest.fixture
def multi_record_file(tmp_path):
    """Three records, so positions cannot be reported for an abundant word."""
    path = tmp_path / "multi.fasta"
    path.write_text(
        ">one\nACGTACGTACGTACGTACGTAC\n"
        ">two\nGGCATTACGATCGATTACGCAT\n"
        ">three\nTTACGGATCCAATGCATCGTTA\n",
        encoding="utf-8",
    )
    return path


def test_version_prints_the_package_version_and_exits_zero(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"kmer-counter {__version__}"


def test_a_subcommand_is_required(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main([])
    assert exit_info.value.code == 2
    assert "command" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["count", "profile", "compare"])
@pytest.mark.parametrize("output_format", ["table", "tsv", "json", "none"])
def test_each_subcommand_writes_each_format(
    command, output_format, dna_file, examples, capsys
):
    arguments = [command, "-i", str(dna_file)]
    if command == "compare":
        arguments += ["-i", str(examples / "phix174.fasta")]
    status, out, err = _run(arguments + ["--format", output_format], capsys)
    assert status == 0
    assert err == ""
    if output_format == "none":
        # 'none' is what silences the summary when an output owns the stream.
        assert out == ""
    elif output_format == "json":
        assert json.loads(out)
    elif output_format == "tsv":
        # A TSV on standard output starts with its header line.
        assert "\t" in out.splitlines()[0]
    else:
        assert out.strip()


@pytest.mark.parametrize("command", ["count", "profile", "compare"])
def test_the_json_of_each_command_carries_its_qualifications(
    command, dna_file, examples, capsys
):
    arguments = [command, "-i", str(dna_file)]
    if command == "compare":
        arguments += ["-i", str(examples / "phix174.fasta")]
    status, out, _ = _run(arguments + ["--format", "json"], capsys)
    assert status == 0
    payload = json.loads(out)
    assert "qualifications" in payload
    assert isinstance(payload["qualifications"], list)


@pytest.mark.parametrize(
    ("argument", "message"),
    [
        (["-k", "9"], "k must be between 1 and 8: 9"),
        (["-k", "0"], "must be 1 or greater: 0"),
        (["-k", "-1"], "must not be negative: -1"),
        (["--top", "-1"], "must not be negative: -1"),
        (["--limit", "-3"], "must not be negative: -3"),
        (["--min-run", "0"], "must be 1 or greater: 0"),
        (["-k", "four"], "not an integer: 'four'"),
    ],
)
def test_count_refuses_an_argument_out_of_range(argument, message, dna_file, capsys):
    # argparse rejects these before any file is opened, so the input need not
    # even be read for the refusal to be the right one.
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["count", "-i", str(dna_file), *argument])
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("inf", "must be a finite number: 'inf'"),
        ("-inf", "must be a finite number: '-inf'"),
        ("nan", "must be a finite number: 'nan'"),
        ("high", "not a number: 'high'"),
        ("-1", "must not be negative: -1.0"),
    ],
)
def test_profile_refuses_a_threshold_that_is_not_a_finite_score(
    value, message, dna_file, capsys
):
    # A threshold of inf or nan compares false against every score, which
    # would silently turn the low-complexity call into a constant. The value
    # is attached with '=' so that a leading minus is not read as an option.
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["profile", "-i", str(dna_file), f"--threshold={value}"])
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


def test_a_missing_input_is_refused_by_name(tmp_path, capsys):
    missing = tmp_path / "absent.fasta"
    _fails(["count", "-i", str(missing)], capsys, f"no such file: {missing}")


def test_a_directory_as_input_is_refused(tmp_path, capsys):
    _fails(["count", "-i", str(tmp_path)], capsys, f"not a file: {tmp_path}")


def test_standard_input_cannot_be_read_twice(capsys):
    # Two inputs both named '-' would hand the second an exhausted stream.
    _fails(
        ["count", "-i", "-", "-i", "-"], capsys, "standard input can only be read once"
    )


def test_a_table_and_an_output_cannot_share_standard_output(dna_file, capsys):
    # Two documents interleaved on one stream are neither a readable summary
    # nor a parsable TSV, so the summary has to be silenced explicitly.
    _fails(
        ["count", "-i", str(dna_file), "-o", "-", "--format", "table"],
        capsys,
        "the summary has to be silenced with --format none",
    )


def test_two_outputs_cannot_both_go_to_standard_output(dna_file, capsys):
    _fails(
        [
            "profile",
            "-i",
            str(dna_file),
            "--regions",
            "-",
            "--bed",
            "-",
            "--format",
            "none",
        ],
        capsys,
        "two outputs cannot share standard output",
    )


def test_an_output_sent_to_standard_output_alone_is_written(dna_file, capsys):
    status, out, err = _run(
        ["count", "-i", str(dna_file), "-o", "-", "--format", "none"], capsys
    )
    assert status == 0
    assert err == ""
    assert out.splitlines()[0] == "word\tcount\tfrequency"


def test_an_output_that_is_an_input_under_a_hard_link_is_refused(
    tmp_path, dna_file, capsys
):
    # Comparing names alone would let this through and truncate the input
    # half-way through reading it.
    link = tmp_path / "hard.tsv"
    os.link(dna_file, link)
    _fails(
        ["count", "-i", str(dna_file), "-o", str(link), "--format", "none"],
        capsys,
        "writing it would truncate what is being read",
    )


def test_an_output_that_is_an_input_under_a_symlink_is_refused(
    tmp_path, dna_file, capsys
):
    link = tmp_path / "soft.tsv"
    link.symlink_to(dna_file)
    _, err = _fails(
        ["count", "-i", str(dna_file), "-o", str(link), "--format", "none"],
        capsys,
        "writing it would truncate what is being read",
    )
    assert str(dna_file) in err


def test_a_report_that_is_an_input_is_refused(tmp_path, dna_file, capsys):
    link = tmp_path / "report.html"
    os.link(dna_file, link)
    _fails(
        ["count", "-i", str(dna_file), "--report", str(link), "--format", "none"],
        capsys,
        "writing it would truncate what is being read",
    )


def test_an_output_in_a_directory_that_does_not_exist_is_refused(
    tmp_path, dna_file, capsys
):
    absent = tmp_path / "nowhere"
    _fails(
        [
            "count",
            "-i",
            str(dna_file),
            "-o",
            str(absent / "spectrum.tsv"),
            "--format",
            "none",
        ],
        capsys,
        f"no such directory: {absent}",
    )
    assert not absent.exists()


def test_an_output_that_is_a_directory_is_refused(tmp_path, dna_file, capsys):
    _fails(
        ["count", "-i", str(dna_file), "-o", str(tmp_path), "--format", "none"],
        capsys,
        f"output is a directory: {tmp_path}",
    )


def test_an_output_is_checked_before_the_input_is_parsed(tmp_path, capsys):
    # This input exists but is not sequence at all, so parsing it would fail
    # too. The refusal names the output, which is the point of checking the
    # outputs before any work starts: a run that cannot finish does not begin.
    prose = tmp_path / "notes.fasta"
    prose.write_text("this is not a sequence\n", encoding="utf-8")
    _fails(
        [
            "count",
            "-i",
            str(prose),
            "-o",
            str(tmp_path / "gone" / "out.tsv"),
            "--format",
            "none",
        ],
        capsys,
        "no such directory",
    )


def test_count_writes_one_report_per_input_named_after_it(
    tmp_path, examples, capsys
):
    # Several inputs sharing one --report would have each overwrite the last,
    # so the name carries the input it describes.
    status, out, err = _run(
        [
            "count",
            "-i",
            str(examples / "aav2.fasta"),
            "-i",
            str(examples / "phix174.fasta"),
            "--report",
            str(tmp_path / "out.html"),
            "--format",
            "none",
        ],
        capsys,
    )
    assert status == 0
    assert err == "" and out == ""
    written = sorted(path.name for path in tmp_path.iterdir())
    assert written == ["out.aav2.fasta.html", "out.phix174.fasta.html"]
    assert "aav2.fasta" in (tmp_path / "out.aav2.fasta.html").read_text()


def test_several_inputs_cannot_share_one_report_on_standard_output(
    examples, capsys
):
    _fails(
        [
            "count",
            "-i",
            str(examples / "aav2.fasta"),
            "-i",
            str(examples / "phix174.fasta"),
            "--report",
            "-",
            "--format",
            "none",
        ],
        capsys,
        "several inputs cannot share one report on standard output",
    )


def test_count_names_the_input_in_a_spectrum_written_for_several(
    tmp_path, dna_file, examples, capsys
):
    output = tmp_path / "spectra.tsv"
    status, _, err = _run(
        [
            "count",
            "-i",
            str(dna_file),
            "-i",
            str(examples / "phix174.fasta"),
            "-o",
            str(output),
            "--format",
            "none",
        ],
        capsys,
    )
    assert status == 0 and err == ""
    lines = output.read_text().splitlines()
    assert lines[0] == "source\tword\tcount\tfrequency"
    assert {line.split("\t")[0] for line in lines[1:]} == {
        "small.fasta",
        "phix174.fasta",
    }


def test_compare_refuses_a_single_input(dna_file, capsys):
    _fails(
        ["compare", "-i", str(dna_file)], capsys, "comparing needs two or more inputs"
    )


def test_compare_refuses_inputs_of_different_alphabets(examples, capsys):
    # A DNA 4-mer is drawn from 256 words and a protein 4-mer from 160,000, so
    # a distance between the two spectra would be meaningless rather than
    # merely large.
    _, err = _fails(
        [
            "compare",
            "-i",
            str(examples / "aav2.fasta"),
            "-i",
            str(examples / "aav2_vp1.fasta"),
        ],
        capsys,
        "the inputs are not in one alphabet (dna, protein)",
    )
    assert "--alphabet" in err


def test_canonical_counting_is_refused_on_protein(examples, capsys):
    _fails(
        ["count", "-i", str(examples / "aav2_vp1.fasta"), "--canonical"],
        capsys,
        "canonical counting needs a complement, which protein has not",
    )


def test_profile_writes_bed_as_zero_based_half_open_coordinates(
    tmp_path, poly_a_file, capsys
):
    # The DUST score of a 64-base homopolymer is (64 - 2) / 2 = 31, and no
    # window holding a residue other than A can reach it, so at a threshold of
    # 31 exactly one window is called: the tract at 101-164. BED writes that
    # region as '100 164' - start 0-based, end exclusive - and the span stays
    # 64 bases either way.
    bed = tmp_path / "regions.bed"
    status, _, err = _run(
        [
            "profile",
            "-i",
            str(poly_a_file),
            "--threshold",
            "31",
            "--bed",
            str(bed),
            "--format",
            "none",
        ],
        capsys,
    )
    assert status == 0 and err == ""
    fields = bed.read_text().rstrip("\n").split("\t")
    assert fields[0] == "synth"
    assert fields[1] == "100"
    assert fields[2] == "164"
    assert int(fields[2]) - int(fields[1]) == 64
    assert fields[3] == "low_complexity_dust"
    assert float(fields[4]) == pytest.approx(31.0)


def test_profile_writes_the_same_region_one_based_and_inclusive_as_a_tsv(
    tmp_path, poly_a_file, capsys
):
    # The same region through --regions, where positions are the package's
    # own: 101-164, inclusive, 64 residues long.
    regions = tmp_path / "regions.tsv"
    status, _, err = _run(
        [
            "profile",
            "-i",
            str(poly_a_file),
            "--threshold",
            "31",
            "--regions",
            str(regions),
            "--format",
            "none",
        ],
        capsys,
    )
    assert status == 0 and err == ""
    lines = regions.read_text().splitlines()
    assert lines[0] == (
        "source\trecord\tstart\tend\tlength\twindows\tpeak_score\tmean_score"
    )
    assert len(lines) == 2
    source, record, start, end, length = lines[1].split("\t")[:5]
    assert source == "polya.fasta"
    assert record == "synth"
    assert (start, end, length) == ("101", "164", "64")


def test_profile_writes_a_window_table_only_when_asked_for_one(
    tmp_path, dna_file, capsys
):
    output = tmp_path / "windows.tsv"
    status, _, err = _run(
        ["profile", "-i", str(dna_file), "-o", str(output), "--format", "none"], capsys
    )
    assert status == 0 and err == ""
    lines = output.read_text().splitlines()
    assert lines[0] == (
        "source\trecord\tstart\tend\tscore\tentropy_bits\tgc\tcountable"
        "\tlow_complexity"
    )
    # A 100-base record offers 100 - 64 + 1 = 37 windows of the default 64.
    assert len(lines) == 1 + 37


@pytest.mark.parametrize(
    "fixture",
    ["ambiguous_file", "tiny_file", "multi_record_file"],
)
def test_every_terminal_qualification_reaches_the_json_of_the_same_run(
    fixture, request, capsys
):
    # The load-bearing rule: a machine reading the spectrum cannot see a
    # warning printed beside it, so anything the summary says about how to
    # read these numbers has to be in the JSON too.
    path = request.getfixturevalue(fixture)
    status, summary, _ = _run(["count", "-i", str(path), "--format", "table"], capsys)
    assert status == 0
    printed = _notes(summary)
    assert printed, "this input was chosen because it qualifies its own numbers"
    status, document, _ = _run(["count", "-i", str(path), "--format", "json"], capsys)
    assert status == 0
    carried = json.loads(document)["qualifications"]
    assert printed == carried


def test_an_ambiguity_heavy_input_says_how_many_windows_it_could_not_count(
    ambiguous_file, capsys
):
    _, summary, _ = _run(
        ["count", "-i", str(ambiguous_file), "--format", "table"], capsys
    )
    assert any(
        "were not counted, because they hold a residue the DNA alphabet does not"
        " count" in note
        for note in _notes(summary)
    )


def test_a_short_input_says_its_alphabet_was_read_from_little_evidence(
    tiny_file, capsys
):
    # Six residues is below the twenty the detection wants before it reports
    # an alphabet as a fact.
    _, summary, _ = _run(["count", "-i", str(tiny_file), "--format", "table"], capsys)
    assert any(
        note.startswith("alphabet read as DNA: only 6 residues to judge from")
        for note in _notes(summary)
    )


def test_a_multi_record_input_says_why_it_reports_no_positions(
    multi_record_file, capsys
):
    _, summary, _ = _run(
        ["count", "-i", str(multi_record_file), "--format", "table"], capsys
    )
    assert any(
        "abundant words are reported with counts but without positions" in note
        for note in _notes(summary)
    )
    _, document, _ = _run(
        ["count", "-i", str(multi_record_file), "--format", "json"], capsys
    )
    payload = json.loads(document)
    assert all(not word["positions"] for word in payload["abundant"])


def test_every_profile_qualification_reaches_the_json_of_the_same_run(
    examples, capsys
):
    # The same parity rule for the profile command, whose JSON carries the
    # alphabet and low-complexity notes rather than the spectrum ones.
    arguments = ["profile", "-i", str(examples / "aav2_vp1.fasta")]
    _, summary, _ = _run(arguments + ["--format", "table"], capsys)
    _, document, _ = _run(arguments + ["--format", "json"], capsys)
    carried = json.loads(document)["qualifications"]
    assert set(_notes(summary)) <= set(carried)


def test_every_compare_qualification_reaches_the_json_of_the_same_run(
    examples, tiny_file, capsys
):
    # compare prefixes each note with the input it belongs to, since several
    # inputs are summarised in one matrix.
    arguments = [
        "compare",
        "-i",
        str(tiny_file),
        "-i",
        str(examples / "aav2.fasta"),
        "--alphabet",
        "dna",
    ]
    _, summary, _ = _run(arguments + ["--format", "table"], capsys)
    _, document, _ = _run(arguments + ["--format", "json"], capsys)
    carried = json.loads(document)["qualifications"]
    printed = _notes(summary)
    assert printed
    assert printed == carried
    assert all(note.startswith(("tiny.fasta: ", "aav2.fasta: ")) for note in printed)


def test_a_threshold_no_window_can_reach_is_reported_in_both_renderings(
    dna_file, capsys
):
    # A 12-residue window scores at most (12 - 2) / 2 = 5 on DUST, so a
    # threshold of 20 with that window can never be met and the
    # low-complexity call becomes a constant. Saying so is the whole point of
    # the qualification, and both renderings have to say it.
    arguments = ["count", "-i", str(dna_file), "--window", "12", "--threshold", "20"]
    _, summary, _ = _run(arguments + ["--format", "table"], capsys)
    printed = _notes(summary)
    assert any("no window can be called" in note for note in printed)
    _, document, _ = _run(arguments + ["--format", "json"], capsys)
    assert printed == json.loads(document)["qualifications"]


def test_count_json_reports_the_parameters_it_ran_under(dna_file, capsys):
    _, document, _ = _run(
        ["count", "-i", str(dna_file), "-k", "3", "--window", "32", "--format", "json"],
        capsys,
    )
    payload = json.loads(document)
    assert payload["parameters"]["k"] == 3
    assert payload["parameters"]["window"] == 32
    assert payload["parameters"]["measure"] == "dust"
    assert payload["source"] == "small.fasta"


def test_the_terminal_summary_can_be_asked_to_omit_its_bars(dna_file, capsys):
    _, with_bars, _ = _run(["count", "-i", str(dna_file)], capsys)
    _, without, _ = _run(["count", "-i", str(dna_file), "--no-bars"], capsys)
    assert "most abundant words" in with_bars
    assert "most abundant words" not in without


def test_the_threshold_caveat_does_not_depend_on_which_outputs_were_asked_for(
    tmp_path, capsys, examples
):
    # The regions are called from every window whatever the reporting options,
    # so a caveat about the threshold applies to a run, not to a rendering. It
    # used to be read off a kept window profile, which made it appear only
    # when an output happened to need one.
    base = [
        "profile",
        "-i",
        str(examples / "aav2.fasta"),
        "--window",
        "12",
        "--threshold",
        "20",
    ]
    status, printed, _ = _run(base, capsys)
    assert status == 0
    assert "! low-complexity calls: no window can be called" in printed

    status, _, _ = _run(
        base + ["-o", str(tmp_path / "windows.tsv"), "--format", "none"], capsys
    )
    assert status == 0

    status, document, _ = _run(base + ["--format", "json"], capsys)
    assert status == 0
    carried = json.loads(document)["qualifications"]
    assert any("no window can be called" in note for note in carried)


def test_count_prints_the_threshold_caveat_with_no_window_profile_kept(
    capsys, examples
):
    status, printed, _ = _run(
        [
            "count",
            "-i",
            str(examples / "aav2.fasta"),
            "--window",
            "12",
            "--threshold",
            "20",
            "--tracks",
            "0",
            "--no-bars",
        ],
        capsys,
    )
    assert status == 0
    assert "! low-complexity calls: no window can be called" in printed


def test_the_parser_offers_the_three_subcommands_under_the_tool_name():
    parser = cli.build_parser()
    assert parser.prog == "kmer-counter"
    actions = [
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    ]
    assert len(actions) == 1
    assert sorted(actions[0].choices) == ["compare", "count", "profile"]


def test_two_outputs_cannot_name_the_same_file(tmp_path, capsys, examples):
    # Writing both to one path leaves whichever is written last and silently
    # loses the other, which is data loss a preflight can prevent.
    target = tmp_path / "result"
    _fails(
        [
            "count",
            "-i",
            str(examples / "aav2.fasta"),
            "-o",
            str(target),
            "--report",
            str(target),
            "-f",
            "none",
        ],
        capsys,
        "two outputs name the same file",
    )
    assert not target.exists()


def test_two_spellings_of_one_output_path_are_still_one_file(tmp_path, capsys, examples):
    target = tmp_path / "result.tsv"
    _fails(
        [
            "count",
            "-i",
            str(examples / "aav2.fasta"),
            "-o",
            str(target),
            "--report",
            str(tmp_path / "." / "result.tsv"),
            "-f",
            "none",
        ],
        capsys,
        "two outputs name the same file",
    )
    assert not target.exists()


def test_a_generated_report_name_cannot_overwrite_another_input(tmp_path, capsys):
    # With several inputs the report name is generated, so it is the one name
    # the caller never typed and never checked. --report out.html beside an
    # input called out.aav2.fasta.html is the collision.
    first = tmp_path / "aav2.fasta"
    first.write_text(">a\nACGTACGTACGTACGT\n", encoding="utf-8")
    victim = tmp_path / "out.second.fasta.html"
    victim.write_text(">b\nACGTACGTACGTACGT\n", encoding="utf-8")
    second = tmp_path / "second.fasta"
    second.write_text(">b\nACGTACGTACGTACGT\n", encoding="utf-8")

    before = victim.read_text()
    _fails(
        [
            "count",
            "-i",
            str(first),
            "-i",
            str(second),
            "-i",
            str(victim),
            "--report",
            str(tmp_path / "out.html"),
            "-f",
            "none",
        ],
        capsys,
        "would truncate what is being read",
    )
    assert victim.read_text() == before


def test_a_generated_report_name_cannot_collide_with_another_report(tmp_path, capsys):
    # Two inputs whose names differ only outside [\w.-] are sanitised to the
    # same report name, which would leave one report for two inputs.
    first = tmp_path / "a b.fasta"
    second = tmp_path / "a+b.fasta"
    for path in (first, second):
        path.write_text(">a\nACGTACGTACGTACGT\n", encoding="utf-8")
    _fails(
        [
            "count",
            "-i",
            str(first),
            "-i",
            str(second),
            "--report",
            str(tmp_path / "out.html"),
            "-f",
            "none",
        ],
        capsys,
        "two outputs name the same file",
    )


def test_the_json_carries_the_composition_tables_it_documents(capsys, examples):
    status, document, _ = _run(
        ["count", "-i", str(examples / "aav2.fasta"), "-k", "3", "-f", "json"], capsys
    )
    assert status == 0
    payload = json.loads(document)
    assert sorted(payload["tables"]) == ["1", "2", "3"]
    assert payload["tables_omitted"] == []
    triples = payload["tables"]["3"]["words"]
    assert len(triples) == 64
    assert {"word", "count", "expected", "ratio", "z_score", "model"} <= set(triples[0])
    assert all(row["model"] == "zero-order" for row in triples)


def test_a_table_too_large_for_the_json_is_named_rather_than_dropped(capsys, examples):
    # Protein triples are 8,000 rows, past what a summary document should
    # carry; the omission is listed so a reader is not left guessing.
    status, document, _ = _run(
        ["count", "-i", str(examples / "aav2_vp1.fasta"), "-k", "3", "-f", "json"],
        capsys,
    )
    assert status == 0
    payload = json.loads(document)
    assert payload["tables_omitted"] == [3]
    assert sorted(payload["tables"]) == ["1", "2"]
