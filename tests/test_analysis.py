"""Measuring one input end to end: what is decided once, and what is kept.

Three decisions are protected here. The alphabet is decided once, from enough
residues to be worth deciding from, and applied to every record. Which records
keep a window profile changes the size of the output and nothing else: the
regions, the per-record figures and the qualifications are the same either
way. And every caveat belongs to exactly one of three groups, so a renderer
can show the ones that apply to it without inventing or dropping any.
"""

import io

import pytest

from kmer_counter.alphabet import DNA, PROTEIN
from kmer_counter.analysis import (
    ALL_TRACKS,
    DEFAULT_K,
    DEFAULT_TRACKS,
    TABLE_LENGTHS,
    analyse,
    check_residues,
    resolve_alphabet,
)
from kmer_counter.reader import Record, SequenceError


def _fasta(*records):
    return io.StringIO("".join(f">{name}\n{sequence}\n" for name, sequence in records))


def test_the_alphabet_can_be_named_instead_of_detected():
    records = [Record("r", "", "ACGTACGTAC")]
    alphabet, detection = resolve_alphabet(records, "protein")
    assert alphabet.name == "protein"
    assert detection is None


def test_a_named_alphabet_that_cannot_accept_the_residues_is_refused():
    records = [Record("r", "", "ACGTACGT")]
    with pytest.raises(SequenceError, match="does not accept"):
        resolve_alphabet([Record("r", "", "ACGT123")], "dna")
    # A nucleotide file holding a protein-only residue is the real case.
    with pytest.raises(SequenceError, match=r"'E'"):
        resolve_alphabet([Record("r", "", "ACGTEEEE")], "dna")
    assert resolve_alphabet(records, "dna")[0].name == "dna"


def test_the_refusal_says_how_to_name_the_alphabet_when_it_was_detected():
    with pytest.raises(SequenceError, match=r"--alphabet"):
        resolve_alphabet([Record("r", "", "ACGT*")], "auto")


def test_the_alphabet_is_decided_from_enough_residues_to_judge_from():
    # The first record is two residues long; deciding from it alone would call
    # a 1,200-base DNA file protein on the strength of 'MA'.
    analysis = analyse(_fasta(("short", "MA"), ("long", "ACGT" * 300)))
    assert analysis.alphabet.name == "dna"
    assert analysis.detection is not None
    assert analysis.detection.certain is True
    assert analysis.alphabet_notes == []


def test_a_short_input_says_the_alphabet_is_weakly_supported():
    analysis = analyse(_fasta(("tiny", "ACGTAC")))
    assert analysis.detection.certain is False
    assert "alphabet read as DNA" in analysis.alphabet_notes[0]
    assert "6 residues" in analysis.alphabet_notes[0]


def test_every_record_is_measured_under_the_one_alphabet():
    analysis = analyse(_fasta(("a", "ACGTACGTAC"), ("b", "ACGTACGTAC" * 2)))
    assert len(analysis.records) == 2
    assert analysis.length == 30
    assert analysis.spectra[1].sequences == 2
    # No window spans two records, so the windows are one short per record.
    assert analysis.spectra[2].windows == 9 + 19


def test_keeping_a_track_changes_the_output_and_not_the_findings():
    records = [(f"r{index}", "ACGTTTTTTTTTTTTTTTTTTTTACGT") for index in range(6)]
    kept = analyse(_fasta(*records), window=12, threshold=3.0, tracks=2)
    none = analyse(_fasta(*records), window=12, threshold=3.0, tracks=0)
    every = analyse(_fasta(*records), window=12, threshold=3.0, tracks=ALL_TRACKS)

    assert [len(kept.tracks), len(none.tracks), len(every.tracks)] == [2, 0, 6]
    assert kept.regions == none.regions == every.regions > 0
    assert kept.masked == none.masked == every.masked

    def calls(analysis):
        return [
            note for note in analysis.profile_notes
            if note.startswith("low-complexity calls")
        ]

    assert calls(kept) == calls(none) == calls(every) == []

    # And the caveat itself, when there is one, reaches a run that kept no
    # window profile at all: it belongs to the settings, not to a rendering.
    unreachable = [
        analyse(_fasta(*records), window=12, threshold=20.0, tracks=count)
        for count in (0, 2, ALL_TRACKS)
    ]
    assert all(
        calls(analysis) and "no window can be called" in calls(analysis)[0]
        for analysis in unreachable
    )


def test_capping_the_tracks_says_so_and_keeping_them_all_does_not():
    records = [(f"r{index}", "ACGTACGTACGTACGT") for index in range(6)]
    capped = analyse(_fasta(*records), tracks=2)
    every = analyse(_fasta(*records), tracks=ALL_TRACKS)
    assert any("window profiles are kept for the first 2" in note for note in capped.profile_notes)
    assert not any("window profiles are kept" in note for note in every.profile_notes)


def test_a_negative_track_count_other_than_every_record_is_refused():
    with pytest.raises(ValueError, match="or -1 for every record"):
        analyse(_fasta(("a", "ACGTACGT")), tracks=-2)


def test_the_repeat_search_can_be_skipped_without_losing_the_regions():
    sequence = ("ACGT" * 4) + ("A" * 30) + ("ACGT" * 4)
    with_repeats = analyse(_fasta(("a", sequence)), window=12, threshold=3.0)
    without = analyse(
        _fasta(("a", sequence)), window=12, threshold=3.0, find_repeats=False
    )
    assert with_repeats.records[0].runs
    assert without.records[0].runs == ()
    assert without.records[0].tandems == ()
    assert without.regions == with_repeats.regions > 0


def test_counting_canonically_needs_a_complement():
    with pytest.raises(ValueError, match="canonical counting needs a complement"):
        analyse(_fasta(("p", "MKVLAAGIVGLNLGGMKVLAAG")), canonical=True)


def test_counting_canonically_leaves_no_composition_tables():
    plain = analyse(_fasta(("a", "ACGTACGTACGTAAGG")))
    canonical = analyse(_fasta(("a", "ACGTACGTACGTAAGG")), canonical=True)
    assert sorted(plain.tables) == [length for length in TABLE_LENGTHS]
    assert canonical.tables == {}
    assert canonical.signature == {}


def test_the_record_limit_stops_the_read():
    analysis = analyse(_fasta(*[(f"r{index}", "ACGTACGTAC") for index in range(10)]), limit=3)
    assert len(analysis.records) == 3
    assert analysis.length == 30


def test_an_input_with_no_records_is_refused():
    with pytest.raises(SequenceError):
        analyse(io.StringIO(""))


def test_the_default_word_length_counts_every_length_below_it():
    analysis = analyse(_fasta(("a", "ACGTACGTACGTACGT")))
    assert analysis.k == DEFAULT_K
    assert sorted(analysis.spectra) == [1, 2, 3, 4]
    assert analysis.spectrum is analysis.spectra[DEFAULT_K]


def test_the_qualifications_are_the_three_groups_and_nothing_else():
    analysis = analyse(_fasta(("a", "ACGTNNNNACGTACGTACGTAC"), ("b", "ACGTACGTACGT")))
    assert analysis.qualifications == (
        analysis.alphabet_notes + analysis.spectrum_notes + analysis.profile_notes
    )
    assert len(analysis.qualifications) == len(set(analysis.qualifications))


def test_an_input_too_short_for_its_word_length_says_so_once():
    analysis = analyse(_fasta(("a", "AC")), k=8)
    assert analysis.spectrum.windows == 0
    assert len(analysis.spectrum_notes) == 1
    assert "no window of length 8 could be counted" in analysis.spectrum_notes[0]


def test_positions_are_reported_for_one_record_and_withheld_for_several():
    one = analyse(_fasta(("a", "ACGTACGTACGTAC")), k=2, abundant=3)
    several = analyse(_fasta(("a", "ACGTACGTACGTAC"), ("b", "ACGTACGT")), k=2, abundant=3)
    assert one.positions_reported is True
    assert one.abundant[0].positions
    assert several.positions_reported is False
    assert several.abundant[0].positions == ()
    assert any("without positions" in note for note in several.spectrum_notes)


def test_uncounted_residues_are_classified_rather_than_ignored():
    analysis = analyse(_fasta(("a", "ACGTNNACGT-ACGT")), alphabet="dna", k=2)
    record = analysis.records[0]
    assert record.length == 15
    assert record.countable == 12
    assert record.gaps == 1
    assert record.ambiguous == 2
    assert record.stops == 0
    assert analysis.ambiguous == 2

    # A stop is protein punctuation; a nucleotide file holding one is refused
    # rather than counted as an uncountable residue.
    with pytest.raises(SequenceError, match=r"does not accept: '\*'"):
        analyse(_fasta(("a", "ACGTACGT*")), alphabet="dna")


def test_the_default_track_count_is_the_documented_one():
    records = [(f"r{index}", "ACGTACGTACGTACGT") for index in range(DEFAULT_TRACKS + 2)]
    analysis = analyse(_fasta(*records))
    assert len(analysis.tracks) == DEFAULT_TRACKS


def test_every_record_is_checked_against_the_resolved_alphabet():
    # The alphabet is settled from the first records; a later record holding a
    # residue it does not accept is an error, not an ambiguity code. Counting
    # that uracil as uncountable would be a wrong answer dressed as a caveat.
    mixed = _fasta(("a", "ACGT" * 10), ("b", "ACGUACGUACGU"))
    with pytest.raises(SequenceError, match=r"record b: .*does not accept: 'U'"):
        analyse(mixed)


def test_a_later_record_is_checked_against_an_alphabet_that_was_named_too():
    mixed = _fasta(("a", "ACGTACGTAC"), ("b", "ACGTEEEE"))
    with pytest.raises(SequenceError, match=r"record b: .*does not accept: 'E'"):
        analyse(mixed, alphabet="dna")


def test_a_record_the_alphabet_accepts_is_still_counted_as_ambiguous():
    # The check is about residues outside the alphabet, not about ambiguity
    # codes, which are accepted and skipped as documented.
    analysis = analyse(_fasta(("a", "ACGT" * 10), ("b", "ACGTNNNNACGT")))
    assert analysis.ambiguous == 4
    assert len(analysis.records) == 2


def test_the_saturation_note_quotes_the_floor_rather_than_overstating_it(examples):
    analysis = analyse(str(examples / "aav2.fasta"), k=6)
    note = next(note for note in analysis.spectrum_notes if "possible words" in note)
    assert "at least 579 of them (12.4%)" in note
    assert "almost every word" not in note


def test_the_repeat_parameters_travel_with_the_result():
    analysis = analyse(_fasta(("a", "ACGTACGTACGT")), min_run=7, max_period=3)
    assert (analysis.min_run, analysis.max_period) == (7, 3)


def test_checking_one_record_names_the_record_and_the_residues():
    # The direct contract, since this is what every record is put through:
    # the record it was, the residues it held, and how to say the alphabet
    # was read wrongly.
    good = Record("fine", "", "ACGTACGT")
    check_residues(good, DNA)

    with pytest.raises(SequenceError) as refusal:
        check_residues(Record("bad", "a description", "ACGTEFIL"), DNA)
    message = str(refusal.value)
    assert message.startswith("record bad: the DNA alphabet does not accept: ")
    assert "'E', 'F', 'I', 'L'" in message
    assert "--alphabet" in message


def test_a_named_alphabet_is_not_told_to_name_itself_again():
    with pytest.raises(SequenceError, match="check the input, or the alphabet you named"):
        check_residues(Record("bad", "", "ACGTU"), DNA, requested="dna")


def test_only_the_first_few_offending_residues_are_listed():
    with pytest.raises(SequenceError, match=r"\.\.\."):
        check_residues(Record("bad", "", "EFILMPQ"), DNA)


def test_ambiguity_codes_and_gaps_are_accepted_by_the_check():
    check_residues(Record("fine", "", "ACGTNRYSWKMBDHV-.~"), DNA)
    check_residues(Record("fine", "", "ACDEFGHIKLMNPQRSTVWYXBZJUO*-"), PROTEIN)
