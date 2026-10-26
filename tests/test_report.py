"""The HTML dashboard: self-contained, inert, and the same bytes every time.

The document has to open from a local disk, an email attachment or a
regulatory submission years from now, which means it must fetch nothing, run
nothing, and render identically on every pass. These tests hold it to that:
every inline SVG is well-formed XML, no script element and no http reference
appears anywhere, the title names its subject, the caveats the terminal prints
are in the body, and each embedded data URI decodes to exactly the text
``rows.render`` would have written to a file - which is the guarantee that a
table downloaded from the report is the table the CLI would write.
"""

import base64
import re
import xml.etree.ElementTree as ElementTree

import pytest

from kmer_counter import report, rows
from kmer_counter.analysis import analyse
from kmer_counter.report import render_html

#: Every inline chart, as a complete element. The charts are written as single
#: lines, but DOTALL costs nothing and survives a reflow.
_SVG = re.compile(r"<svg\b.*?</svg>", re.DOTALL)

#: Each download link, as (filename, base64 payload).
_DOWNLOAD = re.compile(
    r'<a download="([^"]+)" '
    r'href="data:text/tab-separated-values;base64,([^"]+)"'
)

_PARAMETERS = {
    "Input": "example",
    "Word length": "k=4, counted from 1",
    "kmer-counter": "1.0.0test",
}


@pytest.fixture(scope="module")
def nucleotide(examples):
    """The AAV2 genome: a single DNA record with no low-complexity region."""
    return analyse(str(examples / "aav2.fasta"), k=4)


@pytest.fixture(scope="module")
def protein(examples):
    """Huntingtin: a protein record with regions, repeats and caveats."""
    return analyse(str(examples / "huntingtin.fasta"), k=4)


@pytest.fixture(scope="module")
def nucleotide_html(nucleotide):
    return render_html(nucleotide, _PARAMETERS, "1.0.0test")


@pytest.fixture(scope="module")
def protein_html(protein):
    return render_html(protein, _PARAMETERS, "1.0.0test")


def _documents(nucleotide_html, protein_html):
    return {"nucleotide": nucleotide_html, "protein": protein_html}


def test_every_svg_fragment_parses_as_xml(nucleotide_html, protein_html):
    for kind, document in _documents(nucleotide_html, protein_html).items():
        fragments = _SVG.findall(document)
        assert fragments, f"{kind} report drew no charts"
        for fragment in fragments:
            # Raises on an unbalanced tag, an unescaped '&' or a bad
            # attribute, any of which would break the chart in a browser.
            element = ElementTree.fromstring(fragment)
            assert element.tag.endswith("svg")


def test_the_dashboard_runs_no_script(nucleotide_html, protein_html):
    for document in _documents(nucleotide_html, protein_html).values():
        assert "<script" not in document.lower()
        assert "javascript:" not in document.lower()
        assert "onclick" not in document.lower()


def test_the_dashboard_fetches_nothing_from_the_network(
    nucleotide_html, protein_html
):
    # The SVG namespace declaration is an http URI and is not a fetch; a src
    # or href pointing at http is.
    for document in _documents(nucleotide_html, protein_html).values():
        assert 'src="http' not in document
        assert 'href="http' not in document
        assert "<img" not in document
        assert "@import" not in document


def test_the_title_and_the_heading_name_the_input(nucleotide_html, protein_html):
    assert "<title>Word composition of aav2.fasta</title>" in nucleotide_html
    assert "<h1>Word composition of aav2.fasta</h1>" in nucleotide_html
    assert "<title>Word composition of huntingtin.fasta</title>" in protein_html


def test_every_qualification_reaches_the_body(
    nucleotide, protein, nucleotide_html, protein_html
):
    # A caveat shown in the terminal and absent from the report is worse than
    # no caveat at all, so each one is in the document verbatim.
    for analysis, document in ((nucleotide, nucleotide_html), (protein, protein_html)):
        assert analysis.qualifications
        for note in analysis.qualifications:
            assert note in document
        assert '<div class="caveat">' in document


def test_the_parameters_it_is_given_are_shown(nucleotide_html):
    for name, value in _PARAMETERS.items():
        assert f"<dt>{name}</dt><dd>{value}</dd>" in nucleotide_html


def test_each_embedded_table_decodes_to_what_rows_render_writes(
    nucleotide, protein, nucleotide_html, protein_html
):
    for analysis, document in ((nucleotide, nucleotide_html), (protein, protein_html)):
        expected = {
            f"spectrum-k{analysis.k}.tsv": rows.render(
                rows.SPECTRUM_COLUMNS, rows.spectrum_rows(analysis)
            ),
            "complexity.tsv": rows.render(
                rows.ENTROPY_COLUMNS, rows.entropy_rows(analysis)
            ),
            "windows.tsv": rows.render(
                rows.WINDOW_COLUMNS, rows.window_rows(analysis.tracks)
            ),
            "regions.tsv": rows.render(
                rows.REGION_COLUMNS, rows.region_rows(analysis)
            ),
            "residues.tsv": rows.render(
                rows.COMPOSITION_COLUMNS, rows.composition_rows(analysis.tables[1])
            ),
            "pairs.tsv": rows.render(
                rows.COMPOSITION_COLUMNS, rows.composition_rows(analysis.tables[2])
            ),
            f"triples-{analysis.alphabet.name}.tsv": rows.render(
                rows.COMPOSITION_COLUMNS, rows.composition_rows(analysis.tables[3])
            ),
            "repeats.tsv": rows.render(
                rows.REPEAT_COLUMNS, rows.repeat_rows(analysis)
            ),
            "records.tsv": rows.render(
                rows.RECORD_COLUMNS, rows.record_rows(analysis)
            ),
        }
        embedded = _DOWNLOAD.findall(document)
        assert embedded
        for filename, payload in embedded:
            assert filename in expected, f"unexpected download {filename}"
            decoded = base64.b64decode(payload).decode("utf-8")
            assert decoded == expected[filename]


def test_the_spectrum_complexity_and_record_tables_are_always_offered(
    nucleotide_html, protein_html
):
    for document in _documents(nucleotide_html, protein_html).values():
        offered = {filename for filename, _ in _DOWNLOAD.findall(document)}
        assert {"spectrum-k4.tsv", "complexity.tsv", "records.tsv"} <= offered


def test_a_download_link_says_how_many_rows_it_carries(nucleotide, nucleotide_html):
    # The spectrum download is annotated with its row count, which is the
    # number of distinct words.
    expected = f"{nucleotide.spectrum.distinct:,} rows"
    assert expected in nucleotide_html


def test_an_input_without_a_region_says_so_rather_than_showing_an_empty_table(
    nucleotide, nucleotide_html
):
    # AAV2 reaches the DUST threshold nowhere, and an empty table would read
    # as a missing measurement rather than as a negative result.
    assert not rows.region_rows(nucleotide)
    assert "no region was called" in nucleotide_html
    assert "regions.tsv" not in nucleotide_html


def test_a_protein_report_names_the_measure_its_track_is_in(
    nucleotide_html, protein_html
):
    # The two alphabets are scored by different measures running in opposite
    # directions, so the axis label of the track has to say which one is
    # plotted. ('DUST score' alone appears in both footers, which cite it.)
    assert "DUST score \u00b7 position" in nucleotide_html
    assert "residue entropy, bits \u00b7 position" in protein_html
    assert "DUST score \u00b7 position" not in protein_html
    # A nucleotide track is drawn twice, once for the score and once for the
    # base entropy that only a nucleotide window has.
    assert "base entropy, bits \u00b7 position" in nucleotide_html
    assert "base entropy, bits \u00b7 position" not in protein_html


def test_rendering_the_same_analysis_twice_gives_identical_text(
    nucleotide, protein, nucleotide_html, protein_html
):
    # Nothing in the document may depend on the clock, the filesystem or the
    # iteration order of a set.
    assert render_html(nucleotide, _PARAMETERS, "1.0.0test") == nucleotide_html
    assert render_html(protein, _PARAMETERS, "1.0.0test") == protein_html


def test_the_document_is_a_complete_html_page(nucleotide_html):
    assert nucleotide_html.startswith("<!DOCTYPE html>\n<html lang=\"en\">")
    assert nucleotide_html.rstrip("\n").endswith("</html>")
    assert '<meta charset="utf-8">' in nucleotide_html
    # The style is inline, so the page needs no stylesheet beside it.
    assert "<style>" in nucleotide_html
    assert "<link" not in nucleotide_html


def test_a_table_too_large_to_embed_says_so_instead_of_embedding_it(monkeypatch, examples):
    # A document carrying a megabyte table is one no mail server will deliver;
    # the report says where to get the file rather than silently omitting it.
    monkeypatch.setattr(report, "MAX_EMBEDDED", 200)
    analysis = analyse(str(examples / "aav2.fasta"), k=4)
    document = report.render_html(analysis, {"Input": "aav2.fasta"}, "test")

    assert "too large to embed" in document
    assert "<code>--output</code>" in document
    assert "data:text/tab-separated-values" not in document


def test_the_report_states_the_periods_that_were_actually_searched(examples):
    # The wording used to be fixed at six residues whatever --max-period was,
    # which makes the report say something untrue about its own run.
    analysis = analyse(str(examples / "aav2.fasta"), k=4, min_run=7, max_period=3)
    document = render_html(analysis, {"Input": "aav2.fasta"}, "test")
    assert "unit of up to 3 residues" in document
    assert "stretch of 7 residues or more" in document
    assert "up to six residues" not in document


def test_the_report_shows_the_repeat_share_against_the_floor(examples):
    # 4,674 windows over 4,096 six-mers force 12.4% of them into a repeated
    # word; the observed share means nothing without that floor beside it.
    analysis = analyse(str(examples / "aav2.fasta"), k=6)
    document = render_html(analysis, {"Input": "aav2.fasta"}, "test")
    assert "against a floor of 12.39%" in document
