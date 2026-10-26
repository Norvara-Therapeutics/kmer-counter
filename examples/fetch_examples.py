"""Re-fetch the example sequences from NCBI.

Every example in this directory is a real database record, downloaded
unmodified. None of it is simulated, and nothing is sliced or edited, so a
figure computed from one of these files can be checked against the public
record.

    python examples/fetch_examples.py

Set a contact address in the ``NCBI_EMAIL`` environment variable if you want
NCBI to be able to reach you about the requests; E-utilities works without one.
"""

from __future__ import annotations

import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

#: ``filename -> (database, accession, what it is, expected length)``. The
#: lengths are asserted after download, so a silently truncated or superseded
#: record is caught rather than quietly replacing a known example.
RECORDS: dict[str, tuple[str, str, str, int]] = {
    "aav2.fasta": (
        "nuccore",
        "NC_001401.2",
        "adeno-associated virus 2, a compact GC-rich viral genome; its "
        "terminal repeats are inverted, which the perfect-tandem search here "
        "does not look for",
        4_679,
    ),
    "phix174.fasta": (
        "nuccore",
        "NC_001422.1",
        "Escherichia phage phiX174, a compact genome with little repetition",
        5_386,
    ),
    "aav2_vp1.fasta": (
        "protein",
        "YP_680426.1",
        "AAV2 major coat protein VP1",
        735,
    ),
    "huntingtin.fasta": (
        "protein",
        "NP_002102.4",
        "human huntingtin isoform 2, whose polyglutamine and polyproline "
        "tracts are what protein low-complexity calling is for",
        3_144,
    ),
}

EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def fetch(database: str, accession: str, email: str | None) -> str:
    """Download one record as FASTA."""
    query = {
        "db": database,
        "id": accession,
        "rettype": "fasta",
        "retmode": "text",
        "tool": "kmer-counter-examples",
    }
    if email:
        query["email"] = email
    url = f"{EFETCH}?{urllib.parse.urlencode(query)}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8")


def main() -> int:
    email = os.environ.get("NCBI_EMAIL") or None
    directory = Path(__file__).parent
    failures = 0

    for filename, (database, accession, description, expected) in RECORDS.items():
        try:
            text = fetch(database, accession, email)
        except OSError as error:
            print(f"{filename}: download failed: {error}", file=sys.stderr)
            failures += 1
            continue
        sequence = "".join(line.strip() for line in text.splitlines()[1:])
        if len(sequence) != expected:
            print(
                f"{filename}: {accession} is {len(sequence):,} residues, expected "
                f"{expected:,}; the record may have been revised",
                file=sys.stderr,
            )
            failures += 1
            continue
        (directory / filename).write_text(text, encoding="utf-8")
        print(f"{filename}: {accession}, {len(sequence):,} residues, {description}")
        time.sleep(0.4)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
