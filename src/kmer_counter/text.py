"""Number formatting shared by the terminal summary and the HTML report.

Only human-facing output is formatted here. The TSV and JSON outputs carry
unrounded values, so a share that is not quite 100% is never written as 100%
to a file another program will read, and a ratio of 0.996 is never written as
1.00 to one.
"""

from __future__ import annotations

from kmer_counter.alphabet import Alphabet


def thousands(value: int) -> str:
    """Group an integer in thousands."""
    return f"{value:,}"


def plural(count: int, singular: str, suffix: str = "s") -> str:
    """Count something, agreeing the noun with the number."""
    return f"{thousands(count)} {singular}{'' if count == 1 else suffix}"


def number(value: float | None, places: int = 2, missing: str = "n/a") -> str:
    """Format an optional number to a fixed number of decimal places."""
    if value is None:
        return missing
    return f"{value:,.{places}f}"


def percent(value: float | None, places: int = 2, missing: str = "n/a") -> str:
    """Format a percentage, never rounding a near-miss to 0% or 100%.

    A share of 99.9986% is written '>99.99%' rather than '100.00%', because a
    report that claims every window of a sequence is low-complexity when one
    is not is the kind of claim the reader will act on.
    """
    if value is None:
        return missing
    rounded = round(value, places)
    smallest = 10.0**-places
    if value > 0 and rounded == 0:
        return f"<{smallest:.{places}f}%"
    if value < 100 and rounded == 100:
        return f">{100 - smallest:.{places}f}%"
    if value > 100 and rounded == 100:
        return f"<{100 + smallest:.{places}f}%"
    return f"{value:.{places}f}%"


def share(value: float | None, places: int = 2, missing: str = "n/a") -> str:
    """Format a share held as a fraction as a percentage."""
    if value is None:
        return missing
    return percent(100 * value, places, missing)


def ratio(value: float | None, places: int = 2, missing: str = "n/a") -> str:
    """Format an observed-over-expected ratio, never rounding a near-miss to 1.

    A dinucleotide at 0.996 of its expectation is not at its expectation, and
    a reader comparing a column of ratios to 1.00 will not look twice at one
    written as 1.00.
    """
    if value is None:
        return missing
    rounded = round(value, places)
    smallest = 10.0**-places
    if value != 1.0 and rounded == 1.0:
        # Below one the value sits above the next bound down, so it is written
        # '>0.99'; above one it sits below the next bound up, '<1.01'. Getting
        # this backwards prints a bound the value does not satisfy.
        if value < 1.0:
            return f">{1.0 - smallest:.{places}f}"
        return f"<{1.0 + smallest:.{places}f}"
    if value > 0 and rounded == 0:
        return f"<{smallest:.{places}f}"
    return f"{value:.{places}f}"


def bits(value: float | None, places: int = 2, missing: str = "n/a") -> str:
    """Format an entropy in bits."""
    if value is None:
        return missing
    return f"{value:.{places}f} bits"


def residues(count: int, alphabet: Alphabet) -> str:
    """Write a length in the alphabet's own unit.

    Nucleotide lengths are in bases, protein lengths in residues, because a
    protein length written 'bp' is the kind of slip that makes a reader doubt
    everything else in the table.
    """
    unit = "bp" if alphabet.is_nucleotide else "aa"
    return f"{thousands(count)} {unit}"


def position_range(start: int, end: int) -> str:
    """Write an inclusive 1-based interval."""
    return f"{thousands(start)}-{thousands(end)}"


def bar(count: float, widest: float, width: int = 24, mark: str = "#") -> str:
    """A proportional text bar, one mark wide at minimum for a non-zero value."""
    if count <= 0 or widest <= 0:
        return ""
    return mark * max(1, round(width * count / widest))


def truncate(text: str, width: int = 40) -> str:
    """Shorten a long identifier for a fixed-width column."""
    if len(text) <= width:
        return text
    return text[: width - 1] + "\u2026"
