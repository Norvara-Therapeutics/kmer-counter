"""Shared fixtures."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def examples() -> Path:
    """Directory holding the bundled example sequences."""
    directory = Path(__file__).resolve().parent.parent / "examples"
    if not directory.is_dir():
        pytest.skip("example sequences are not present in this build")
    return directory
