"""Suite-wide guards.

The targeted re-fetch (`refetch_behind`) sleeps REFETCH_DELAY_S and calls
Yahoo. A test whose synthetic universe happens to be >20% behind (two tickers,
one stale, is 50%) would otherwise sleep 60 s and hit the network - which is
exactly what happened the first time the suite ran after it landed (61 s run).
Inert by default, so a new test cannot reach the network by accident; a test
that exercises the re-fetch substitutes its own `refetch_recent`.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import sigma_screener as _ss  # noqa: E402


@pytest.fixture(autouse=True)
def _refetch_is_inert(monkeypatch):
    def _no_network(symbols):
        return None
    monkeypatch.setattr(_ss, "refetch_recent", _no_network)
    monkeypatch.setattr(_ss, "REFETCH_DELAY_S", 0)
