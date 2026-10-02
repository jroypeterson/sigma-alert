"""Suite-wide guards.

The targeted re-fetch (`refetch_behind`) sleeps REFETCH_DELAY_S and calls
Yahoo. A test whose synthetic universe happens to be >20% behind (two tickers,
one stale, is 50%) would otherwise sleep 60 s and hit the network - which is
exactly what happened the first time the suite ran after it landed (61 s run).
Inert by default, so a new test cannot reach the network by accident; a test
that exercises the re-fetch substitutes its own `refetch_recent`.

The FMP fallback (`fmp_fallback`, close mode, after the re-fetch) is inert the
same way: `FMP_API_KEY` is removed (a developer shell or CI may carry the real
key) and all three network functions (`fmp_get_quote`, `fmp_get_eod_price`,
`fmp_corporate_action_symbols`) are replaced by stubs that return nothing - so even a test that sets a key cannot
reach FMP unless it substitutes its own quote function.
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


@pytest.fixture(autouse=True)
def _fmp_is_inert(monkeypatch):
    def _no_network(symbol, key):
        return None
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    monkeypatch.setattr(_ss, "fmp_get_quote", _no_network)
    monkeypatch.setattr(_ss, "fmp_get_eod_price", lambda symbol, key, day: None)
    monkeypatch.setattr(_ss, "fmp_corporate_action_symbols", lambda key, day: set())
    monkeypatch.setattr(_ss, "FMP_MIN_INTERVAL_S", 0)


@pytest.fixture(autouse=True)
def _cycle_stamp_is_redirected(monkeypatch, tmp_path_factory):
    """Every test that drives `main()` reaches a cycle-stamp exit (board #445).
    Without this the suite would rewrite the real `readable/return_map.cycles.json`
    on every run - the 'test suite is a pipeline run' failure."""
    import return_map
    monkeypatch.setattr(return_map, "CYCLES_PATH",
                        tmp_path_factory.mktemp("cycles") / "return_map.cycles.json")
