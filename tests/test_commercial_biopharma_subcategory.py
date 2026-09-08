"""The Commercial Biopharma digest bucket.

⛑ WHY IT EXISTS. A 2-sigma alert whose ticker matches NO subcategory is DROPPED
at render time. So a commercial biopharma name that is not Core, not held, not
`Large Pharma` and not in the S&P 500 never appeared in the digest at all. JP:
"I would like this broader list of 121 as names that something like sigma-alert
... would pick up."
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from sigma_screener import SUBCATEGORIES

SP500 = {"LLY"}


def _pred(label):
    for name, p in SUBCATEGORIES:
        if name == label:
            return p
    raise AssertionError("no %r subcategory" % label)


def _a(**kw):
    base = {"ticker": "X", "sector": "Biopharma", "subsector": "Biotech",
            "commercial": ""}
    base.update(kw)
    return base


def test_a_commercial_name_is_picked_up():
    assert _pred("Commercial Biopharma")(_a(ticker="ALNY", commercial="Y"), SP500)


def test_a_non_commercial_biopharma_name_is_not():
    assert not _pred("Commercial Biopharma")(_a(ticker="TINY", commercial=""), SP500)


def test_large_pharma_is_excluded_because_it_renders_directly_above():
    """⛑ An alert is duplicated across EVERY category it matches, and Large
    Pharma is a strict subset of commercial by construction. Without this the
    digest repeats all 28 of those names back-to-back for no information."""
    a = _a(ticker="LLY", subsector="Large Pharma", commercial="Y")
    assert _pred("Large Pharma")(a, SP500)
    assert not _pred("Commercial Biopharma")(a, SP500), "LLY listed twice"


def test_the_bucket_would_have_rescued_an_otherwise_invisible_alert():
    """The whole point: this name matches no other subcategory, so before this
    change a 2-sigma move on it was computed and then silently dropped."""
    a = _a(ticker="ALKS", commercial="Y")
    others = [(n, p) for n, p in SUBCATEGORIES if n != "Commercial Biopharma"]
    assert not any(p(a, SP500) for _, p in others), (
        "fixture is already covered elsewhere; it proves nothing")
    assert any(p(a, SP500) for _, p in SUBCATEGORIES)


def test_the_flag_is_read_from_metadata_not_inferred_from_the_sector():
    """A blank flag on a Biopharma row means 'not commercial' (or unknown), and
    must not be rescued by the sector alone."""
    assert not _pred("Commercial Biopharma")(
        _a(ticker="PRECOM", sector="Biopharma", commercial=""), SP500)
